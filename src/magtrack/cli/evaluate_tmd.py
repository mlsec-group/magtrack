from __future__ import annotations

import datetime
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List

import duckdb
import numpy as np
import optuna
import pandas as pd
import typer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from magtrack.utils.evaluation import train_test_split, find_best_threshold_snr
from magtrack.utils.loader import read_pickle, get_metadata_from_dataset
from magtrack.utils.tmd_functions import (
    compute_fft_spectrum,
    compute_psd_spectrum,
    compute_snr_from_spectrum,
    compute_snr_goertzel,
)

# Discrete nperseg choices for the PSD search.  Spectra for each value are
# precomputed once before Optuna starts; the search picks among them as a
# Categorical parameter.
PSD_NPERSEG_CHOICES: tuple[int, ...] = (128, 192, 256, 384, 512)

app = typer.Typer(help="Hyperparameter search for FFT / PSD transport mode detectors.")

# ---------------------------------------------------------------------------
# Worker-process globals
# ---------------------------------------------------------------------------
# Each split's data must carry a "sample_id" column matching keys in the
# spectra caches below.  The caches map sample_id → (freqs, spectrum).
_optuna_train_data: pd.DataFrame | None = None
_optuna_test_data: pd.DataFrame | None = None
_optuna_all_splits: list | None = None
_optuna_metric: str = "mcc"

# In-memory spectrum caches, populated in the main process and inherited by
# fork()'d workers.  FFT cache: sample_id → (freqs, amplitude).  PSD cache:
# (sample_id, nperseg) → (freqs, psd).
_fft_spectra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
_psd_spectra: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] | None = None


# ---------------------------------------------------------------------------
# Spectrum cache (DuckDB next to the .pkl) + parallel precomputation
# ---------------------------------------------------------------------------

_FFT_TABLE = "fft_spectra"
_PSD_TABLE = "psd_spectra"


def _make_sample_id(row_id: str, chunk_data: pd.DataFrame) -> str:
    """Stable, content-derived identifier for one sample (id-chunk).

    Uses the recording id plus the first/last timestamp of the chunk and the
    sample count.  Two rows produced from the same source chunk always map to
    the same id; an unrelated chunk almost certainly differs in at least one
    of these fields.
    """
    ts = chunk_data["timestamp"].astype("int64").to_numpy()
    if ts.size == 0:
        return f"{row_id}#empty"
    return f"{row_id}#{int(ts[0])}#{int(ts[-1])}#{int(ts.size)}"


def _attach_sample_ids(data: pd.DataFrame) -> pd.DataFrame:
    sample_ids = [
        _make_sample_id(row.id, row.data) for row in data.itertuples(index=False)
    ]
    return data.assign(sample_id=sample_ids)


def _open_spectrum_cache(dataset_path: Path) -> tuple[duckdb.DuckDBPyConnection, Path]:
    cache_path = dataset_path.with_suffix(".cache.duckdb")
    con = duckdb.connect(str(cache_path))
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_FFT_TABLE} (
            sample_id TEXT NOT NULL,
            detrend   BOOLEAN NOT NULL,
            "window"  BOOLEAN NOT NULL,
            fs        DOUBLE NOT NULL,
            freqs     DOUBLE[] NOT NULL,
            amplitude DOUBLE[] NOT NULL,
            PRIMARY KEY (sample_id, detrend, "window")
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_PSD_TABLE} (
            sample_id TEXT NOT NULL,
            nperseg   INTEGER NOT NULL,
            fs        DOUBLE NOT NULL,
            freqs     DOUBLE[] NOT NULL,
            psd       DOUBLE[] NOT NULL,
            PRIMARY KEY (sample_id, nperseg)
        )
    """)
    n_fft = con.execute(f"SELECT COUNT(*) FROM {_FFT_TABLE}").fetchone()[0]
    n_psd = con.execute(f"SELECT COUNT(*) FROM {_PSD_TABLE}").fetchone()[0]
    typer.echo(f"  Spectrum cache: {cache_path}  (fft={n_fft:,}, psd={n_psd:,} entries)")
    return con, cache_path


def _get_cached_fft(
        con: duckdb.DuckDBPyConnection,
        sample_ids: list[str],
        detrend: bool,
        window: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not sample_ids:
        return {}
    qdf = pd.DataFrame({"sample_id": sample_ids})
    con.register("_qids", qdf)
    try:
        rows = con.execute(f"""
            SELECT s.sample_id, s.freqs, s.amplitude
            FROM {_FFT_TABLE} s
            INNER JOIN _qids q ON s.sample_id = q.sample_id
            WHERE s.detrend = ? AND s."window" = ?
        """, [bool(detrend), bool(window)]).fetchall()
    finally:
        con.unregister("_qids")
    return {
        r[0]: (np.asarray(r[1], dtype=np.float64), np.asarray(r[2], dtype=np.float64))
        for r in rows
    }


def _insert_fft(
        con: duckdb.DuckDBPyConnection,
        rows: list[tuple],
) -> None:
    """rows: list of (sample_id, detrend, window, fs, freqs_list, amplitude_list)."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["sample_id", "detrend", "window", "fs", "freqs", "amplitude"])
    con.register("_ir", df)
    try:
        con.execute(f"""
            INSERT INTO {_FFT_TABLE}
            SELECT sample_id, detrend, "window", fs, freqs, amplitude FROM _ir
            ON CONFLICT DO NOTHING
        """)
    finally:
        con.unregister("_ir")


def _get_cached_psd(
        con: duckdb.DuckDBPyConnection,
        sample_ids: list[str],
        nperseg: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not sample_ids:
        return {}
    qdf = pd.DataFrame({"sample_id": sample_ids})
    con.register("_qids", qdf)
    try:
        rows = con.execute(f"""
            SELECT s.sample_id, s.freqs, s.psd
            FROM {_PSD_TABLE} s
            INNER JOIN _qids q ON s.sample_id = q.sample_id
            WHERE s.nperseg = ?
        """, [int(nperseg)]).fetchall()
    finally:
        con.unregister("_qids")
    return {
        r[0]: (np.asarray(r[1], dtype=np.float64), np.asarray(r[2], dtype=np.float64))
        for r in rows
    }


def _insert_psd(
        con: duckdb.DuckDBPyConnection,
        rows: list[tuple],
) -> None:
    """rows: list of (sample_id, nperseg, fs, freqs_list, psd_list)."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["sample_id", "nperseg", "fs", "freqs", "psd"])
    con.register("_ir", df)
    try:
        con.execute(f"""
            INSERT INTO {_PSD_TABLE}
            SELECT sample_id, nperseg, fs, freqs, psd FROM _ir
            ON CONFLICT DO NOTHING
        """)
    finally:
        con.unregister("_ir")


def _fft_spectrum_worker(args: tuple) -> tuple:
    """Worker: compute one FFT spectrum.

    args: (sample_id, magnitude_arr, timestamp_arr_ns, detrend, window)
    returns: (sample_id, detrend, window, fs, freqs_list, amplitude_list)
    """
    sample_id, mag, ts_ns, detrend, window = args
    chunk = pd.DataFrame({
        "magnitude": mag,
        "timestamp": pd.to_datetime(ts_ns, unit="ns"),
    })
    freqs, amplitude, fs = compute_fft_spectrum(chunk, detrend=detrend, window=window)
    return sample_id, bool(detrend), bool(window), float(fs), freqs.tolist(), amplitude.tolist()


def _psd_spectrum_worker(args: tuple) -> tuple:
    """Worker: compute one PSD spectrum at a given nperseg.

    args: (sample_id, magnitude_arr, timestamp_arr_ns, nperseg)
    returns: (sample_id, nperseg, fs, freqs_list, psd_list)
    """
    sample_id, mag, ts_ns, nperseg = args
    chunk = pd.DataFrame({
        "magnitude": mag,
        "timestamp": pd.to_datetime(ts_ns, unit="ns"),
    })
    freqs, psd, fs = compute_psd_spectrum(chunk, nperseg=int(nperseg))
    return sample_id, int(nperseg), float(fs), freqs.tolist(), psd.tolist()


def _precompute_fft_spectra(
        data: pd.DataFrame,
        *,
        detrend: bool,
        window: bool,
        n_jobs: int,
        con: duckdb.DuckDBPyConnection,
        chunksize: int = 32,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sample_ids = list(data["sample_id"].drop_duplicates())
    cached = _get_cached_fft(con, sample_ids, detrend, window)
    missing_ids = [sid for sid in sample_ids if sid not in cached]

    if missing_ids:
        # Build per-id payload from the first occurrence of each sample_id
        first_idx = data.drop_duplicates(subset="sample_id").set_index("sample_id")
        args = []
        for sid in missing_ids:
            chunk = first_idx.at[sid, "data"]
            mag = chunk["magnitude"].to_numpy(dtype=np.float64)
            ts_ns = chunk["timestamp"].astype("int64").to_numpy()
            args.append((sid, mag, ts_ns, bool(detrend), bool(window)))

        try:
            mp_ctx = multiprocessing.get_context("fork")
        except Exception:
            mp_ctx = None

        batch: list[tuple] = []
        with ProcessPoolExecutor(max_workers=max(1, n_jobs), mp_context=mp_ctx) as pool:
            with tqdm(total=len(sample_ids), initial=len(cached),
                      desc="  FFT spectra", leave=False) as pbar:
                for result in pool.map(_fft_spectrum_worker, args, chunksize=chunksize):
                    sid, det, win, fs, freqs_list, amp_list = result
                    cached[sid] = (
                        np.asarray(freqs_list, dtype=np.float64),
                        np.asarray(amp_list, dtype=np.float64),
                    )
                    batch.append(result)
                    pbar.update(1)
                    if len(batch) >= 256:
                        _insert_fft(con, batch)
                        con.execute("CHECKPOINT")
                        batch.clear()
        if batch:
            _insert_fft(con, batch)
            con.execute("CHECKPOINT")

    return cached


def _precompute_psd_spectra(
        data: pd.DataFrame,
        *,
        nperseg_values: tuple[int, ...],
        n_jobs: int,
        con: duckdb.DuckDBPyConnection,
        chunksize: int = 32,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    sample_ids = list(data["sample_id"].drop_duplicates())
    first_idx = data.drop_duplicates(subset="sample_id").set_index("sample_id")

    out: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    args: list[tuple] = []
    for nperseg in nperseg_values:
        cached_n = _get_cached_psd(con, sample_ids, nperseg)
        for sid in sample_ids:
            if sid in cached_n:
                out[(sid, int(nperseg))] = cached_n[sid]
            else:
                chunk = first_idx.at[sid, "data"]
                mag = chunk["magnitude"].to_numpy(dtype=np.float64)
                ts_ns = chunk["timestamp"].astype("int64").to_numpy()
                args.append((sid, mag, ts_ns, int(nperseg)))

    if args:
        try:
            mp_ctx = multiprocessing.get_context("fork")
        except Exception:
            mp_ctx = None

        total = len(sample_ids) * len(nperseg_values)
        batch: list[tuple] = []
        with ProcessPoolExecutor(max_workers=max(1, n_jobs), mp_context=mp_ctx) as pool:
            with tqdm(total=total, initial=total - len(args),
                      desc="  PSD spectra", leave=False) as pbar:
                for result in pool.map(_psd_spectrum_worker, args, chunksize=chunksize):
                    sid, nperseg, fs, freqs_list, psd_list = result
                    out[(sid, int(nperseg))] = (
                        np.asarray(freqs_list, dtype=np.float64),
                        np.asarray(psd_list, dtype=np.float64),
                    )
                    batch.append(result)
                    pbar.update(1)
                    if len(batch) >= 256:
                        _insert_psd(con, batch)
                        con.execute("CHECKPOINT")
                        batch.clear()
        if batch:
            _insert_psd(con, batch)
            con.execute("CHECKPOINT")

    return out


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "samples": int(len(y_true)),
        "samples_trainride": int(y_true.sum()),
        "samples_no_trainride": int(len(y_true) - y_true.sum()),
    }


def _apply_sliding_window(
        data: pd.DataFrame,
        predicted: pd.Series,
        length: int,
        threshold: float,
) -> pd.Series:
    if length <= 1:
        return predicted
    try:
        return (
            data.assign(_pred=predicted)
            .groupby("id", group_keys=False)["_pred"]
            .apply(
                lambda s: (
                        s.astype(int)
                        .rolling(window=length, center=True, min_periods=1)
                        .mean()
                        >= threshold
                ).astype(bool)
            )
        )
    except Exception:
        return predicted


# ---------------------------------------------------------------------------
# SNR computation helpers (read from precomputed spectrum caches)
# ---------------------------------------------------------------------------

def _snr_from_cached_fft(
        data: pd.DataFrame,
        *,
        signal_delta: float,
        spectrum_min_freq: float,
        spectrum_max_freq: float,
        harmonics_mask: bool,
        noise_model: str,
        guard_band: float,
        noise_band: float,
        target_frequency: float,
) -> np.ndarray:
    """Per-sample FFT SNR using ``_fft_spectra``; NaN for missing/failing samples."""
    out = np.empty(len(data), dtype=np.float64)
    cache = _fft_spectra
    sids = data["sample_id"].to_numpy()
    for i, sid in enumerate(sids):
        spec = cache.get(sid) if cache is not None else None
        if spec is None:
            out[i] = np.nan
            continue
        freqs, amplitude = spec
        out[i] = compute_snr_from_spectrum(
            freqs, amplitude,
            target_freq=target_frequency,
            signal_delta=signal_delta,
            spectrum_min_freq=spectrum_min_freq,
            spectrum_max_freq=spectrum_max_freq,
            harmonics_mask=harmonics_mask,
            noise_model=noise_model,
            guard_band=guard_band,
            noise_band=noise_band,
            min_noise_bins=4,
        )
    return out


def _snr_from_cached_psd(
        data: pd.DataFrame,
        *,
        signal_delta: float,
        spectrum_min_freq: float,
        spectrum_max_freq: float,
        harmonics_mask: bool,
        noise_model: str,
        guard_band: float,
        noise_band: float,
        nperseg: int,
        target_frequency: float,
) -> np.ndarray:
    """Per-sample Welch SNR using ``_psd_spectra``; NaN for missing/failing samples."""
    out = np.empty(len(data), dtype=np.float64)
    cache = _psd_spectra
    nperseg_int = int(nperseg)
    sids = data["sample_id"].to_numpy()
    for i, sid in enumerate(sids):
        spec = cache.get((sid, nperseg_int)) if cache is not None else None
        if spec is None:
            out[i] = np.nan
            continue
        freqs, psd = spec
        out[i] = compute_snr_from_spectrum(
            freqs, psd,
            target_freq=target_frequency,
            signal_delta=signal_delta,
            spectrum_min_freq=spectrum_min_freq,
            spectrum_max_freq=spectrum_max_freq,
            harmonics_mask=harmonics_mask,
            noise_model=noise_model,
            guard_band=guard_band,
            noise_band=noise_band,
            min_noise_bins=5,
        )
    return out


def _evaluate_with_snr(
        data: pd.DataFrame,
        snr_values: np.ndarray,
        snr_threshold: float,
        sliding_window_length: int,
        sliding_window_threshold: float,
) -> dict:
    predicted = pd.Series(snr_values >= snr_threshold, index=data.index)
    predicted = _apply_sliding_window(data, predicted, sliding_window_length, sliding_window_threshold)
    return _compute_metrics(
        data["trainride"].to_numpy().astype(bool),
        predicted.to_numpy().astype(bool),
    )


# ---------------------------------------------------------------------------
# FFT trial runner
# ---------------------------------------------------------------------------

def _run_trial_fft(params: dict) -> Tuple[float, dict, dict]:
    snr_kwargs = {k: params[k] for k in (
        "signal_delta", "spectrum_min_freq", "spectrum_max_freq",
        "harmonics_mask", "noise_model", "guard_band", "noise_band",
        "target_frequency",
    )}
    sw_len = params["sliding_window_length"]
    sw_thr = params["sliding_window_threshold"]

    snr_train = _snr_from_cached_fft(_optuna_train_data, **snr_kwargs)
    snr_test = _snr_from_cached_fft(_optuna_test_data, **snr_kwargs)

    best_thr, _ = find_best_threshold_snr(
        snr_train,
        _optuna_train_data["trainride"].to_numpy().astype(bool),
        metric=_optuna_metric,
    )

    stats_train = _evaluate_with_snr(_optuna_train_data, snr_train, best_thr, sw_len, sw_thr)
    stats_test = _evaluate_with_snr(_optuna_test_data, snr_test, best_thr, sw_len, sw_thr)
    stats_train["snr_threshold"] = float(best_thr)
    return stats_test["f1"], stats_test, stats_train


# ---------------------------------------------------------------------------
# PSD trial runner
# ---------------------------------------------------------------------------

def _run_trial_psd(params: dict) -> Tuple[float, dict, dict]:
    snr_kwargs = {k: params[k] for k in (
        "signal_delta", "spectrum_min_freq", "spectrum_max_freq",
        "harmonics_mask", "noise_model", "guard_band", "noise_band",
        "nperseg", "target_frequency",
    )}
    sw_len = params["sliding_window_length"]
    sw_thr = params["sliding_window_threshold"]

    snr_train = _snr_from_cached_psd(_optuna_train_data, **snr_kwargs)
    snr_test = _snr_from_cached_psd(_optuna_test_data, **snr_kwargs)

    best_thr, _ = find_best_threshold_snr(
        snr_train,
        _optuna_train_data["trainride"].to_numpy().astype(bool),
        metric=_optuna_metric,
    )

    stats_train = _evaluate_with_snr(_optuna_train_data, snr_train, best_thr, sw_len, sw_thr)
    stats_test = _evaluate_with_snr(_optuna_test_data, snr_test, best_thr, sw_len, sw_thr)
    stats_train["snr_threshold"] = float(best_thr)
    return stats_test["f1"], stats_test, stats_train


# ---------------------------------------------------------------------------
# Goertzel trial runner (no precompute / cache — recomputed per trial)
# ---------------------------------------------------------------------------

def _snr_goertzel(
        data: pd.DataFrame,
        *,
        target_frequency: float,
        spectrum_min_freq: float,
        spectrum_max_freq: float,
        guard_band: float,
        noise_n_probes: int,
        probe_split: float,
        noise_model: str,
) -> np.ndarray:
    """Per-sample Goertzel SNR.  Dedupes work across rows that share a sample_id."""
    out = np.empty(len(data), dtype=np.float64)
    cache: dict[str, float] = {}
    sids = data["sample_id"].to_numpy()
    chunks = data["data"].to_numpy()
    for i in range(len(data)):
        sid = sids[i]
        if sid in cache:
            out[i] = cache[sid]
            continue
        v = compute_snr_goertzel(
            chunks[i],
            target_freq=float(target_frequency),
            spectrum_min_freq=float(spectrum_min_freq),
            spectrum_max_freq=float(spectrum_max_freq),
            guard_band=float(guard_band),
            noise_n_probes=int(noise_n_probes),
            probe_split=float(probe_split),
            noise_model=str(noise_model),
        )
        cache[sid] = v
        out[i] = v
    return out


def _run_trial_goertzel(params: dict) -> Tuple[float, dict, dict]:
    snr_kwargs = {k: params[k] for k in (
        "target_frequency", "spectrum_min_freq", "spectrum_max_freq",
        "guard_band", "noise_n_probes", "probe_split", "noise_model",
    )}
    sw_len = params["sliding_window_length"]
    sw_thr = params["sliding_window_threshold"]

    snr_train = _snr_goertzel(_optuna_train_data, **snr_kwargs)
    snr_test = _snr_goertzel(_optuna_test_data, **snr_kwargs)

    best_thr, _ = find_best_threshold_snr(
        snr_train,
        _optuna_train_data["trainride"].to_numpy().astype(bool),
        metric=_optuna_metric,
    )

    stats_train = _evaluate_with_snr(_optuna_train_data, snr_train, best_thr, sw_len, sw_thr)
    stats_test = _evaluate_with_snr(_optuna_test_data, snr_test, best_thr, sw_len, sw_thr)
    stats_train["snr_threshold"] = float(best_thr)
    return stats_test["f1"], stats_test, stats_train


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def _run_trial_multi_runs(run_fn, params: dict) -> Tuple[float, dict, dict]:
    global _optuna_train_data, _optuna_test_data, _optuna_all_splits

    all_stats_test: list[dict] = []
    all_stats_train: list[dict] = []

    for train_d, test_d in _optuna_all_splits:
        _optuna_train_data = train_d
        _optuna_test_data = test_d
        _, stats_test, stats_train = run_fn(params)
        all_stats_test.append(stats_test)
        all_stats_train.append(stats_train)

    def _aggregate(all_stats: list[dict]) -> dict:
        agg: dict = {}
        for key in all_stats[0]:
            vals = [float(s[key]) for s in all_stats]
            agg[key] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        return agg

    mean_test = _aggregate(all_stats_test)
    mean_train = _aggregate(all_stats_train)
    mean_f1 = float(np.mean([s["f1"] for s in all_stats_test]))
    return mean_f1, mean_test, mean_train


# ---------------------------------------------------------------------------
# Conditional noise-model sampling
# ---------------------------------------------------------------------------

_GUARD_BAND_DEFAULT = 0.5
_NOISE_BAND_DEFAULT = 3.0


def _sample_noise_params(trial: optuna.Trial) -> dict:
    """Conditionally sample guard_band / noise_band only when noise_model is 'interpolate'."""
    if trial.params.get("noise_model") == "interpolate":
        return {
            "guard_band": trial.suggest_float("guard_band", 0.1, 2.0),
            "noise_band": trial.suggest_float("noise_band", 1.0, 10.0),
        }
    return {
        "guard_band": _GUARD_BAND_DEFAULT,
        "noise_band": _NOISE_BAND_DEFAULT,
    }


# ---------------------------------------------------------------------------
# Common Optuna search loop
# ---------------------------------------------------------------------------

def _print_dataset_stats(data: pd.DataFrame, label: str) -> None:
    total = len(data)
    n_pos = int(data["trainride"].sum())
    n_neg = total - n_pos
    n_ids = data["id"].nunique()
    typer.echo(f"  {label}: {total} samples ({n_pos} trainride / {n_neg} no-trainride), {n_ids} unique IDs")


def _optuna_search_common(
        *,
        dataset_path: Path,
        output_path: Path,
        n_trials: int,
        n_jobs: int,
        sampler: str,
        storage: Optional[str],
        distributions: dict,
        fixed_params: dict,
        run_trial_fn,
        label: str,
        test_frac: float,
        seed: str,
        metric: str = "mcc",
        runs: int = 0,
        conditional_sample_fn=None,
        conditional_param_names: list[str] | None = None,
        cli_params: dict | None = None,
        precompute_fn=None,
) -> None:
    if metric not in ("mcc", "f1"):
        typer.echo(f"Unknown metric '{metric}', must be 'mcc' or 'f1'. Falling back to 'mcc'.", err=True)
        exit(1)

    seed_int = resolve_seed(seed)
    sampler_obj: optuna.samplers.BaseSampler
    if sampler == "tpe":
        sampler_obj = optuna.samplers.TPESampler(seed=seed_int)
    elif sampler == "random":
        sampler_obj = optuna.samplers.RandomSampler(seed=seed_int)
    elif sampler == "cmaes":
        sampler_obj = optuna.samplers.CmaEsSampler(seed=seed_int)
    else:
        typer.echo(f"Unknown sampler '{sampler}', falling back to TPE.", err=True)
        sampler_obj = optuna.samplers.TPESampler(seed=seed_int)

    optimised_param_names = list(distributions.keys())
    _cond_names = conditional_param_names or []
    all_param_names = optimised_param_names + _cond_names
    multi_run = runs > 1

    typer.echo(f"\nRunning Optuna search ({label}) on dataset: {dataset_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    sw = fixed_params.get("sliding_window_length", "na")
    results_csv_path = output_path / f"{dataset_path.stem}.optuna_search_{label}_sw{sw}.csv"
    if storage is None:
        sqlite_path = output_path / f"{dataset_path.stem}.optuna_{label}_sw{sw}.sqlite"
        storage = f"sqlite:///{sqlite_path}"
    typer.echo(f"  Results CSV:    {results_csv_path}")
    typer.echo(f"  Optuna storage: {storage}")

    typer.echo("  Loading dataset …")
    full_data = read_pickle(dataset_path)
    full_data = _attach_sample_ids(full_data)
    dataset_meta = get_metadata_from_dataset(dataset_path)

    typer.echo("  Dataset class distribution:")
    _print_dataset_stats(full_data, "Full dataset")

    # Precompute spectra (cached in DuckDB next to the dataset).  The resulting
    # dicts live in module globals so fork()'d Optuna workers inherit them via
    # copy-on-write — no per-trial recomputation, no pickling.
    fft_spec: dict | None = None
    psd_spec: dict | None = None
    if precompute_fn is not None:
        con, _ = _open_spectrum_cache(dataset_path)
        try:
            fft_spec, psd_spec = precompute_fn(full_data, n_jobs, con)
        finally:
            con.close()

    if multi_run:
        split_seeds = [seed_int + i for i in range(runs)]
        splits = [train_test_split(full_data, test_frac=test_frac, random_state=s, split_column='id') for s in split_seeds]
        typer.echo(f"  Multi-run mode: {runs} runs with different train/test splits")

        for i, (tr, te) in enumerate(splits, start=1):
            _print_dataset_stats(tr, f"Train set (run {i}/{runs})")
            _print_dataset_stats(te, f"Test set  (run {i}/{runs})")

        train_data, test_data = splits[0]
    else:
        splits = None
        train_data, test_data = train_test_split(full_data, test_frac=test_frac, random_state=seed_int, split_column='id')

        _print_dataset_stats(train_data, "Train set")
        _print_dataset_stats(test_data, "Test set")

    # Set module globals so fork()'d workers inherit data + spectrum caches
    # without paying the pickling cost of initargs.
    global _optuna_train_data, _optuna_test_data, _optuna_all_splits
    global _fft_spectra, _psd_spectra, _optuna_metric
    _optuna_train_data = train_data
    _optuna_test_data = test_data
    _optuna_all_splits = splits
    _fft_spectra = fft_spec
    _psd_spectra = psd_spec
    _optuna_metric = metric

    std_cols = (
            ["n_runs"]
            + [f"{m}_test_std" for m in ("f1", "mcc", "accuracy", "precision", "recall")]
            + [f"{m}_train_std" for m in ("f1", "mcc", "accuracy", "precision", "recall")]
            + ["snr_threshold_std"]
    ) if multi_run else []

    csv_header = (
            ["trial_number", "dataset_file", "test_frac", "seed"]
            + ["method", "duration", "trainride_start_seconds"]
            + all_param_names
            + list(fixed_params.keys())
            + ["snr_threshold"]
            + [
                "f1_test", "mcc_test", "accuracy_test", "precision_test", "recall_test",
                "TP_test", "FP_test", "TN_test", "FN_test",
                "samples_test", "samples_trainride_test", "samples_no_trainride_test",
            ]
            + [
                "f1_train", "mcc_train", "accuracy_train", "precision_train", "recall_train",
                "TP_train", "FP_train", "TN_train", "FN_train",
                "samples_train", "samples_trainride_train", "samples_no_trainride_train",
            ]
            + std_cols
    )

    if results_csv_path.exists():
        typer.echo(f"  Appending results to existing CSV: {results_csv_path}")
    else:
        pd.DataFrame(columns=csv_header).to_csv(results_csv_path, index=False)

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    study_name = f"magtrack-tmd-{label}-{metric}-{dataset_path.stem}-window{fixed_params.get('sliding_window_length', 'na')}-{timestamp}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler_obj,
        study_name=study_name,
        storage=storage,
    )

    # Persist CLI parameters / options as study user attributes for traceability
    if cli_params:
        for k, v in cli_params.items():
            try:
                if isinstance(v, (os.PathLike,)):
                    val = str(v)
                elif hasattr(v, "item") and not isinstance(v, (str, bytes)):
                    try:
                        val = v.item()
                    except Exception:
                        val = str(v)
                elif isinstance(v, (list, tuple, dict, str, int, float, bool, type(None))):
                    val = v
                else:
                    val = str(v)
            except Exception:
                val = str(v)
            try:
                study.set_user_attr(k, val)
            except Exception:
                typer.echo(f"Warning: failed to set study user attr {k}", err=True)

    trial_counter = 0
    trials_done = 0
    pbar = tqdm(total=n_trials, desc=f"Optuna trials ({dataset_path.name})")

    try:
        mp_ctx = multiprocessing.get_context('fork')
    except Exception:
        mp_ctx = None

    # Workers inherit the data + spectrum caches via fork; no initializer needed.
    with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp_ctx,
    ) as pool:
        while trials_done < n_trials:
            batch_size = min(n_jobs, n_trials - trials_done)
            batch_trials = []
            for _ in range(batch_size):
                t = study.ask(fixed_distributions=distributions)
                cond_params = conditional_sample_fn(t) if conditional_sample_fn else {}
                batch_trials.append((t, cond_params))

            futures = {}
            for t, cond_params in batch_trials:
                payload = {**t.params, **cond_params, **fixed_params}
                if multi_run:
                    futures[pool.submit(_run_trial_multi_runs, run_trial_fn, payload)] = (t, cond_params)
                else:
                    futures[pool.submit(run_trial_fn, payload)] = (t, cond_params)

            for fut in as_completed(futures):
                t, cond_params = futures[fut]
                try:
                    _, stats_test, stats_train = fut.result()
                    objective_value = float(stats_test[metric])
                except Exception as exc:
                    typer.echo(f"\n  Trial failed: {exc}", err=True)
                    study.tell(t, state=optuna.trial.TrialState.FAIL)
                    pbar.update(1)
                    trials_done += 1
                    continue

                user_attrs: dict = {}
                for mk in ("f1", "mcc", "TP", "FP", "TN", "FN"):
                    user_attrs[f"{mk}_test"] = stats_test[mk]
                    user_attrs[f"{mk}_test_std"] = stats_test.get(mk + '_std', 0.0)
                    user_attrs[f"{mk}_train"] = stats_train[mk]
                    user_attrs[f"{mk}_train_std"] = stats_train.get(mk + '_std', 0.0)
                user_attrs["snr_threshold"] = stats_train["snr_threshold"]
                user_attrs["snr_threshold_std"] = stats_train.get("snr_threshold_std", 0.0)

                all_trial_params = {**t.params, **cond_params}
                all_trial_dists = dict(distributions)
                for ck in _cond_names:
                    if ck in t.distributions:
                        all_trial_dists[ck] = t.distributions[ck]
                    elif ck in cond_params and ck not in all_trial_dists:
                        all_trial_dists[ck] = optuna.distributions.FloatDistribution(
                            float(cond_params[ck]), float(cond_params[ck])
                        )

                study.add_trial(
                    optuna.trial.create_trial(
                        params=all_trial_params,
                        distributions=all_trial_dists,
                        value=objective_value,
                        user_attrs=user_attrs,
                    )
                )

                row: dict = {
                    "trial_number": trial_counter,
                    "dataset_file": os.path.basename(dataset_path),
                    "test_frac": float(test_frac),
                    "seed": seed,
                    "method": label,
                    "duration": dataset_meta["duration"].iloc[0],
                    "trainride_start_seconds": dataset_meta["trainride_start_seconds"].iloc[0],
                }
                for k in optimised_param_names:
                    d = distributions[k]
                    if isinstance(d, optuna.distributions.CategoricalDistribution):
                        row[k] = t.params[k]
                    elif isinstance(d, optuna.distributions.IntDistribution):
                        row[k] = int(t.params[k])
                    else:
                        row[k] = float(t.params[k])
                for k in _cond_names:
                    row[k] = cond_params.get(k)
                for k, v in fixed_params.items():
                    row[k] = v

                for mk in ("f1", "mcc", "accuracy", "precision", "recall"):
                    row[f"{mk}_test"] = float(stats_test[mk])
                for mk in ("TP", "FP", "TN", "FN", "samples", "samples_trainride", "samples_no_trainride"):
                    row[f"{mk}_test"] = int(stats_test[mk])

                for mk in ("f1", "mcc", "accuracy", "precision", "recall"):
                    row[f"{mk}_train"] = float(stats_train[mk])
                for mk in ("TP", "FP", "TN", "FN", "samples", "samples_trainride", "samples_no_trainride"):
                    row[f"{mk}_train"] = int(stats_train[mk])

                # SNR threshold — learned from training data, applied to both sets
                row["snr_threshold"] = float(stats_train["snr_threshold"])

                if multi_run:
                    row["n_runs"] = runs
                    for mk in ("f1", "mcc", "accuracy", "precision", "recall"):
                        row[f"{mk}_test_std"] = float(stats_test.get(f"{mk}_std", 0.0))
                        row[f"{mk}_train_std"] = float(stats_train.get(f"{mk}_std", 0.0))
                    row["snr_threshold_std"] = float(stats_train.get("snr_threshold_std", 0.0))

                trial_counter += 1

                try:
                    pd.DataFrame([row], columns=csv_header).to_csv(results_csv_path, index=False, mode="a", header=False)
                except Exception as e:
                    typer.echo(f"\n  Failed to save trial result: {e}", err=True)

                pbar.update(1)
                trials_done += 1

    pbar.close()

    try:
        best = study.best_trial
    except ValueError as e:
        typer.echo("\nNo successful trials were completed or best trial could not be retrieved.", err=True)
        typer.echo(f"Study error: {e}", err=True)
        typer.echo(f"Results (if any) saved to: {results_csv_path}")
        return

    typer.echo(f"\n=== Best configuration for {dataset_path.name} ({label}, metric={metric}) ===")
    f1_test = best.user_attrs.get("f1_test") if hasattr(best, "user_attrs") else None
    f1_test_std = best.user_attrs.get("f1_test_std") if hasattr(best, "user_attrs") else None
    mcc_test = best.user_attrs.get("mcc_test") if hasattr(best, "user_attrs") else None
    mcc_test_std = best.user_attrs.get("mcc_test_std") if hasattr(best, "user_attrs") else None
    snr_thr_test = best.user_attrs.get("snr_threshold") if hasattr(best, "user_attrs") else None

    if f1_test is not None:
        try:
            typer.echo(f"  F1 (test):  {float(f1_test):.4f} ± {float(f1_test_std or 0.0):.4f}")
        except Exception:
            typer.echo(f"  F1 (test):  {f1_test}")
    else:
        if metric == "f1":
            typer.echo(f"  F1 (test):  {best.value:.4f}")

    if mcc_test is not None:
        try:
            typer.echo(f"  MCC (test): {float(mcc_test):.4f} ± {float(mcc_test_std or 0.0):.4f}")
        except Exception:
            typer.echo(f"  MCC (test): {mcc_test}")
    else:
        if metric == "mcc":
            typer.echo(f"  MCC (test): {best.value:.4f}")

    if snr_thr_test is not None:
        try:
            typer.echo(f"  SNR threshold (learned from train): {float(snr_thr_test):.4f}")
        except Exception:
            typer.echo(f"  SNR threshold (learned from train): {snr_thr_test}")

    for pname in all_param_names:
        if pname in best.params:
            typer.echo(f"  {pname:30s} {best.params[pname]}")
    typer.echo(f"  Results saved to: {results_csv_path}")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command("optuna-search-fft")
def optuna_search_fft(
        dataset_path: Path = typer.Argument(..., help="Path to a tmd dataset (pkl)"),
        output_path: Path = typer.Option(
            ...,
            "--output-path", "-o",
            help="Directory to write the results CSV and the (default) Optuna SQLite into. Created if missing.",
        ),
        n_trials: int = typer.Option(5000, help="Number of Optuna trials"),
        n_jobs: Optional[int] = typer.Option(multiprocessing.cpu_count() - 1,
                                             help="Number of parallel Optuna workers (default: cpu_count() - 1)"),
        sampler: str = typer.Option(
            "tpe",
            help="Optuna sampler: 'tpe' (Bayesian), 'random', or 'cmaes'",
        ),
        storage: Optional[str] = typer.Option(
            None,
            help="Optuna storage URL override (e.g. 'sqlite:///optuna.db' or 'postgresql://…'). "
                 "Defaults to a SQLite file inside --output-path.",
        ),
        target_frequency: float = typer.Option(16.7, help="Target frequency (Hz)"),
        signal_delta_min: float = typer.Option(0.05, help="signal_delta lower bound (Hz)"),
        signal_delta_max: float = typer.Option(2.0, help="signal_delta upper bound (Hz)"),
        spectrum_min_freq_min: float = typer.Option(0.5, help="spectrum_min_freq lower bound (Hz)"),
        spectrum_min_freq_max: float = typer.Option(16.4, help="spectrum_min_freq upper bound (Hz)"),
        spectrum_max_freq_min: float = typer.Option(17.0, help="spectrum_max_freq lower bound (Hz)"),
        spectrum_max_freq_max: float = typer.Option(45.0, help="spectrum_max_freq upper bound (Hz)"),
        noise_models: List[str] = typer.Option(["median", "interpolate", "spline"],
                                               help="Noise model choices to include in the search. Provide multiple times to include more than one."),
        test_frac: float = typer.Option(0.3, help="Fraction of unique IDs held out as test set (0–1)"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split"),
        metric: str = typer.Option("mcc", help="Metric to optimise: 'mcc' (default) or 'f1'"),
        runs: int = typer.Option(10, help="Number of evaluation runs (0 = single run, >1 = average over N splits)"),
        sliding_window_length: int = typer.Option(
            1, help="Sliding window length to use (single value). Default: 1"
        ),
        sliding_window_threshold_min: float = typer.Option(0.1, help="sliding_window_threshold lower bound"),
        sliding_window_threshold_max: float = typer.Option(0.9, help="sliding_window_threshold upper bound"),
) -> None:
    """Optuna hyperparameter search for the FFT-based detector (check_electric_fft).

    The SNR threshold is not a search parameter — it is determined automatically
    for each trial by sweeping the sorted SNR values on the training split and
    picking the threshold that maximises *metric*.
    """
    if not noise_models:
        typer.echo("At least one noise_model must be provided.", err=True)
        raise typer.Exit(code=1)

    dist: dict = {
        "signal_delta": optuna.distributions.FloatDistribution(signal_delta_min, signal_delta_max),
        "spectrum_min_freq": optuna.distributions.FloatDistribution(spectrum_min_freq_min, spectrum_min_freq_max),
        "spectrum_max_freq": optuna.distributions.FloatDistribution(spectrum_max_freq_min, spectrum_max_freq_max),
        "harmonics_mask": optuna.distributions.CategoricalDistribution([True, False]),
        "noise_model": optuna.distributions.CategoricalDistribution(noise_models),
    }

    fixed_params: dict = {
        "target_frequency": target_frequency,
        "sliding_window_length": int(sliding_window_length),
    }
    if sliding_window_length > 1:
        dist["sliding_window_threshold"] = optuna.distributions.FloatDistribution(
            sliding_window_threshold_min, sliding_window_threshold_max
        )
    else:
        fixed_params["sliding_window_threshold"] = 1

    def _precompute(data: pd.DataFrame, n_jobs: int, con: duckdb.DuckDBPyConnection):
        fft = _precompute_fft_spectra(
            data, detrend=True, window=True, n_jobs=n_jobs, con=con,
        )
        return fft, None

    cli_params = {
        "target_frequency": float(target_frequency),
        "signal_delta_min": float(signal_delta_min),
        "signal_delta_max": float(signal_delta_max),
        "spectrum_min_freq_min": float(spectrum_min_freq_min),
        "spectrum_min_freq_max": float(spectrum_min_freq_max),
        "spectrum_max_freq_min": float(spectrum_max_freq_min),
        "spectrum_max_freq_max": float(spectrum_max_freq_max),
        "noise_models": list(noise_models),
        "test_frac": float(test_frac),
        "seed": seed,
        "metric": metric,
        "runs": int(runs),
        "sliding_window_length": int(sliding_window_length),
        "sliding_window_threshold_min": float(sliding_window_threshold_min),
        "sliding_window_threshold_max": float(sliding_window_threshold_max),
    }

    _optuna_search_common(
        dataset_path=dataset_path,
        output_path=output_path,
        n_trials=n_trials,
        n_jobs=n_jobs,
        sampler=sampler,
        storage=storage,
        distributions=dist,
        fixed_params=fixed_params,
        run_trial_fn=_run_trial_fft,
        label="fft",
        test_frac=test_frac,
        seed=seed,
        metric=metric,
        runs=runs,
        conditional_sample_fn=_sample_noise_params,
        conditional_param_names=["guard_band", "noise_band"],
        cli_params=cli_params,
        precompute_fn=_precompute,
    )


@app.command("optuna-search-psd")
def optuna_search_psd(
        dataset_path: Path = typer.Argument(..., help="Path to a tmd dataset (pkl)"),
        output_path: Path = typer.Option(
            ...,
            "--output-path", "-o",
            help="Directory to write the results CSV and the (default) Optuna SQLite into. Created if missing.",
        ),
        n_trials: int = typer.Option(5000, help="Number of Optuna trials"),
        n_jobs: Optional[int] = typer.Option(multiprocessing.cpu_count() - 1,
                                             help="Number of parallel Optuna workers (default: cpu_count() - 1)"),
        sampler: str = typer.Option(
            "tpe",
            help="Optuna sampler: 'tpe' (Bayesian), 'random', or 'cmaes'",
        ),
        storage: Optional[str] = typer.Option(
            None,
            help="Optuna storage URL override (e.g. 'sqlite:///optuna.db' or 'postgresql://…'). "
                 "Defaults to a SQLite file inside --output-path.",
        ),
        target_frequency: float = typer.Option(16.7, help="Target frequency (Hz)"),
        signal_delta_min: float = typer.Option(0.05, help="signal_delta lower bound (Hz)"),
        signal_delta_max: float = typer.Option(2.0, help="signal_delta upper bound (Hz)"),
        spectrum_min_freq_min: float = typer.Option(0.5, help="spectrum_min_freq lower bound (Hz)"),
        spectrum_min_freq_max: float = typer.Option(16.4, help="spectrum_min_freq upper bound (Hz)"),
        spectrum_max_freq_min: float = typer.Option(17.0, help="spectrum_max_freq lower bound (Hz)"),
        spectrum_max_freq_max: float = typer.Option(45.0, help="spectrum_max_freq upper bound (Hz)"),
        noise_models: List[str] = typer.Option(["median", "interpolate", "spline"],
                                               help="Noise model choices to include in the search. Provide multiple times to include more than one."),
        test_frac: float = typer.Option(0.3, help="Fraction of unique IDs held out as test set (0–1)"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split"),
        metric: str = typer.Option("mcc", help="Metric to optimise: 'mcc' (default) or 'f1'"),
        runs: int = typer.Option(0, help="Number of evaluation runs (0 = single run, >1 = average over N splits)"),
        sliding_window_length: int = typer.Option(
            1, help="Sliding window length to use (single value). Default: 1"
        ),
        sliding_window_threshold_min: float = typer.Option(0.1, help="sliding_window_threshold lower bound"),
        sliding_window_threshold_max: float = typer.Option(0.9, help="sliding_window_threshold upper bound"),
) -> None:
    """Optuna hyperparameter search for the PSD-based detector (check_electric_psd).

    The SNR threshold is not a search parameter — it is determined automatically
    for each trial by sweeping the sorted SNR values on the training split and
    picking the threshold that maximises *metric*.
    """
    if not noise_models:
        typer.echo("At least one noise_model must be provided.", err=True)
        raise typer.Exit(code=1)

    dist: dict = {
        "signal_delta": optuna.distributions.FloatDistribution(signal_delta_min, signal_delta_max),
        "spectrum_min_freq": optuna.distributions.FloatDistribution(spectrum_min_freq_min, spectrum_min_freq_max),
        "spectrum_max_freq": optuna.distributions.FloatDistribution(spectrum_max_freq_min, spectrum_max_freq_max),
        "harmonics_mask": optuna.distributions.CategoricalDistribution([True, False]),
        "noise_model": optuna.distributions.CategoricalDistribution(noise_models),
        # PSD-specific: discrete so spectra can be precomputed and cached.
        "nperseg": optuna.distributions.CategoricalDistribution(list(PSD_NPERSEG_CHOICES)),
    }

    fixed_params: dict = {
        "target_frequency": target_frequency,
        "sliding_window_length": int(sliding_window_length),
    }
    if sliding_window_length > 1:
        dist["sliding_window_threshold"] = optuna.distributions.FloatDistribution(
            sliding_window_threshold_min, sliding_window_threshold_max
        )
    else:
        fixed_params["sliding_window_threshold"] = 0.5

    def _precompute(data: pd.DataFrame, n_jobs: int, con: duckdb.DuckDBPyConnection):
        psd = _precompute_psd_spectra(
            data, nperseg_values=PSD_NPERSEG_CHOICES, n_jobs=n_jobs, con=con,
        )
        return None, psd

    cli_params = {
        "target_frequency": float(target_frequency),
        "signal_delta_min": float(signal_delta_min),
        "signal_delta_max": float(signal_delta_max),
        "spectrum_min_freq_min": float(spectrum_min_freq_min),
        "spectrum_min_freq_max": float(spectrum_min_freq_max),
        "spectrum_max_freq_min": float(spectrum_max_freq_min),
        "spectrum_max_freq_max": float(spectrum_max_freq_max),
        "noise_models": list(noise_models),
        "test_frac": float(test_frac),
        "seed": seed,
        "metric": metric,
        "runs": int(runs),
        "sliding_window_length": int(sliding_window_length),
        "sliding_window_threshold_min": float(sliding_window_threshold_min),
        "sliding_window_threshold_max": float(sliding_window_threshold_max),
    }

    _optuna_search_common(
        dataset_path=dataset_path,
        output_path=output_path,
        n_trials=n_trials,
        n_jobs=n_jobs,
        sampler=sampler,
        storage=storage,
        distributions=dist,
        fixed_params=fixed_params,
        run_trial_fn=_run_trial_psd,
        label="psd",
        test_frac=test_frac,
        seed=seed,
        metric=metric,
        runs=runs,
        conditional_sample_fn=_sample_noise_params,
        conditional_param_names=["guard_band", "noise_band"],
        cli_params=cli_params,
        precompute_fn=_precompute,
    )


@app.command("preprocess-fft")
def preprocess_fft(
        dataset_path: Path = typer.Argument(..., help="Path to a tmd dataset (pkl)"),
        n_jobs: Optional[int] = typer.Option(
            multiprocessing.cpu_count() - 1,
            help="Number of parallel workers (default: cpu_count() - 1)",
        ),
) -> None:
    """Precompute FFT spectra for *dataset_path* into its .cache.duckdb."""
    typer.echo(f"\nPrecomputing FFT spectra for: {dataset_path}")
    typer.echo("  Loading dataset …")
    full_data = read_pickle(dataset_path)
    full_data = _attach_sample_ids(full_data)
    _print_dataset_stats(full_data, "Full dataset")

    con, _ = _open_spectrum_cache(dataset_path)
    try:
        _precompute_fft_spectra(
            full_data, detrend=True, window=True, n_jobs=int(n_jobs or 1), con=con,
        )
    finally:
        con.close()
    typer.echo("  Done.")


@app.command("preprocess-psd")
def preprocess_psd(
        dataset_path: Path = typer.Argument(..., help="Path to a tmd dataset (pkl)"),
        n_jobs: Optional[int] = typer.Option(
            multiprocessing.cpu_count() - 1,
            help="Number of parallel workers (default: cpu_count() - 1)",
        ),
) -> None:
    """Precompute PSD spectra (all nperseg choices) for *dataset_path* into its .cache.duckdb."""
    typer.echo(f"\nPrecomputing PSD spectra for: {dataset_path}")
    typer.echo(f"  nperseg values: {list(PSD_NPERSEG_CHOICES)}")
    typer.echo("  Loading dataset …")
    full_data = read_pickle(dataset_path)
    full_data = _attach_sample_ids(full_data)
    _print_dataset_stats(full_data, "Full dataset")

    con, _ = _open_spectrum_cache(dataset_path)
    try:
        _precompute_psd_spectra(
            full_data, nperseg_values=PSD_NPERSEG_CHOICES, n_jobs=int(n_jobs or 1), con=con,
        )
    finally:
        con.close()
    typer.echo("  Done.")


def _amp_goertzel_target(data: pd.DataFrame, target_frequency: float) -> np.ndarray:
    from fastgoertzel import goertzel
    out = np.empty(len(data), dtype=np.float64)
    cache: dict[str, float] = {}
    sids = data["sample_id"].to_numpy()
    chunks = data["data"].to_numpy()
    for i in range(len(data)):
        sid = sids[i]
        if sid in cache:
            out[i] = cache[sid]
            continue
        chunk = chunks[i]
        try:
            sig = chunk["magnitude"].to_numpy(dtype=np.float64)
            ts = chunk["timestamp"].astype("int64").to_numpy() / 1e9
            if sig.size < 8 or ts.size < 2:
                v = float("nan")
            else:
                fs = 1.0 / float(np.median(np.diff(ts)))
                norm = float(target_frequency) / fs
                if norm <= 0.0 or norm >= 0.5:
                    v = float("nan")
                else:
                    amp, _ = goertzel(sig, norm)
                    v = float(amp)
        except Exception:
            v = float("nan")
        cache[sid] = v
        out[i] = v
    return out


@app.command("goertzel-single")
def goertzel_single(
        dataset_path: Path = typer.Argument(..., help="Path to a tmd dataset (pkl)"),
        output_path: Path = typer.Option(
            ...,
            "--output-path", "-o",
            help="Directory to write the results CSV into. Created if missing.",
        ),
        target_frequency: float = typer.Option(16.7, help="Target frequency (Hz)"),
        test_frac: float = typer.Option(0.3, help="Fraction of unique IDs held out as test set (0–1)"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split"),
        metric: str = typer.Option("mcc", help="Metric used for threshold calibration: 'mcc' or 'f1'"),
        runs: int = typer.Option(10, help="Number of evaluation runs (0 = single run, >1 = average over N splits)"),
        sliding_window_length: int = typer.Option(1, help="Sliding window length for majority vote (default: 1)"),
        sliding_window_threshold: float = typer.Option(0.5, help="Sliding window majority-vote threshold"),
) -> None:
    """Single-frequency Goertzel baseline: amplitude at *target_frequency* only.

    Calibrates a threshold on the train split that maximises *metric*, then
    evaluates on the test split.  No Optuna; writes the same CSV layout as
    the search commands (one row per run).
    """
    if metric not in ("mcc", "f1"):
        typer.echo(f"Unknown metric '{metric}', must be 'mcc' or 'f1'.", err=True)
        raise typer.Exit(code=1)

    seed_int = resolve_seed(seed)
    multi_run = runs > 1
    label = "goertzel_single"

    output_path.mkdir(parents=True, exist_ok=True)
    results_csv_path = output_path / f"{dataset_path.stem}.{label}.csv"
    typer.echo(f"\nRunning {label} on dataset: {dataset_path}")
    typer.echo(f"  Results CSV:    {results_csv_path}")

    typer.echo("  Loading dataset …")
    full_data = read_pickle(dataset_path)
    full_data = _attach_sample_ids(full_data)
    dataset_meta = get_metadata_from_dataset(dataset_path)
    typer.echo("  Dataset class distribution:")
    _print_dataset_stats(full_data, "Full dataset")

    if multi_run:
        split_seeds = [seed_int + i for i in range(runs)]
        splits = [train_test_split(full_data, test_frac=test_frac, random_state=s, split_column='id') for s in split_seeds]
        typer.echo(f"  Multi-run mode: {runs} runs with different train/test splits")
        for i, (tr, te) in enumerate(splits, start=1):
            _print_dataset_stats(tr, f"Train set (run {i}/{runs})")
            _print_dataset_stats(te, f"Test set  (run {i}/{runs})")
    else:
        splits = [train_test_split(full_data, test_frac=test_frac, random_state=seed_int, split_column='id')]
        _print_dataset_stats(splits[0][0], "Train set")
        _print_dataset_stats(splits[0][1], "Test set")

    fixed_params = {
        "target_frequency": float(target_frequency),
        "sliding_window_length": int(sliding_window_length),
        "sliding_window_threshold": float(sliding_window_threshold),
    }

    csv_header = (
            ["run", "dataset_file", "test_frac", "seed"]
            + ["method", "duration", "trainride_start_seconds"]
            + list(fixed_params.keys())
            + [
                "snr_threshold",
                "f1_test", "mcc_test", "accuracy_test", "precision_test", "recall_test",
                "TP_test", "FP_test", "TN_test", "FN_test",
                "samples_test", "samples_trainride_test", "samples_no_trainride_test",
                "f1_train", "mcc_train", "accuracy_train", "precision_train", "recall_train",
                "TP_train", "FP_train", "TN_train", "FN_train",
                "samples_train", "samples_trainride_train", "samples_no_trainride_train",
            ]
    )
    pd.DataFrame(columns=csv_header).to_csv(results_csv_path, index=False)

    all_stats_train: list[dict] = []
    all_stats_test: list[dict] = []
    for run_i, (train_data, test_data) in enumerate(
            tqdm(splits, desc="Runs"), start=1
    ):
        amp_train = _amp_goertzel_target(train_data, target_frequency)
        amp_test = _amp_goertzel_target(test_data, target_frequency)

        best_thr, _ = find_best_threshold_snr(
            amp_train,
            train_data["trainride"].to_numpy().astype(bool),
            metric=metric,
        )
        stats_train = _evaluate_with_snr(
            train_data, amp_train, best_thr,
            int(sliding_window_length), float(sliding_window_threshold),
        )
        stats_test = _evaluate_with_snr(
            test_data, amp_test, best_thr,
            int(sliding_window_length), float(sliding_window_threshold),
        )
        stats_train["snr_threshold"] = float(best_thr)
        all_stats_train.append(stats_train)
        all_stats_test.append(stats_test)

        row: dict = {
            "run": run_i,
            "dataset_file": os.path.basename(dataset_path),
            "test_frac": float(test_frac),
            "seed": seed,
            "method": label,
            "duration": dataset_meta["duration"].iloc[0],
            "trainride_start_seconds": dataset_meta["trainride_start_seconds"].iloc[0],
            **fixed_params,
            "snr_threshold": float(best_thr),
        }
        for mk in ("f1", "mcc", "accuracy", "precision", "recall"):
            row[f"{mk}_test"] = float(stats_test[mk])
            row[f"{mk}_train"] = float(stats_train[mk])
        for mk in ("TP", "FP", "TN", "FN", "samples", "samples_trainride", "samples_no_trainride"):
            row[f"{mk}_test"] = int(stats_test[mk])
            row[f"{mk}_train"] = int(stats_train[mk])
        pd.DataFrame([row], columns=csv_header).to_csv(results_csv_path, index=False, mode="a", header=False)

    typer.echo(f"\n=== Results ({label}, metric={metric}) ===")
    if multi_run:
        for mk in ("f1", "mcc"):
            vals_test = np.array([s[mk] for s in all_stats_test], dtype=np.float64)
            vals_train = np.array([s[mk] for s in all_stats_train], dtype=np.float64)
            typer.echo(
                f"  {mk.upper():3s}  test: {vals_test.mean():.4f} ± {vals_test.std():.4f}   "
                f"train: {vals_train.mean():.4f} ± {vals_train.std():.4f}"
            )
        thrs = np.array([s["snr_threshold"] for s in all_stats_train], dtype=np.float64)
        typer.echo(f"  Threshold (mean over runs): {thrs.mean():.6f} ± {thrs.std():.6f}")
    else:
        s_te = all_stats_test[0]
        s_tr = all_stats_train[0]
        typer.echo(f"  F1   test: {s_te['f1']:.4f}   train: {s_tr['f1']:.4f}")
        typer.echo(f"  MCC  test: {s_te['mcc']:.4f}   train: {s_tr['mcc']:.4f}")
        typer.echo(f"  Threshold: {s_tr['snr_threshold']:.6f}")
    typer.echo(f"  Results saved to: {results_csv_path}")


@app.command("optuna-search-goertzel")
def optuna_search_goertzel(
        dataset_path: Path = typer.Argument(..., help="Path to a tmd dataset (pkl)"),
        output_path: Path = typer.Option(
            ...,
            "--output-path", "-o",
            help="Directory to write the results CSV and the (default) Optuna SQLite into. Created if missing.",
        ),
        n_trials: int = typer.Option(5000, help="Number of Optuna trials"),
        n_jobs: Optional[int] = typer.Option(multiprocessing.cpu_count() - 1,
                                             help="Number of parallel Optuna workers (default: cpu_count() - 1)"),
        sampler: str = typer.Option(
            "tpe",
            help="Optuna sampler: 'tpe' (Bayesian), 'random', or 'cmaes'",
        ),
        storage: Optional[str] = typer.Option(
            None,
            help="Optuna storage URL override (e.g. 'sqlite:///optuna.db' or 'postgresql://…'). "
                 "Defaults to a SQLite file inside --output-path.",
        ),
        target_frequency: float = typer.Option(16.7, help="Target frequency (Hz)"),
        spectrum_min_freq_min: float = typer.Option(0.5, help="spectrum_min_freq lower bound (Hz)"),
        spectrum_min_freq_max: float = typer.Option(16.4, help="spectrum_min_freq upper bound (Hz)"),
        spectrum_max_freq_min: float = typer.Option(17.0, help="spectrum_max_freq lower bound (Hz)"),
        spectrum_max_freq_max: float = typer.Option(45.0, help="spectrum_max_freq upper bound (Hz)"),
        guard_band_min: float = typer.Option(0.1, help="guard_band lower bound (Hz)"),
        guard_band_max: float = typer.Option(2.0, help="guard_band upper bound (Hz)"),
        noise_n_probes_min: int = typer.Option(4, help="noise_n_probes lower bound"),
        noise_n_probes_max: int = typer.Option(32, help="noise_n_probes upper bound"),
        probe_split_min: float = typer.Option(0.0, help="probe_split lower bound (fraction below target)"),
        probe_split_max: float = typer.Option(1.0, help="probe_split upper bound (fraction below target)"),
        noise_models: List[str] = typer.Option(["median", "interpolate"],
                                               help="Noise model choices ('median' or 'interpolate'). Provide multiple times to include more than one."),
        test_frac: float = typer.Option(0.3, help="Fraction of unique IDs held out as test set (0–1)"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split"),
        metric: str = typer.Option("mcc", help="Metric to optimise: 'mcc' (default) or 'f1'"),
        runs: int = typer.Option(10, help="Number of evaluation runs (0 = single run, >1 = average over N splits)"),
        sliding_window_length: int = typer.Option(
            1, help="Sliding window length to use (single value). Default: 1"
        ),
        sliding_window_threshold_min: float = typer.Option(0.1, help="sliding_window_threshold lower bound"),
        sliding_window_threshold_max: float = typer.Option(0.9, help="sliding_window_threshold upper bound"),
) -> None:
    """Optuna hyperparameter search for the Goertzel-based detector.

    Single-shot Goertzel (one window over the full chunk) at the target
    frequency plus an asymmetric set of off-target probe frequencies.  The
    SNR threshold is learned from the training split, identical to the
    FFT/PSD searches.  No spectrum cache — Goertzel is recomputed per trial.
    """
    if not noise_models:
        typer.echo("At least one noise_model must be provided.", err=True)
        raise typer.Exit(code=1)
    for nm in noise_models:
        if nm not in ("median", "interpolate"):
            typer.echo(
                f"Unsupported noise_model '{nm}' for goertzel (use 'median' or 'interpolate').",
                err=True,
            )
            raise typer.Exit(code=1)

    dist: dict = {
        "spectrum_min_freq": optuna.distributions.FloatDistribution(spectrum_min_freq_min, spectrum_min_freq_max),
        "spectrum_max_freq": optuna.distributions.FloatDistribution(spectrum_max_freq_min, spectrum_max_freq_max),
        "guard_band": optuna.distributions.FloatDistribution(guard_band_min, guard_band_max),
        "noise_n_probes": optuna.distributions.IntDistribution(noise_n_probes_min, noise_n_probes_max),
        "probe_split": optuna.distributions.FloatDistribution(probe_split_min, probe_split_max),
        "noise_model": optuna.distributions.CategoricalDistribution(noise_models),
    }

    fixed_params: dict = {
        "target_frequency": target_frequency,
        "sliding_window_length": int(sliding_window_length),
    }
    if sliding_window_length > 1:
        dist["sliding_window_threshold"] = optuna.distributions.FloatDistribution(
            sliding_window_threshold_min, sliding_window_threshold_max
        )
    else:
        fixed_params["sliding_window_threshold"] = 1

    cli_params = {
        "target_frequency": float(target_frequency),
        "spectrum_min_freq_min": float(spectrum_min_freq_min),
        "spectrum_min_freq_max": float(spectrum_min_freq_max),
        "spectrum_max_freq_min": float(spectrum_max_freq_min),
        "spectrum_max_freq_max": float(spectrum_max_freq_max),
        "guard_band_min": float(guard_band_min),
        "guard_band_max": float(guard_band_max),
        "noise_n_probes_min": int(noise_n_probes_min),
        "noise_n_probes_max": int(noise_n_probes_max),
        "probe_split_min": float(probe_split_min),
        "probe_split_max": float(probe_split_max),
        "noise_models": list(noise_models),
        "test_frac": float(test_frac),
        "seed": seed,
        "metric": metric,
        "runs": int(runs),
        "sliding_window_length": int(sliding_window_length),
        "sliding_window_threshold_min": float(sliding_window_threshold_min),
        "sliding_window_threshold_max": float(sliding_window_threshold_max),
    }

    _optuna_search_common(
        dataset_path=dataset_path,
        output_path=output_path,
        n_trials=n_trials,
        n_jobs=n_jobs,
        sampler=sampler,
        storage=storage,
        distributions=dist,
        fixed_params=fixed_params,
        run_trial_fn=_run_trial_goertzel,
        label="goertzel",
        test_frac=test_frac,
        seed=seed,
        metric=metric,
        runs=runs,
        conditional_sample_fn=None,
        conditional_param_names=None,
        cli_params=cli_params,
        precompute_fn=None,
    )


if __name__ == "__main__":
    app()
