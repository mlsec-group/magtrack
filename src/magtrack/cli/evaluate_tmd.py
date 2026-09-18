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
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from magtrack.utils.evaluation import train_test_split, find_best_threshold_snr
from magtrack.utils.loader import read_pickle, get_metadata_from_dataset
from magtrack.utils.tmd_functions import (
    compute_fft_spectrum,
    compute_psd_spectrum,
    compute_snr_goertzel,
)
from magtrack.utils.utils import resolve_seed

PSD_NPERSEG_CHOICES: tuple[int, ...] = (128, 192, 256, 384, 512)

app = typer.Typer(help="Hyperparameter search for FFT / PSD transport mode detectors.")

# ---------------------------------------------------------------------------
# Worker-process globals
# ---------------------------------------------------------------------------
_optuna_splits: list[tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]] | None = None
_optuna_metric: str = "mcc"

_fft_spectra: "_Spectra | None" = None
_psd_spectra: "dict[int, _Spectra] | None" = None
_goertzel_chunks: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Spectra container
# ---------------------------------------------------------------------------

class _Spectra:
    """Spectra for a set of samples, padded into two dense matrices.

    ``freqs`` and ``values`` are ``(n_samples, n_bins)`` float64 arrays; row *i*
    holds sample *i*'s spectrum followed by padding — ``+inf`` in ``freqs`` and
    ``0.0`` in ``values``.  Every band selection in the SNR kernels is an
    interval test on ``freqs``, so padding bins fail all of them and never reach
    a reduction.  That is what lets a trial evaluate whole blocks of samples
    with a handful of numpy calls instead of one Python call per sample.
    """

    __slots__ = ("freqs", "values")

    def __init__(self, freqs: np.ndarray, values: np.ndarray) -> None:
        self.freqs = freqs
        self.values = values

    def __len__(self) -> int:
        return int(self.freqs.shape[0])

    @classmethod
    def from_rows(cls, rows: list[tuple[np.ndarray, np.ndarray]]) -> "_Spectra":
        """Build from a list of per-sample ``(freqs, values)`` pairs."""
        n_bins = max((f.size for f, _ in rows), default=0)
        freqs = np.full((len(rows), n_bins), np.inf, dtype=np.float64)
        values = np.zeros((len(rows), n_bins), dtype=np.float64)
        for i, (f, v) in enumerate(rows):
            freqs[i, :f.size] = f
            values[i, :v.size] = v
        return cls(freqs, values)


# ---------------------------------------------------------------------------
# Spectrum cache (DuckDB next to the .pkl)
# ---------------------------------------------------------------------------

_FFT_TABLE = "fft_spectra_v2"
_PSD_TABLE = "psd_spectra_v2"
_LEGACY_TABLES = ("fft_spectra", "psd_spectra")

_PRECOMPUTE_BLOCK = 128


def _make_sample_id(row_id: str, chunk_data: pd.DataFrame) -> str:
    """Stable, content-derived identifier for one sample (id-chunk).

    Uses the recording id plus the first/last timestamp of the chunk and the
    sample count.  Two rows produced from the same source chunk always map to
    the same id; an unrelated chunk almost certainly differs in at least one
    of these fields.
    """
    ts = chunk_data["timestamp"].to_numpy()
    ts = ts.view("int64") if ts.dtype.kind == "M" else ts.astype("int64")
    if ts.size == 0:
        return f"{row_id}#empty"
    return f"{row_id}#{ts[0]}#{ts[-1]}#{ts.size}"


def _attach_sample_ids(data: pd.DataFrame) -> pd.DataFrame:
    ids = data["id"].to_numpy()
    chunks = data["data"].to_numpy()
    sample_ids = [_make_sample_id(ids[i], chunks[i]) for i in range(len(data))]
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
            n_signal  INTEGER NOT NULL,
            amplitude BLOB NOT NULL,
            PRIMARY KEY (sample_id, detrend, "window")
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_PSD_TABLE} (
            sample_id TEXT NOT NULL,
            nperseg   INTEGER NOT NULL,
            fs        DOUBLE NOT NULL,
            n_fft     INTEGER NOT NULL,
            psd       BLOB NOT NULL,
            PRIMARY KEY (sample_id, nperseg)
        )
    """)
    for legacy in _LEGACY_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {legacy}")
    n_fft = con.execute(f"SELECT COUNT(*) FROM {_FFT_TABLE}").fetchone()[0]
    n_psd = con.execute(f"SELECT COUNT(*) FROM {_PSD_TABLE}").fetchone()[0]
    typer.echo(f"  Spectrum cache: {cache_path}  (fft={n_fft:,}, psd={n_psd:,} entries)")
    return con, cache_path


def _spectrum_from_row(fs: float, n_transform: int, blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild ``(freqs, values)`` from a cache row.

    The frequency axis is not stored: it is fully determined by the sampling
    rate and the transform length, and ``rfftfreq`` reproduces it exactly.
    """
    values = np.frombuffer(blob, dtype=np.float64)
    freqs = np.fft.rfftfreq(int(n_transform), d=1 / float(fs))
    return freqs, values


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
            SELECT s.sample_id, s.fs, s.n_signal, s.amplitude
            FROM {_FFT_TABLE} s
            INNER JOIN _qids q ON s.sample_id = q.sample_id
            WHERE s.detrend = ? AND s."window" = ?
        """, [bool(detrend), bool(window)]).fetchall()
    finally:
        con.unregister("_qids")
    return {r[0]: _spectrum_from_row(r[1], r[2], r[3]) for r in rows}


def _insert_fft(
        con: duckdb.DuckDBPyConnection,
        rows: list[tuple],
) -> None:
    """rows: list of (sample_id, detrend, window, fs, n_signal, amplitude_bytes)."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["sample_id", "detrend", "window", "fs", "n_signal", "amplitude"])
    con.register("_ir", df)
    try:
        con.execute(f"""
            INSERT INTO {_FFT_TABLE}
            SELECT sample_id, detrend, "window", fs, n_signal, amplitude FROM _ir
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
            SELECT s.sample_id, s.fs, s.n_fft, s.psd
            FROM {_PSD_TABLE} s
            INNER JOIN _qids q ON s.sample_id = q.sample_id
            WHERE s.nperseg = ?
        """, [int(nperseg)]).fetchall()
    finally:
        con.unregister("_qids")
    return {r[0]: _spectrum_from_row(r[1], r[2], r[3]) for r in rows}


def _insert_psd(
        con: duckdb.DuckDBPyConnection,
        rows: list[tuple],
) -> None:
    """rows: list of (sample_id, nperseg, fs, n_fft, psd_bytes)."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["sample_id", "nperseg", "fs", "n_fft", "psd"])
    con.register("_ir", df)
    try:
        con.execute(f"""
            INSERT INTO {_PSD_TABLE}
            SELECT sample_id, nperseg, fs, n_fft, psd FROM _ir
            ON CONFLICT DO NOTHING
        """)
    finally:
        con.unregister("_ir")


_precompute_chunks: np.ndarray | None = None


def _fft_block_worker(args: tuple) -> tuple:
    """Worker: compute the FFT spectra of one block of samples.

    args: ((detrend, window), positions)
    returns: ((detrend, window), [(position, fs, n_signal, amplitude_bytes), …])
    """
    key, positions = args
    detrend, window = key
    rows = []
    for pos in positions:
        chunk = _precompute_chunks[pos]
        _, amplitude, fs = compute_fft_spectrum(chunk, detrend=detrend, window=window)
        rows.append((int(pos), float(fs), int(len(chunk)), amplitude.tobytes()))
    return key, rows


def _psd_block_worker(args: tuple) -> tuple:
    """Worker: compute the Welch spectra of one block of samples.

    args: (nperseg, positions)
    returns: (nperseg, [(position, fs, n_fft, psd_bytes), …])
    """
    nperseg, positions = args
    rows = []
    for pos in positions:
        chunk = _precompute_chunks[pos]
        _, psd, fs = compute_psd_spectrum(chunk, nperseg=int(nperseg))
        rows.append((int(pos), float(fs), min(int(nperseg), len(chunk)), psd.tobytes()))
    return nperseg, rows


def _blocks(key, positions: np.ndarray, size: int = _PRECOMPUTE_BLOCK) -> list[tuple]:
    return [(key, positions[i:i + size]) for i in range(0, positions.size, size)]


def _run_precompute(worker, args_list: list, *, n_jobs: int, total: int, initial: int, desc: str):
    """Run *worker* over *args_list* in a fork()'d pool, yielding ``(key, rows)``."""
    try:
        mp_ctx = multiprocessing.get_context("fork")
    except Exception:
        mp_ctx = None
    with ProcessPoolExecutor(max_workers=max(1, n_jobs), mp_context=mp_ctx) as pool:
        with tqdm(total=total, initial=initial, desc=desc, leave=False) as pbar:
            for key, rows in pool.map(worker, args_list):
                pbar.update(len(rows))
                yield key, rows


def _precompute_fft_spectra(
        data: pd.DataFrame,
        *,
        detrend: bool,
        window: bool,
        n_jobs: int,
        con: duckdb.DuckDBPyConnection,
) -> _Spectra:
    """Compute (or load) the FFT spectrum of every sample in *data*.

    *data* must already be deduplicated by "sample_id"; the returned _Spectra is
    aligned to its row order.
    """
    global _precompute_chunks

    sample_ids = data["sample_id"].to_numpy()
    cached = _get_cached_fft(con, list(sample_ids), detrend, window)
    rows: list = [cached.get(sid) for sid in sample_ids]
    missing = np.flatnonzero([r is None for r in rows])

    if missing.size:
        _precompute_chunks = data["data"].to_numpy()
        pending: list[tuple] = []
        try:
            for (det, win), block in _run_precompute(
                    _fft_block_worker, _blocks((bool(detrend), bool(window)), missing),
                    n_jobs=n_jobs, total=len(sample_ids), initial=len(cached),
                    desc="  FFT spectra",
            ):
                for pos, fs, n_signal, blob in block:
                    rows[pos] = _spectrum_from_row(fs, n_signal, blob)
                    pending.append((sample_ids[pos], det, win, fs, n_signal, blob))
                if len(pending) >= 4096:
                    _insert_fft(con, pending)
                    pending = []
        finally:
            _precompute_chunks = None
        _insert_fft(con, pending)
        con.execute("CHECKPOINT")

    return _Spectra.from_rows(rows)


def _precompute_psd_spectra(
        data: pd.DataFrame,
        *,
        nperseg_values: tuple[int, ...],
        n_jobs: int,
        con: duckdb.DuckDBPyConnection,
) -> dict[int, _Spectra]:
    """Compute (or load) the Welch spectra of every sample in *data*.

    *data* must already be deduplicated by "sample_id"; every returned _Spectra
    is aligned to its row order.
    """
    global _precompute_chunks

    sample_ids = data["sample_id"].to_numpy()
    total = len(sample_ids) * len(nperseg_values)
    per_nperseg: dict[int, list] = {}
    args_list: list[tuple] = []
    n_cached = 0
    for nperseg in nperseg_values:
        cached = _get_cached_psd(con, list(sample_ids), nperseg)
        rows = [cached.get(sid) for sid in sample_ids]
        per_nperseg[int(nperseg)] = rows
        n_cached += len(cached)
        args_list += _blocks(int(nperseg), np.flatnonzero([r is None for r in rows]))

    if args_list:
        _precompute_chunks = data["data"].to_numpy()
        pending: list[tuple] = []
        try:
            for nperseg, block in _run_precompute(
                    _psd_block_worker, args_list, n_jobs=n_jobs,
                    total=total, initial=n_cached, desc="  PSD spectra",
            ):
                for pos, fs, n_fft, blob in block:
                    per_nperseg[nperseg][pos] = _spectrum_from_row(fs, n_fft, blob)
                    pending.append((sample_ids[pos], nperseg, fs, n_fft, blob))
                if len(pending) >= 4096:
                    _insert_psd(con, pending)
                    pending = []
        finally:
            _precompute_chunks = None
        _insert_psd(con, pending)
        con.execute("CHECKPOINT")

    return {n: _Spectra.from_rows(rows) for n, rows in per_nperseg.items()}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary classification metrics, derived from the confusion matrix."""
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[False, True]).ravel())
    n = tn + fp + fn + tp
    counts = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    true_sum = counts.sum(axis=1)
    pred_sum = counts.sum(axis=0)
    n_correct = np.trace(counts)
    n_samples = pred_sum.sum()
    cov_ytyp = n_correct * n_samples - true_sum @ pred_sum
    cov_ypyp = n_samples ** 2 - pred_sum @ pred_sum
    cov_ytyt = n_samples ** 2 - true_sum @ true_sum
    f1_denom = (tp + fn) + (tp + fp)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": (2.0 * tp) / f1_denom if f1_denom else 0.0,
        "mcc": 0.0 if cov_ypyp * cov_ytyt == 0 else float(cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)),
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
    """Majority-vote *predicted* within a centred window, per recording id."""
    if length <= 1:
        return predicted
    try:
        return _sliding_window_vote(data, predicted, length, threshold)
    except Exception:
        pass
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


def _sliding_window_vote(
        data: pd.DataFrame,
        predicted: pd.Series,
        length: int,
        threshold: float,
) -> pd.Series:
    """Vectorised equivalent of the per-group rolling majority vote.

    Computes the same per-group ``rolling(center=True, min_periods=1).mean()``
    from one grouped prefix sum, instead of building a rolling object per
    recording.  Window sums are small integers, so summing them a different way
    cannot change a single bit of the resulting means.
    """
    ids = data["id"].to_numpy()
    values = predicted.to_numpy().astype(np.int64)
    n = values.size

    codes = pd.factorize(ids)[0]
    if (codes < 0).any():
        raise ValueError("cannot group rows with a null id")

    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.concatenate(([0], np.cumsum(sizes)))
    base = np.repeat(starts[:-1], sizes)
    within = np.arange(n) - base
    group_size = np.repeat(sizes, sizes)

    prefix = np.concatenate(([0], np.cumsum(values[order])))
    lo = base + np.maximum(within - length // 2, 0)
    hi = base + np.minimum(within + (length - 1) // 2 + 1, group_size)

    voted = (prefix[hi] - prefix[lo]) / (hi - lo) >= threshold
    out = np.empty(n, dtype=bool)
    out[order] = voted
    return pd.Series(out, index=data.index)


# ---------------------------------------------------------------------------
# SNR computation over precomputed spectra
# ---------------------------------------------------------------------------
_SNR_BLOCK = 4096


def _masked_reduce(matrix: np.ndarray, mask: np.ndarray, reducer) -> np.ndarray:
    """Row-wise ``reducer(matrix[i][mask[i]])`` for every row of *matrix*.

    Rows are grouped by how many entries they select; each group is gathered
    into a dense ``(rows, k)`` array and reduced along axis 1 with a single
    numpy call.  A gathered row is a contiguous copy of exactly the values the
    per-row expression would select, in the same order, so ``np.mean`` sums them
    in the same pairwise order and ``np.median`` picks the same order statistics
    — the result is identical to the row-at-a-time version, not merely close.
    Rows selecting nothing keep NaN, which every caller treats as a failure.
    """
    out = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return out
    counts = np.bincount(rows, minlength=matrix.shape[0])
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for k in np.unique(counts[counts > 0]):
        selected = np.flatnonzero(counts == k)
        taken = cols[offsets[selected][:, None] + np.arange(k)]
        out[selected] = reducer(matrix[selected[:, None], taken], axis=1)
    return out


def _noise_floor_block(
        freqs: np.ndarray,
        values: np.ndarray,
        noise: np.ndarray,
        usable: np.ndarray,
        *,
        target_freq: float,
        noise_model: str,
        guard_band: float,
        noise_band: float,
) -> np.ndarray:
    """Per-row noise floor at *target_freq*; NaN where it cannot be estimated."""
    if noise_model == "median":
        return _masked_reduce(values, noise & usable[:, None], np.median)

    if noise_model == "interpolate":
        window = noise & usable[:, None]
        lo = window & (freqs >= target_freq - guard_band - noise_band) & (freqs < target_freq - guard_band)
        hi = window & (freqs > target_freq + guard_band) & (freqs <= target_freq + guard_band + noise_band)
        f_lo = _masked_reduce(freqs, lo, np.mean)
        f_hi = _masked_reduce(freqs, hi, np.mean)
        s_lo = _masked_reduce(values, lo, np.median)
        s_hi = _masked_reduce(values, hi, np.median)
        with np.errstate(all="ignore"):
            log_f_lo = np.log(f_lo)
            log_s_lo = np.log(s_lo)
            slope = (np.log(s_hi) - log_s_lo) / (np.log(f_hi) - log_f_lo)
            floor = np.exp(log_s_lo + slope * (np.log(target_freq) - log_f_lo))
            floor[~((s_lo > 0) & (s_hi > 0) & (f_lo > 0) & (f_hi > 0))] = np.nan
        return floor

    if noise_model == "spline":
        return _spline_noise_floor(freqs, values, noise, usable, target_freq)

    raise ValueError(
        f"Unknown noise_model '{noise_model}'. Choose from 'median', 'interpolate', 'spline'."
    )


def _spline_noise_floor(
        freqs: np.ndarray,
        values: np.ndarray,
        noise: np.ndarray,
        usable: np.ndarray,
        target_freq: float,
) -> np.ndarray:
    """Smoothing-spline noise floor, fitted one row at a time.

    FITPACK fits a spline to a single curve, so unlike the other noise models
    this one has no batched form.  Selecting the bins to fit and counting them
    still happens for the whole block at once; only the fit itself is per row.
    """
    out = np.full(freqs.shape[0], np.nan, dtype=np.float64)
    fit_bins = noise & (values > 0) & (freqs > 0)
    n_bins = fit_bins.sum(axis=1)
    log_target = np.log(target_freq)
    for i in np.flatnonzero(usable & (n_bins >= 4)):
        row = fit_bins[i]
        try:
            spline = UnivariateSpline(
                np.log(freqs[i][row]), np.log(values[i][row]), k=3, s=int(n_bins[i]),
            )
            floor = float(np.exp(float(spline(log_target))))
        except Exception:
            continue
        if floor > 0:
            out[i] = floor
    return out


def _snr_block(
        freqs: np.ndarray,
        values: np.ndarray,
        *,
        target_frequency: float,
        signal_delta: float,
        spectrum_min_freq: float,
        spectrum_max_freq: float,
        harmonics_mask: bool,
        noise_model: str,
        guard_band: float,
        noise_band: float,
        min_noise_bins: int,
) -> np.ndarray:
    """Peak-to-noise-floor SNR for a block of spectra; NaN where undefined."""
    in_band = (freqs >= spectrum_min_freq) & (freqs <= spectrum_max_freq)
    off_target = np.abs(freqs - target_frequency) > signal_delta
    target = in_band & ~off_target
    noise = in_band & off_target
    if harmonics_mask:
        for harmonic in range(2, 4):
            noise &= np.abs(freqs - target_frequency * harmonic) > signal_delta

    out = np.full(freqs.shape[0], np.nan, dtype=np.float64)
    usable = target.any(axis=1) & (noise.sum(axis=1) >= min_noise_bins)
    if not usable.any():
        return out

    floor = _noise_floor_block(
        freqs, values, noise, usable,
        target_freq=target_frequency, noise_model=noise_model,
        guard_band=guard_band, noise_band=noise_band,
    )
    estimated = usable & (floor > 0)
    peak = np.max(values, axis=1, where=target, initial=-np.inf)
    out[estimated] = peak[estimated] / floor[estimated]
    return out


def _snr_from_spectra(spectra: "_Spectra", *, min_noise_bins: int, **params) -> np.ndarray:
    """Per-sample SNR for every spectrum in *spectra*, evaluated in blocks."""
    n = len(spectra)
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, _SNR_BLOCK):
        stop = min(start + _SNR_BLOCK, n)
        out[start:stop] = _snr_block(
            spectra.freqs[start:stop], spectra.values[start:stop],
            min_noise_bins=min_noise_bins, **params,
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
# Per-method SNR: one value per sample, for the whole dataset
# ---------------------------------------------------------------------------

def _snr_all_fft(params: dict) -> np.ndarray:
    return _snr_from_spectra(
        _fft_spectra,
        min_noise_bins=4,
        **{k: params[k] for k in (
            "signal_delta", "spectrum_min_freq", "spectrum_max_freq",
            "harmonics_mask", "noise_model", "guard_band", "noise_band",
            "target_frequency",
        )},
    )


def _snr_all_psd(params: dict) -> np.ndarray:
    return _snr_from_spectra(
        _psd_spectra[int(params["nperseg"])],
        min_noise_bins=5,
        **{k: params[k] for k in (
            "signal_delta", "spectrum_min_freq", "spectrum_max_freq",
            "harmonics_mask", "noise_model", "guard_band", "noise_band",
            "target_frequency",
        )},
    )


def _snr_all_goertzel(params: dict) -> np.ndarray:
    """Per-sample Goertzel SNR.  No precompute — recomputed per trial."""
    chunks = _goertzel_chunks
    out = np.empty(len(chunks), dtype=np.float64)
    for i in range(len(chunks)):
        out[i] = compute_snr_goertzel(
            chunks[i],
            target_freq=float(params["target_frequency"]),
            spectrum_min_freq=float(params["spectrum_min_freq"]),
            spectrum_max_freq=float(params["spectrum_max_freq"]),
            guard_band=float(params["guard_band"]),
            noise_n_probes=int(params["noise_n_probes"]),
            probe_split=float(params["probe_split"]),
            noise_model=str(params["noise_model"]),
        )
    return out


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def _run_trial(snr_fn, params: dict) -> Tuple[float, dict, dict]:
    """Score one parameter set over every evaluation run.

    A sample's SNR depends only on the sample and the trial parameters — not on
    which split it happens to be in — so it is computed once for the whole
    dataset and then indexed per run, instead of being recomputed for every
    train and test split of every run.
    """
    snr_all = snr_fn(params)
    sw_len = params["sliding_window_length"]
    sw_thr = params["sliding_window_threshold"]

    all_stats_test: list[dict] = []
    all_stats_train: list[dict] = []

    for train_data, test_data, train_pos, test_pos in _optuna_splits:
        snr_train = snr_all[train_pos]
        snr_test = snr_all[test_pos]

        best_thr, _ = find_best_threshold_snr(
            snr_train,
            train_data["trainride"].to_numpy().astype(bool),
            metric=_optuna_metric,
        )

        stats_train = _evaluate_with_snr(train_data, snr_train, best_thr, sw_len, sw_thr)
        stats_test = _evaluate_with_snr(test_data, snr_test, best_thr, sw_len, sw_thr)
        stats_train["snr_threshold"] = float(best_thr)
        all_stats_test.append(stats_test)
        all_stats_train.append(stats_train)

    if len(all_stats_test) == 1:
        return all_stats_test[0]["f1"], all_stats_test[0], all_stats_train[0]

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


def _prepare_evaluation(
        *,
        dataset_path: Path,
        n_jobs: int,
        precompute_fn,
        runs: int,
        test_frac: float,
        seed_int: int,
        metric: str,
        verbose: bool = True,
) -> pd.DataFrame:
    """Load a dataset, precompute its spectra and publish the evaluation splits.

    Shared by the Optuna search and by the ``fast-evaluation tmd`` CLI so that
    scoring a parameter set uses exactly the same code path in both: the splits,
    the spectra and the metric all land in the module globals that
    :func:`_run_trial` reads.  Returns the dataset metadata frame.
    """
    if verbose:
        typer.echo("  Loading dataset …")
    full_data = read_pickle(dataset_path)
    full_data = _attach_sample_ids(full_data)
    dataset_meta = get_metadata_from_dataset(dataset_path)

    if verbose:
        typer.echo("  Dataset class distribution:")
        _print_dataset_stats(full_data, "Full dataset")

    unique_samples = full_data.drop_duplicates(subset="sample_id")
    sample_position = pd.Series(
        np.arange(len(unique_samples), dtype=np.int64),
        index=unique_samples["sample_id"].to_numpy(),
    )

    fft_spec = None
    psd_spec = None
    chunks = None
    if precompute_fn is not None:
        con, _ = _open_spectrum_cache(dataset_path)
        try:
            fft_spec, psd_spec = precompute_fn(unique_samples, n_jobs, con)
        finally:
            con.close()
    else:
        chunks = unique_samples["data"].to_numpy()
    del unique_samples
    full_data = full_data.drop(columns="data")

    if runs > 1:
        split_seeds = [seed_int + i for i in range(runs)]
        splits = [train_test_split(full_data, test_frac=test_frac, random_state=s, split_column='id') for s in
                  split_seeds]
        if verbose:
            typer.echo(f"  Multi-run mode: {runs} runs with different train/test splits")
            for i, (tr, te) in enumerate(splits, start=1):
                _print_dataset_stats(tr, f"Train set (run {i}/{runs})")
                _print_dataset_stats(te, f"Test set  (run {i}/{runs})")
    else:
        splits = [train_test_split(full_data, test_frac=test_frac, random_state=seed_int, split_column='id')]
        if verbose:
            _print_dataset_stats(splits[0][0], "Train set")
            _print_dataset_stats(splits[0][1], "Test set")

    def _positions(split: pd.DataFrame) -> np.ndarray:
        return sample_position.loc[split["sample_id"].to_numpy()].to_numpy()

    global _optuna_splits, _fft_spectra, _psd_spectra, _goertzel_chunks, _optuna_metric
    _optuna_splits = [(tr, te, _positions(tr), _positions(te)) for tr, te in splits]
    _fft_spectra = fft_spec
    _psd_spectra = psd_spec
    _goertzel_chunks = chunks
    _optuna_metric = metric

    return dataset_meta


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

    dataset_meta = _prepare_evaluation(
        dataset_path=dataset_path,
        n_jobs=n_jobs,
        precompute_fn=precompute_fn,
        runs=runs,
        test_frac=test_frac,
        seed_int=seed_int,
        metric=metric,
    )

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
                futures[pool.submit(_run_trial, run_trial_fn, payload)] = (t, cond_params)

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
                    pd.DataFrame([row], columns=csv_header).to_csv(results_csv_path, index=False, mode="a",
                                                                   header=False)
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
        run_trial_fn=_snr_all_fft,
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
        run_trial_fn=_snr_all_psd,
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
            full_data.drop_duplicates(subset="sample_id"),
            detrend=True, window=True, n_jobs=int(n_jobs or 1), con=con,
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
            full_data.drop_duplicates(subset="sample_id"),
            nperseg_values=PSD_NPERSEG_CHOICES, n_jobs=int(n_jobs or 1), con=con,
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
        splits = [train_test_split(full_data, test_frac=test_frac, random_state=s, split_column='id') for s in
                  split_seeds]
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
        run_trial_fn=_snr_all_goertzel,
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
