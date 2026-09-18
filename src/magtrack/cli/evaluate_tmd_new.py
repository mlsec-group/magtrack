from __future__ import annotations

import datetime
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List

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

from magtrack.utils.evaluation import train_test_split, compute_threshold_candidates
from magtrack.utils.loader import read_pickle

app = typer.Typer(help="Hyperparameter search for FFT / PSD transport mode detectors.")

# ---------------------------------------------------------------------------
# Worker-process globals
# ---------------------------------------------------------------------------
_optuna_train_data: pd.DataFrame | None = None
_optuna_test_data: pd.DataFrame | None = None
_optuna_all_splits: list | None = None
_optuna_metric: str = "mcc"


def _optuna_worker_init(train_data: pd.DataFrame, test_data: pd.DataFrame, metric: str = "mcc") -> None:
    global _optuna_train_data, _optuna_test_data, _optuna_metric
    _optuna_train_data = train_data
    _optuna_test_data = test_data
    _optuna_metric = metric


def _optuna_worker_init_multi(splits: list, metric: str = "mcc") -> None:
    global _optuna_all_splits, _optuna_train_data, _optuna_test_data, _optuna_metric
    _optuna_all_splits = splits
    _optuna_metric = metric
    if splits:
        _optuna_train_data, _optuna_test_data = splits[0]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of classification metrics using sklearn."""
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
    """Smooth per-ID predictions with a centred rolling majority vote."""
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
# SNR computation helpers
# ---------------------------------------------------------------------------

def _compute_snr_values_fft(
        data: pd.DataFrame,
        *,
        signal_delta: float,
        spectrum_min_freq: float,
        spectrum_max_freq: float,
        harmonics_mask: bool,
        noise_model: str,
        guard_band: float,
        noise_band: float,
        detrend: bool,
        window: bool,
        target_frequency: float,
) -> pd.Series:
    """Compute per-sample FFT SNR values.  Non-computable samples return NaN."""
    from magtrack.utils.tmd_functions import compute_snr_fft

    return data["data"].apply(
        compute_snr_fft,
        target_freq=target_frequency,
        signal_delta=signal_delta,
        spectrum_min_freq=spectrum_min_freq,
        spectrum_max_freq=spectrum_max_freq,
        harmonics_mask=harmonics_mask,
        noise_model=noise_model,
        guard_band=guard_band,
        noise_band=noise_band,
        detrend=detrend,
        window=window,
    )


def _compute_snr_values_psd(
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
) -> pd.Series:
    """Compute per-sample Welch PSD SNR values.  Non-computable samples return NaN."""
    from magtrack.utils.tmd_functions import compute_snr_psd

    return data["data"].apply(
        compute_snr_psd,
        target_freq=target_frequency,
        signal_delta=signal_delta,
        spectrum_min_freq=spectrum_min_freq,
        spectrum_max_freq=spectrum_max_freq,
        harmonics_mask=harmonics_mask,
        noise_model=noise_model,
        guard_band=guard_band,
        noise_band=noise_band,
        nperseg=nperseg,
    )


def _find_best_snr_threshold(
        snr_values: np.ndarray,
        y_true: np.ndarray,
        metric: str = "mcc",
) -> Tuple[float, float]:
    """Find the optimal SNR threshold by sweeping midpoints of sorted unique values.

    The decision rule is ``snr >= threshold → predict positive (trainride=True)``.
    NaN / inf SNR values are treated as non-detections (predicted False).

    Returns
    -------
    best_threshold : float
    best_metric_value : float
    """
    finite_mask = np.isfinite(snr_values)
    snr_finite = snr_values[finite_mask]

    if len(snr_finite) == 0:
        return 1.0, 0.0

    candidates = compute_threshold_candidates(snr_finite)

    best_val = -np.inf
    best_thr = float(candidates[0])

    for thr in candidates:
        # NaN >= thr evaluates to False in numpy → correctly treated as non-detection
        y_pred = snr_values >= thr
        try:
            if metric == "mcc":
                val = float(matthews_corrcoef(y_true, y_pred))
            else:
                val = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception:
            val = 0.0
        if val > best_val:
            best_val = val
            best_thr = float(thr)

    return best_thr, best_val


def _evaluate_with_snr(
        data: pd.DataFrame,
        snr_values: pd.Series,
        snr_threshold: float,
        sliding_window_length: int,
        sliding_window_threshold: float,
) -> dict:
    """Classify samples by applying *snr_threshold* to raw SNR values, then return metrics."""
    predicted = snr_values >= snr_threshold
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
        "detrend", "window", "target_frequency",
    )}
    sw_len = params["sliding_window_length"]
    sw_thr = params["sliding_window_threshold"]

    snr_train = _compute_snr_values_fft(_optuna_train_data, **snr_kwargs)
    snr_test = _compute_snr_values_fft(_optuna_test_data, **snr_kwargs)

    best_thr, _ = _find_best_snr_threshold(
        snr_train.to_numpy(),
        _optuna_train_data["trainride"].to_numpy().astype(bool),
        _optuna_metric,
    )

    stats_train = _evaluate_with_snr(_optuna_train_data, snr_train, best_thr, sw_len, sw_thr)
    stats_test = _evaluate_with_snr(_optuna_test_data, snr_test, best_thr, sw_len, sw_thr)
    # Record the threshold that was learned from training data
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

    snr_train = _compute_snr_values_psd(_optuna_train_data, **snr_kwargs)
    snr_test = _compute_snr_values_psd(_optuna_test_data, **snr_kwargs)

    best_thr, _ = _find_best_snr_threshold(
        snr_train.to_numpy(),
        _optuna_train_data["trainride"].to_numpy().astype(bool),
        _optuna_metric,
    )

    stats_train = _evaluate_with_snr(_optuna_train_data, snr_train, best_thr, sw_len, sw_thr)
    stats_test = _evaluate_with_snr(_optuna_test_data, snr_test, best_thr, sw_len, sw_thr)
    # Record the threshold that was learned from training data
    stats_train["snr_threshold"] = float(best_thr)
    return stats_test["f1"], stats_test, stats_train


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def _run_trial_multi_runs(run_fn, params: dict) -> Tuple[float, dict, dict]:
    """Run *run_fn* across all stored splits and return mean-aggregated results."""
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
) -> None:
    if metric not in ("mcc", "f1"):
        typer.echo(f"Unknown metric '{metric}', must be 'mcc' or 'f1'. Falling back to 'mcc'.", err=True)
        exit(1)

    # Sampler
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
    results_csv_path = dataset_path.with_suffix(f".optuna_search_{label}.csv")

    # Load dataset
    typer.echo("  Loading dataset …")
    full_data = read_pickle(dataset_path)

    # Print dataset stats
    typer.echo("  Dataset class distribution:")
    _print_dataset_stats(full_data, "Full dataset")

    if multi_run:
        split_seeds = [seed_int + i for i in range(runs)]
        splits = [train_test_split(full_data, test_frac=test_frac, random_state=s) for s in split_seeds]
        typer.echo(f"  Multi-run mode: {runs} runs with different train/test splits")

        # Print stats for every train/test distribution
        for i, (tr, te) in enumerate(splits, start=1):
            _print_dataset_stats(tr, f"Train set (run {i}/{runs})")
            _print_dataset_stats(te, f"Test set  (run {i}/{runs})")

        train_data, test_data = splits[0]
    else:
        splits = None
        train_data, test_data = train_test_split(full_data, test_frac=test_frac, random_state=seed_int)

        _print_dataset_stats(train_data, "Train set")
        _print_dataset_stats(test_data, "Test set")

    # Build CSV header
    std_cols = (
            ["n_runs"]
            + [f"{m}_{s}_std" for m in ("f1", "mcc", "accuracy", "precision", "recall") for s in ("test", "train")]
            + ["snr_threshold_std"]
    ) if multi_run else []

    csv_header = (
            ["trial_number", "dataset_file", "test_frac", "seed"]
            + all_param_names
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

    # Worker pool — pass metric so workers can find the best SNR threshold
    if multi_run:
        _pool_init = _optuna_worker_init_multi
        _pool_initargs = (splits, metric)
    else:
        _pool_init = _optuna_worker_init
        _pool_initargs = (train_data, test_data, metric)

    try:
        mp_ctx = multiprocessing.get_context('fork')
    except Exception:
        mp_ctx = None

    with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp_ctx,
            initializer=_pool_init,
            initargs=_pool_initargs,
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
                    f1_test, stats_test, stats_train = fut.result()
                    objective_value = float(stats_test[metric])
                except Exception as exc:
                    typer.echo(f"\n  Trial failed: {exc}", err=True)
                    study.tell(t, state=optuna.trial.TrialState.FAIL)
                    pbar.update(1)
                    trials_done += 1
                    continue

                # Build user attributes with train/test metrics
                user_attrs: dict = {}
                for mk in ("f1", "mcc", "TP", "FP", "TN", "FN"):
                    user_attrs[f"{mk}_test"] = stats_test[mk]
                    user_attrs[f"{mk}_test_std"] = stats_test.get(mk + '_std', 0.0)
                    user_attrs[f"{mk}_train"] = stats_train[mk]
                    user_attrs[f"{mk}_train_std"] = stats_train.get(mk + '_std', 0.0)
                # snr_threshold is learned from train data — report once
                user_attrs["snr_threshold"] = stats_train["snr_threshold"]
                user_attrs["snr_threshold_std"] = stats_train.get("snr_threshold_std", 0.0)

                # Merge all params and their distributions for create_trial
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

                # Build CSV row
                row: dict = {
                    "trial_number": trial_counter,
                    "dataset_file": os.path.basename(dataset_path),
                    "test_frac": float(test_frac),
                    "seed": seed,
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

                # Test metrics
                for mk in ("f1", "mcc", "accuracy", "precision", "recall"):
                    row[f"{mk}_test"] = float(stats_test[mk])
                for mk in ("TP", "FP", "TN", "FN", "samples", "samples_trainride", "samples_no_trainride"):
                    row[f"{mk}_test"] = int(stats_test[mk])

                # Train metrics
                for mk in ("f1", "mcc", "accuracy", "precision", "recall"):
                    row[f"{mk}_train"] = float(stats_train[mk])
                for mk in ("TP", "FP", "TN", "FN", "samples", "samples_trainride", "samples_no_trainride"):
                    row[f"{mk}_train"] = int(stats_train[mk])

                # SNR threshold — learned from training data, applied to both sets
                row["snr_threshold"] = float(stats_train["snr_threshold"])

                # Multi-run std columns
                if multi_run:
                    row["n_runs"] = runs
                    for mk in ("f1", "mcc", "accuracy", "precision", "recall"):
                        row[f"{mk}_test_std"] = float(stats_test.get(f"{mk}_std", 0.0))
                        row[f"{mk}_train_std"] = float(stats_train.get(f"{mk}_std", 0.0))
                    row["snr_threshold_std"] = float(stats_train.get("snr_threshold_std", 0.0))

                trial_counter += 1

                try:
                    pd.DataFrame([row]).to_csv(results_csv_path, index=False, mode="a", header=False)
                except Exception as e:
                    typer.echo(f"\n  Failed to save trial result: {e}", err=True)

                pbar.update(1)
                trials_done += 1

    pbar.close()

    # Print best result
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

    # Print best trial params
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
        # Optuna settings
        n_trials: int = typer.Option(5000, help="Number of Optuna trials"),
        n_jobs: Optional[int] = typer.Option(multiprocessing.cpu_count() - 1,
                                             help="Number of parallel Optuna workers (default: cpu_count() - 1)"),
        sampler: str = typer.Option(
            "tpe",
            help="Optuna sampler: 'tpe' (Bayesian), 'random', or 'cmaes'",
        ),
        storage: Optional[str] = typer.Option(
            None,
            help="Optuna storage URL (e.g. 'sqlite:///optuna.db'). Defaults to in-memory.",
        ),
        target_frequency: float = typer.Option(16.7, help="Target frequency (Hz)"),
        # distribution bounds
        signal_delta_min: float = typer.Option(0.05, help="signal_delta lower bound (Hz)"),
        signal_delta_max: float = typer.Option(2.0, help="signal_delta upper bound (Hz)"),
        spectrum_min_freq_min: float = typer.Option(0.5, help="spectrum_min_freq lower bound (Hz)"),
        spectrum_min_freq_max: float = typer.Option(16.4, help="spectrum_min_freq upper bound (Hz)"),
        spectrum_max_freq_min: float = typer.Option(17.0, help="spectrum_max_freq lower bound (Hz)"),
        spectrum_max_freq_max: float = typer.Option(45.0, help="spectrum_max_freq upper bound (Hz)"),
        noise_models: List[str] = typer.Option(["median", "interpolate", "spline"],
                                               help="Noise model choices to include in the search. Provide multiple times to include more than one."),
        # other Options
        test_frac: float = typer.Option(0.3, help="Fraction of unique IDs held out as test set (0–1)"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split"),
        metric: str = typer.Option("mcc", help="Metric to optimise: 'mcc' (default) or 'f1'"),
        runs: int = typer.Option(10, help="Number of evaluation runs (1 = single run, >1 = average over N splits)"),
        # Sliding window majority vote
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
        # FFT-specific
        "detrend": optuna.distributions.CategoricalDistribution([True]),
        "window": optuna.distributions.CategoricalDistribution([True]),
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
    )


@app.command("optuna-search-psd")
def optuna_search_psd(
        dataset_path: Path = typer.Argument(..., help="Path to a tmd dataset (pkl)"),
        # Optuna settings
        n_trials: int = typer.Option(5000, help="Number of Optuna trials"),
        n_jobs: Optional[int] = typer.Option(multiprocessing.cpu_count() - 1,
                                             help="Number of parallel Optuna workers (default: cpu_count() - 1)"),
        sampler: str = typer.Option(
            "tpe",
            help="Optuna sampler: 'tpe' (Bayesian), 'random', or 'cmaes'",
        ),
        storage: Optional[str] = typer.Option(
            None,
            help="Optuna storage URL (e.g. 'sqlite:///optuna.db'). Defaults to in-memory.",
        ),
        target_frequency: float = typer.Option(16.7, help="Target frequency (Hz)"),
        # distribution bounds (make the search ranges configurable)
        signal_delta_min: float = typer.Option(0.05, help="signal_delta lower bound (Hz)"),
        signal_delta_max: float = typer.Option(2.0, help="signal_delta upper bound (Hz)"),
        spectrum_min_freq_min: float = typer.Option(0.5, help="spectrum_min_freq lower bound (Hz)"),
        spectrum_min_freq_max: float = typer.Option(16.4, help="spectrum_min_freq upper bound (Hz)"),
        spectrum_max_freq_min: float = typer.Option(17.0, help="spectrum_max_freq lower bound (Hz)"),
        spectrum_max_freq_max: float = typer.Option(45.0, help="spectrum_max_freq upper bound (Hz)"),
        noise_models: List[str] = typer.Option(["median", "interpolate", "spline"],
                                               help="Noise model choices to include in the search. Provide multiple times to include more than one."),
        # other Options
        test_frac: float = typer.Option(0.3, help="Fraction of unique IDs held out as test set (0–1)"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split"),
        metric: str = typer.Option("mcc", help="Metric to optimise: 'mcc' (default) or 'f1'"),
        runs: int = typer.Option(0, help="Number of evaluation runs (0 = single run, >1 = average over N splits)"),
        # Sliding window majority vote
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
        # PSD-specific
        "nperseg": optuna.distributions.IntDistribution(128, 512),
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
    )


if __name__ == "__main__":
    app()
