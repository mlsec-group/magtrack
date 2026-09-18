from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

app = typer.Typer(help="Fast re-checks of the finished TMD and colocation parameter searches.")

# ---------------------------------------------------------------------------
# TMD: re-run the best parameter set from an existing search
# ---------------------------------------------------------------------------

_FAST_FLOAT_METRICS = ("f1", "mcc", "accuracy", "precision", "recall")
_FAST_COUNT_METRICS = (
    "TP", "FP", "TN", "FN", "samples", "samples_trainride", "samples_no_trainride",
)


def _as_bool(value) -> bool:
    """Parse a CSV cell as a bool.  ``bool("False")`` is True, hence this."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes")


def _params_from_row(row) -> dict:
    """Rebuild the parameter dict a trial was scored with from its CSV row."""
    return {
        "signal_delta": float(row["signal_delta"]),
        "spectrum_min_freq": float(row["spectrum_min_freq"]),
        "spectrum_max_freq": float(row["spectrum_max_freq"]),
        "harmonics_mask": _as_bool(row["harmonics_mask"]),
        "noise_model": str(row["noise_model"]),
        "guard_band": float(row["guard_band"]),
        "noise_band": float(row["noise_band"]),
        "target_frequency": float(row["target_frequency"]),
        "sliding_window_length": int(row["sliding_window_length"]),
        "sliding_window_threshold": float(row["sliding_window_threshold"]),
    }


def _compare_to_row(row, stats_test: dict, stats_train: dict, tolerance: float) -> list[dict]:
    """Compare recorded metrics against freshly computed ones.

    Counts are compared as the ints the CSV stored; the rest within *tolerance*.
    """
    import pandas as pd

    checks: list[dict] = []

    def add(name: str, recorded, computed, exact: bool):
        if recorded is None or (isinstance(recorded, float) and pd.isna(recorded)):
            return
        if exact:
            ok = int(recorded) == int(computed)
            delta = float(int(computed) - int(recorded))
        else:
            ok = bool(np.isclose(float(recorded), float(computed), rtol=tolerance, atol=tolerance))
            delta = float(computed) - float(recorded)
        checks.append({
            "name": name, "recorded": recorded, "computed": computed,
            "delta": delta, "ok": ok, "exact": exact,
        })

    for m in _FAST_FLOAT_METRICS:
        add(f"{m}_test", row.get(f"{m}_test"), stats_test[m], exact=False)
        add(f"{m}_train", row.get(f"{m}_train"), stats_train[m], exact=False)
    for m in _FAST_COUNT_METRICS:
        add(f"{m}_test", row.get(f"{m}_test"), stats_test[m], exact=True)
        add(f"{m}_train", row.get(f"{m}_train"), stats_train[m], exact=True)
    add("snr_threshold", row.get("snr_threshold"), stats_train["snr_threshold"], exact=False)
    return checks


def _format_checks(checks: list[dict]) -> list[str]:
    """Every recorded value beside the one just computed, as log lines."""
    name_w = max([len(c["name"]) for c in checks] + [len("metric")])
    header = f"  {'metric':<{name_w}}  {'recorded':>20}  {'computed':>20}  {'delta':>11}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for c in checks:
        if c["exact"]:
            # Counts are compared as the ints the CSV stored; show exactly that.
            rec_s = f"{int(c['recorded'])}"
            com_s = f"{int(c['computed'])}"
            delta_s = f"{int(c['delta']):+d}"
        else:
            rec_s = f"{float(c['recorded']):.12g}"
            com_s = f"{float(c['computed']):.12g}"
            delta_s = f"{c['delta']:+.3e}"
        flag = "" if c["ok"] else "   <-- MISMATCH"
        lines.append(f"  {c['name']:<{name_w}}  {rec_s:>20}  {com_s:>20}  {delta_s:>11}{flag}")
    return lines


_ROW_HEADER = (
    f"  {'dataset':<18} {'trial':>6} {'sw':>3} {'noise_model':<12} "
    f"{'f1 recorded':>12} {'f1 computed':>12} {'mcc recorded':>13} {'mcc computed':>13}  status"
)


def _format_row(r: dict) -> str:
    def num(v):
        return "-" if v is None else f"{float(v):.6f}"

    return (
        f"  {r['dataset']:<18} {r['trial']:>6} {r['sw']:>3} {r['noise_model']:<12} "
        f"{num(r['f1_recorded']):>12} {num(r['f1_computed']):>12} "
        f"{num(r['mcc_recorded']):>13} {num(r['mcc_computed']):>13}  {r['status']}"
    )


_MEM_PER_DATASET_GIB = 3


def _default_dataset_jobs(n_datasets: int) -> int:
    """How many datasets to evaluate at once: bounded by cores, RAM and work."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1

    limit = cores
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    gib = int(line.split()[1]) / 1024 / 1024
                    limit = min(limit, max(1, int(gib / _MEM_PER_DATASET_GIB)))
                    break
    except OSError:
        pass
    return max(1, min(limit, n_datasets))


def _evaluate_one_dataset(task: dict) -> dict:
    """Score one dataset's parameter set, in its own process.

    ``_prepare_evaluation`` publishes the splits and spectra as module globals,
    so each dataset needs its own interpreter to be evaluated concurrently --
    which is exactly what a process pool gives, with no shared state to guard.
    """
    import pandas as pd
    from magtrack.cli.evaluate_tmd import (
        _precompute_fft_spectra,
        _prepare_evaluation,
        _run_trial,
        _snr_all_fft,
    )
    from magtrack.utils.utils import resolve_seed

    def _precompute(data, jobs, con):
        return _precompute_fft_spectra(data, detrend=True, window=True, n_jobs=jobs, con=con), None

    try:
        _prepare_evaluation(
            dataset_path=Path(task["dataset_path"]),
            n_jobs=task["n_jobs"],
            precompute_fn=_precompute,
            runs=task["runs"],
            test_frac=task["test_frac"],
            seed_int=resolve_seed(task["seed"]),
            metric=task["threshold_metric"],
            verbose=False,
        )
        _, stats_test, stats_train = _run_trial(_snr_all_fft, _params_from_row(pd.Series(task["row"])))
    except Exception as exc:  # reported by the parent, which owns the log
        return {"index": task["index"], "error": f"{type(exc).__name__}: {exc}"}

    return {
        "index": task["index"],
        "stats_test": {k: float(v) for k, v in stats_test.items()},
        "stats_train": {k: float(v) for k, v in stats_train.items()},
    }


@app.command("tmd")
def tmd(
        results_dir: Path = typer.Option(
            Path("results/tmd"), "--results-dir",
            help="Directory holding the optuna_search_fft_sw*.csv files from the parameter search.",
        ),
        dataset_dir: Path = typer.Option(
            Path("datasets/tmd_datasets"), "--dataset-dir",
            help="Directory holding the tmd_*.pkl datasets the search ran on.",
        ),
        output_csv: Optional[Path] = typer.Option(
            None, "--output-csv",
            help="Where to write the per-dataset verification summary. Default: <results-dir>/fast_evaluation.csv",
        ),
        log_file: Optional[Path] = typer.Option(
            None, "--log-file",
            help="Where to write the full per-metric comparison. Default: <results-dir>/fast_evaluation.log",
        ),
        select_metric: str = typer.Option(
            "mcc_test", "--metric",
            help="Column used to pick the best row per dataset (the search optimised mcc).",
        ),
        tolerance: float = typer.Option(
            1e-6, "--tolerance",
            help="Relative/absolute tolerance for a metric to count as reproduced.",
        ),
        dataset_jobs: int = typer.Option(
            0, "--dataset-jobs",
            help="Datasets to evaluate concurrently, one process each. 0 auto-sizes from "
                 "cores and free memory (~2.5 GiB per dataset).",
        ),
        n_jobs: int = typer.Option(
            0, "--n-jobs",
            help="Workers each dataset uses for its FFT precompute. 0 splits the cores "
                 "evenly across --dataset-jobs.",
        ),
        threshold_metric: str = typer.Option(
            "mcc", "--threshold-metric",
            help="Metric the SNR threshold is fitted on; must match the search run ('mcc' or 'f1').",
        ),
        dataset: Optional[List[str]] = typer.Option(
            None, "--dataset",
            help="Limit to these dataset files (e.g. tmd_s0_d5.pkl). Repeatable. Default: all found.",
        ),
) -> None:
    """Re-run the best TMD parameter set per dataset and check the results still match.

    The full search explores 5000 trials per dataset and sliding window.  This
    reads the CSVs it already produced, takes the best row for each dataset
    across all sliding windows, and scores just that one parameter set again
    through the same code path the search used.  No sampler is involved, so the
    numbers are expected to reproduce exactly.
    """
    # Imported lazily; the heavy evaluation stack is imported in the workers.
    import pandas as pd
    from magtrack.utils.loader import get_metadata_from_dataset

    if threshold_metric not in ("mcc", "f1"):
        typer.echo(f"--threshold-metric must be 'mcc' or 'f1', got '{threshold_metric}'.", err=True)
        raise typer.Exit(code=1)

    csv_paths = sorted(results_dir.glob("*.optuna_search_fft_sw*.csv"))
    if not csv_paths:
        typer.echo(f"No search results found in {results_dir}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Reading {len(csv_paths)} result file(s) from {results_dir}")
    frames = []
    for path in csv_paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            typer.echo(f"  skipping unreadable {path.name}: {exc}", err=True)
    if not frames:
        typer.echo("No readable result files.", err=True)
        raise typer.Exit(code=1)
    all_results = pd.concat(frames, ignore_index=True)

    if select_metric not in all_results.columns:
        typer.echo(f"Column '{select_metric}' is not in the results CSVs.", err=True)
        raise typer.Exit(code=1)

    all_results = all_results.dropna(subset=[select_metric])
    best_rows = all_results.loc[all_results.groupby("dataset_file")[select_metric].idxmax()]
    best_rows = best_rows.sort_values("dataset_file")
    if dataset:
        wanted = set(dataset)
        best_rows = best_rows[best_rows["dataset_file"].isin(wanted)]
        for name in sorted(wanted - set(best_rows["dataset_file"])):
            typer.echo(f"  no results for requested dataset {name}", err=True)

    typer.echo(f"Best row per dataset by {select_metric}: {len(best_rows)} dataset(s)")
    typer.echo(f"Tolerance: {tolerance:g}")

    out_path = output_csv if output_csv is not None else results_dir / "fast_evaluation.csv"
    log_path = log_file if log_file is not None else results_dir / "fast_evaluation.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")

    def logline(text: str = "") -> None:
        log.write(text + "\n")

    logline(f"Fast evaluation of {results_dir}")
    logline(f"Datasets from {dataset_dir}")
    logline(f"Best row per dataset by {select_metric}, tolerance {tolerance:g}, "
            f"SNR threshold fitted on {threshold_metric}")

    summary: list[dict] = []
    rows: list[dict] = []
    n_ok = n_mismatch = n_skipped = n_stale = 0

    # ── 1. Classify. The provenance checks are cheap (meta.csv only), so do them
    #        here and never hand a skipped or stale dataset to a worker.
    entries: list[dict] = []
    tasks: list[dict] = []

    for _, row in best_rows.iterrows():
        dataset_file = str(row["dataset_file"])
        dataset_path = dataset_dir / dataset_file
        sw = int(row["sliding_window_length"])
        recorded_metric = float(row[select_metric])
        trial_no = int(row["trial_number"]) if "trial_number" in row and not pd.isna(row["trial_number"]) else -1

        entry = {
            "row": row, "dataset_file": dataset_file, "dataset_path": dataset_path,
            "sw": sw, "trial": trial_no, "recorded_metric": recorded_metric,
            "status": None, "reason": "", "result": None,
        }

        if not dataset_path.is_file():
            entry["status"] = "skipped"
            entry["reason"] = f"dataset not found: {dataset_path}"
            entries.append(entry)
            continue

        recorded_samples = int(row["samples_test"]) + int(row["samples_train"])
        disk_samples = None
        try:
            meta = get_metadata_from_dataset(dataset_path)
            if "num_samples" in meta.columns:
                disk_samples = int(meta["num_samples"].iloc[0])
        except Exception as exc:
            typer.echo(f"  {dataset_file}: could not read metadata ({exc}); skipping the provenance check")

        if disk_samples is not None and abs(disk_samples - recorded_samples) > 2:
            entry["status"] = "stale"
            entry["reason"] = (f"dataset has {disk_samples} samples, "
                               f"results recorded {recorded_samples}")
            entries.append(entry)
            continue

        tasks.append({
            "index": len(entries),
            "dataset_path": str(dataset_path),
            "row": row.to_dict(),
            "runs": int(row["n_runs"]) if "n_runs" in row and not pd.isna(row["n_runs"]) else 1,
            "test_frac": float(row["test_frac"]),
            "seed": str(row["seed"]),
            "threshold_metric": threshold_metric,
            "n_jobs": 1,  # filled in once the pool size is known
        })
        entries.append(entry)

    # ── 2. Evaluate. Each dataset is loaded and scored in its own process: the
    #        load and spectrum read dominate the runtime and are per-dataset
    #        serial, so spreading datasets across processes is what actually
    #        parallelises this.
    if tasks:
        pool_size = dataset_jobs if dataset_jobs > 0 else _default_dataset_jobs(len(tasks))
        pool_size = max(1, min(pool_size, len(tasks)))
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = os.cpu_count() or 1
        per_task_jobs = n_jobs if n_jobs > 0 else max(1, cores // pool_size)
        for t in tasks:
            t["n_jobs"] = per_task_jobs

        typer.echo(f"Evaluating {len(tasks)} dataset(s): {pool_size} in parallel, "
                   f"{per_task_jobs} precompute worker(s) each")
        logline(f"Parallelism: {pool_size} dataset(s) at a time, {per_task_jobs} precompute worker(s) each")

        done = 0
        if pool_size == 1:
            for task in tasks:
                result = _evaluate_one_dataset(task)
                entries[result["index"]]["result"] = result
                done += 1
                e = entries[result["index"]]
                typer.echo(f"  [{done}/{len(tasks)}] {e['dataset_file']} (trial {e['trial']}, sw {e['sw']})")
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=pool_size, mp_context=ctx) as pool:
                futures = {pool.submit(_evaluate_one_dataset, task): task for task in tasks}
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        # A worker dying (most often the OOM killer) must cost that
                        # one dataset, not the whole run.
                        result = {
                            "index": task["index"],
                            "error": f"{type(exc).__name__}: {exc}. Lower --dataset-jobs "
                                     f"if the machine ran out of memory.",
                        }
                    entries[result["index"]]["result"] = result
                    done += 1
                    e = entries[result["index"]]
                    typer.echo(f"  [{done}/{len(tasks)}] {e['dataset_file']} (trial {e['trial']}, sw {e['sw']})")

    # ── 3. Emit, in dataset order, so the log and table read the same whatever
    #        order the workers happened to finish in.
    for entry in entries:
        row = entry["row"]
        dataset_file = entry["dataset_file"]
        sw = entry["sw"]
        recorded_metric = entry["recorded_metric"]

        base_row = {
            "dataset": dataset_file, "trial": entry["trial"], "sw": sw,
            "noise_model": str(row["noise_model"]),
            "f1_recorded": float(row["f1_test"]), "f1_computed": None,
            "mcc_recorded": float(row["mcc_test"]), "mcc_computed": None,
            "status": "?",
        }

        logline("")
        logline("=" * 78)
        logline(f"{dataset_file}   trial {entry['trial']}   sliding window {sw}")
        logline("=" * 78)
        logline("parameters:")
        for k, v in sorted(_params_from_row(row).items()):
            logline(f"  {k} = {v}")

        if entry["status"] in ("skipped", "stale"):
            logline(f"status: {entry['status']} -- {entry['reason']}")
            base_row["status"] = entry["status"]
            rows.append(base_row)
            summary.append({
                "dataset_file": dataset_file, "sliding_window_length": sw,
                "status": entry["status"], "reason": entry["reason"],
                "select_metric": select_metric, "recorded": recorded_metric,
                "computed": None, "max_abs_delta": None, "checks": 0, "failed_checks": 0,
            })
            if entry["status"] == "stale":
                n_stale += 1
            else:
                n_skipped += 1
            continue

        result = entry["result"] or {"error": "not evaluated"}
        if "error" in result:
            logline(f"status: error -- {result['error']}")
            base_row["status"] = "error"
            rows.append(base_row)
            n_mismatch += 1
            summary.append({
                "dataset_file": dataset_file, "sliding_window_length": sw,
                "status": "error", "reason": result["error"],
                "select_metric": select_metric, "recorded": recorded_metric,
                "computed": None, "max_abs_delta": None, "checks": 0, "failed_checks": 0,
            })
            continue

        stats_test = result["stats_test"]
        stats_train = result["stats_train"]
        checks = _compare_to_row(row, stats_test, stats_train, tolerance)
        failed = [c for c in checks if not c["ok"]]
        max_delta = max((abs(c["delta"]) for c in checks), default=0.0)
        computed_metric = float(stats_test[select_metric[:-5]]) if select_metric.endswith("_test") else float("nan")

        if failed:
            logline(f"status: mismatch -- {len(failed)} of {len(checks)} metrics differ "
                    f"(max |delta| {max_delta:.3e})")
            n_mismatch += 1
            status = "mismatch"
        else:
            logline(f"status: ok -- all {len(checks)} metrics reproduced "
                    f"(max |delta| {max_delta:.3e})")
            n_ok += 1
            status = "ok"
        logline("")
        for line in _format_checks(checks):
            logline(line)

        base_row["f1_computed"] = float(stats_test["f1"])
        base_row["mcc_computed"] = float(stats_test["mcc"])
        base_row["status"] = status
        rows.append(base_row)

        summary.append({
            "dataset_file": dataset_file, "sliding_window_length": sw,
            "status": status, "reason": "",
            "select_metric": select_metric, "recorded": recorded_metric,
            "computed": computed_metric, "max_abs_delta": max_delta,
            "checks": len(checks), "failed_checks": len(failed),
        })

    pd.DataFrame(summary).to_csv(out_path, index=False)

    typer.echo("")
    typer.echo(_ROW_HEADER)
    typer.echo("  " + "-" * (len(_ROW_HEADER) - 2))
    for r in rows:
        typer.echo(_format_row(r))

    logline("")
    logline("=" * 78)
    logline(f"reproduced: {n_ok}   mismatched: {n_mismatch}   stale dataset: {n_stale}   skipped: {n_skipped}")
    log.close()

    typer.echo("")
    typer.echo(f" reproduced: {n_ok}   mismatched: {n_mismatch}   stale dataset: {n_stale}   skipped: {n_skipped}")
    typer.echo(f" summary: {out_path}")
    typer.echo(f" log:     {log_path}")

    if n_mismatch or n_stale:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
