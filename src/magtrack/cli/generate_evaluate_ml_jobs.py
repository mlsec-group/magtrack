from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from magtrack.utils.loader import get_metadata_from_dataset

app = typer.Typer(help="Generate bash script for evaluate-ml-model.")


def _discover_datasets(
        datasets_path: Path,
        durations: list[int],
        sampling_rates: list[int],
        rolling_windows: list[int],
        trainride_starts: list[int],
) -> list[tuple[Path, dict]]:
    """Return (pkl_path, meta_row) pairs matching the filter criteria."""
    results: list[tuple[Path, dict]] = []
    for pkl_path in sorted(datasets_path.glob("*.pkl")):
        meta = get_metadata_from_dataset(pkl_path).iloc[0].to_dict()
        meta_duration = int(meta.get("duration", -1))
        meta_sr = int(float(meta.get("sampling_rate", -1)))
        meta_rw = int(meta.get("rolling_window", -1))
        meta_tr_start = int(meta.get("trainride_start_seconds", -1))
        if meta_duration not in durations:
            continue
        if meta_sr not in sampling_rates:
            continue
        if meta_rw not in rolling_windows:
            continue
        if meta_tr_start not in trainride_starts:
            continue
        results.append((pkl_path, meta))
    return results


def _result_exists(csv_path: Path, dataset_file: str) -> bool:
    """Check whether a result row for `dataset_file` already exists in the CSV."""
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if "dataset_file" not in df.columns:
        return False
    return bool((df["dataset_file"] == dataset_file).any())


def _eval_cmd_args(
        *,
        model_path: Path,
        hparams_path: Path,
        test_fraction: float,
        threshold: float,
        batch_size: int,
        seed: str,
        time_per_step: bool,
        n_jobs: int,
        results_dir: Path,
) -> list[str]:
    """Return the list of `ml-inference` flag fragments (no dataset)."""
    return [
        f'--model-path "{model_path}"',
        f'--hparams-path "{hparams_path}"',
        f"--test-fraction {test_fraction}",
        f"--threshold {threshold}",
        f"--batch-size {batch_size}",
        f'--seed "{seed}"',
        # f"--{'time-per-step' if time_per_step else 'no-time-per-step'}",
        f"--n-jobs {n_jobs}",
        f'--results-dir "{results_dir}"',
    ]


def _resolve_pending(
        datasets_path: Path,
        durations: list[int],
        sampling_rates: list[int],
        rolling_windows: list[int],
        trainride_starts: list[int],
        results_csv: Optional[Path],
) -> list[Path]:
    """Discover, filter (skip-done), and sort datasets cheapest-first.

    Echoes a TODO/SKIP line to stderr per dataset and exits if nothing is
    pending. Returns the ordered list of pending pkl paths.
    """
    all_datasets = _discover_datasets(
        datasets_path, durations, sampling_rates, rolling_windows, trainride_starts,
    )
    typer.echo(f"Found {len(all_datasets)} dataset(s) matching filters.", err=True)
    if not all_datasets:
        typer.echo("No datasets found. Nothing to do.", err=True)
        raise typer.Exit(code=0)

    pending: list[tuple[Path, dict]] = []
    if results_csv is not None:
        results_csv = results_csv.resolve()
    for pkl_path, meta in all_datasets:
        if results_csv is not None and _result_exists(results_csv, pkl_path.name):
            typer.echo(f"  SKIP (done): {pkl_path.name}", err=True)
        else:
            typer.echo(f"  TODO:        {pkl_path.name}", err=True)
            pending.append((pkl_path, meta))

    if not pending:
        typer.echo("\nAll tasks already completed. Nothing to submit.", err=True)
        raise typer.Exit(code=0)

    pending.sort(
        key=lambda item: int(item[1].get("duration", 0)) * int(float(item[1].get("sampling_rate", 0)))
    )
    typer.echo(f"\n{len(pending)} task(s) remaining out of {len(all_datasets)}.", err=True)
    return [p for p, _ in pending]


def _build_parallel_script(
        *,
        pkl_paths: list[Path],
        model_path: Path,
        hparams_path: Path,
        test_fraction: float,
        threshold: float,
        batch_size: int,
        seed: str,
        time_per_step: bool,
        n_jobs: int,
        results_dir: Path,
        python_sif: str,
        bind_path: str,
        module_mount_path: Optional[str],
        gpu_device: str,
        parallel_jobs: int,
        parallel_output_dir: Path,
) -> str:
    """Build a self-contained GNU-parallel bash script (no separate inputs file).

    Datasets are baked into the script as a bash array; the script writes a
    temporary commands file at runtime, hands it to `parallel`, and cleans up.
    """
    dataset_lines = "\n".join(f'  "{p}"' for p in pkl_paths)
    if module_mount_path:
        module_mount_path = f" --bind {module_mount_path}:/src"
    else:
        module_mount_path = ""
    eval_args = " ".join(_eval_cmd_args(
        model_path=model_path, hparams_path=hparams_path,
        test_fraction=test_fraction, threshold=threshold, batch_size=batch_size,
        seed=seed, time_per_step=time_per_step, n_jobs=n_jobs, results_dir=results_dir,
    ))

    script = f"""#!/usr/bin/env bash
set -euo pipefail

# Self-contained GNU-parallel runner for ml-inference
# Datasets are baked in below; commands are built per dataset and dispatched
# via `parallel`.

### Paths
SOURCE_SIF="{python_sif}"
PYTHON_SIF="{python_sif}"
BIND_PATH="{bind_path}"

### GNU Parallel settings
PARALLEL_JOBS={parallel_jobs}
OUTPUT_DIR="{parallel_output_dir}"
COMMANDS_FILE="${{OUTPUT_DIR}}/evaluate_ml_commands.txt"
PARALLEL_JOBLOG="${{OUTPUT_DIR}}/evaluate_ml_parallel_joblog.tsv"

mkdir -p "${{OUTPUT_DIR}}"
: > "${{COMMANDS_FILE}}"

# ── Datasets to process ──────────────────────────────────────────────────────
DATASETS=(
{dataset_lines}
)

if [[ ! -f "${{PYTHON_SIF}}" ]]; then
  echo "Warning: PYTHON_SIF does not exist: ${{PYTHON_SIF}}"
fi

EVAL_ARGS='{eval_args}'

for dataset_path in "${{DATASETS[@]}}"; do
  dataset_file="$(basename "${{dataset_path}}")"
  dataset_stem="${{dataset_file%.pkl}}"
  log_path="${{OUTPUT_DIR}}/${{dataset_stem}}.log"

  cmd="env CUDA_VISIBLE_DEVICES={gpu_device} apptainer exec --nv --bind \\"${{BIND_PATH}}\\"{module_mount_path} \\"${{PYTHON_SIF}}\\" ml-inference --dataset-path \\"${{dataset_path}}\\" ${{EVAL_ARGS}}"

  echo "${{cmd}} >> \\"${{log_path}}\\" 2>&1" >> "${{COMMANDS_FILE}}"
done

num_commands=$(wc -l < "${{COMMANDS_FILE}}")
if (( num_commands == 0 )); then
  echo "No commands generated."
  exit 1
fi

echo "Generated ${{num_commands}} commands in ${{COMMANDS_FILE}}"
echo "Running with GNU parallel (${{PARALLEL_JOBS}} jobs)..."

parallel --jobs "${{PARALLEL_JOBS}}" --joblog "${{PARALLEL_JOBLOG}}" < "${{COMMANDS_FILE}}"

echo "Done. GNU parallel job log: ${{PARALLEL_JOBLOG}}"
rm -f "${{COMMANDS_FILE}}"
"""
    return script


@app.command("shell")
def parallel(
        datasets_path: Path = typer.Argument(..., help="Directory containing .pkl and .meta.csv files."),
        model_path: Path = typer.Argument(..., help="Path to the trained model .pth file."),
        # Metadata filters
        duration: list[int] = typer.Option(
            [5, 10, 20, 30, 60],
            help="Filter datasets by chunk duration in seconds (meta: duration).",
        ),
        sampling_rate: list[int] = typer.Option(
            [10, 20, 40, 60],
            help="Filter datasets by sampling rate(s) in Hz (meta: sampling_rate).",
        ),
        rolling_window: list[int] = typer.Option(
            [1, 5, 10, 50, 100, 150],
            help="Filter datasets by rolling window(s) (meta: rolling_window).",
        ),
        trainride_start_seconds: list[int] = typer.Option(
            [0, 60, 300, 600, 900],
            help="Filter datasets by train-ride start offset in seconds (meta: trainride_start_seconds).",
        ),
        # evaluate-ml-model pass-through
        hparams_path: Path = typer.Option(
            Path("./coloc_model/model_hparams.yaml"),
            help="Path to the model hparams yaml.",
        ),
        test_fraction: float = typer.Option(0.3, help="Fraction of data held out as test set."),
        threshold: float = typer.Option(0.0, help="Logit threshold for binary classification."),
        batch_size: int = typer.Option(1024, help="Batch size for evaluation."),
        seed: str = typer.Option("magtrack", help="Random seed string."),
        time_per_step: bool = typer.Option(True, help="Synchronize CUDA per batch for accurate inference timing."),
        n_jobs: int = typer.Option(32,
                                   help="Worker processes for parallel signal stacking when building the eval dataset."),
        results_dir: Path = typer.Option(Path("results"), help="Directory where evaluation CSV files are saved."),
        # Skip-already-done filter
        results_csv: Optional[Path] = typer.Option(
            None,
            help="If set, skip datasets that already have a row in this CSV "
                 "(checked against the 'dataset_file' column).",
        ),
        # Container settings
        python_sif: str = typer.Option(
            "./python.sif",
            help="Path to the Apptainer/Singularity .sif image.",
        ),
        bind_path: str = typer.Option(
            "/path/to/colocation/datasets/:/data/",
            help="Bind path for apptainer exec.",
        ),
        module_mount_path: str = typer.Option(None,
                                              help="If set, the path within the container where the magtrack module is mounted to apptainer."),
        # GNU parallel settings
        parallel_jobs: int = typer.Option(4, help="Concurrent jobs for GNU parallel."),
        parallel_output_dir: Path = typer.Option(
            Path("results/logs_parallel"),
            help="Directory for the per-task log files and the parallel joblog.",
        ),
        gpu_device: str = typer.Option(0, help="GPU device to use."),
        # Output
        output: Optional[Path] = typer.Option(None, "-o", "--output",
                                              help="Write the bash script to this file (chmod +x)."),
) -> None:
    """Generate a self-contained GNU-parallel bash script for evaluate-ml-model."""
    datasets_path = datasets_path.resolve()

    pending_paths = _resolve_pending(
        datasets_path, duration, sampling_rate, rolling_window,
        trainride_start_seconds, results_csv,
    )

    script = _build_parallel_script(
        pkl_paths=pending_paths,
        model_path=model_path,
        hparams_path=hparams_path,
        test_fraction=test_fraction,
        threshold=threshold,
        batch_size=batch_size,
        seed=seed,
        time_per_step=time_per_step,
        n_jobs=n_jobs,
        results_dir=results_dir,
        python_sif=python_sif,
        bind_path=bind_path,
        module_mount_path=module_mount_path,
        gpu_device=gpu_device,
        parallel_jobs=parallel_jobs,
        parallel_output_dir=parallel_output_dir,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(script)
        output.chmod(output.stat().st_mode | 0o111)
        typer.echo(f"Bash script written to {output}", err=True)
    else:
        sys.stdout.write(script)


if __name__ == "__main__":
    app()
