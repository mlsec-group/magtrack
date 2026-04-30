from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer


def load_and_prepare_coloc_results(csv_path: Path, score: str, column_suffix: str = '_test') -> pd.DataFrame:
    """Load a results CSV, extract trainride, and keep relevant columns."""
    df = pd.read_csv(csv_path)

    score_col = f"{score}{column_suffix}"
    if score_col not in df.columns:
        typer.echo(f"Error: column '{score_col}' not found in {csv_path.name}. "
                   f"Available: {[c for c in df.columns if c.endswith(f'{column_suffix}')]}", err=True)
        raise typer.Exit(code=1)

    df = df.dropna(subset=["trainride_start_seconds", "sampling_rate", score_col])
    df["trainride_start_seconds"] = df["trainride_start_seconds"].astype(int)
    df["sampling_rate"] = df["sampling_rate"].astype(float)

    # For each (trainride, sampling_rate) group keep the row with the best score
    idx = df.groupby(["trainride_start_seconds", "sampling_rate"])[score_col].idxmax()
    df = df.loc[idx].sort_values(["sampling_rate", "trainride_start_seconds"])
    return df


def load_tmd_results(csv_files: List[Path]) -> pd.DataFrame:
    """Load the results CSV files and concatenate them to a single DataFrame."""
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            typer.echo(f"Skipping unreadable file {f.name}: {e}", err=True)
            continue
        df["_source_file"] = f.name
        dfs.append(df)

    if not dfs:
        typer.echo("No readable CSV input files.", err=True)
        raise typer.Exit(code=1)

    full = pd.concat(dfs, ignore_index=True, sort=False)
    return full


_NUMERIC_COLS: List[str] = [
    "mcc_test", "f1_test", "accuracy_test",
    "sliding_window_length", "duration", "trainride_start_seconds",
]

_DEFAULT_GROUP_KEYS: List[str] = [
    "trainride_start_seconds",
    "duration",
    "sliding_window_length",
]


def load_best_tmd_results(
        csv_files: List[Path],
        best_metric: str = "mcc_test",
        group_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load TMD CSV results and return only the best row per group.

    For each combination of *group_keys* the single row with the highest
    *best_metric* value is kept.  The result is sorted by *group_keys*.

    Args:
        csv_files: Paths to optuna-search CSV files.
        best_metric: Column name used to rank rows within each group.
        group_keys: Columns to group by.  Defaults to
            ``["trainride_start_seconds", "duration", "sliding_window_length"]``.

    Returns:
        DataFrame with one best row per group, sorted by *group_keys*.

    Raises:
        typer.Exit: If no files could be read or required columns are missing.
    """
    if group_keys is None:
        group_keys = _DEFAULT_GROUP_KEYS

    full = load_tmd_results(csv_files)

    for col in _NUMERIC_COLS:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")

    required = list(group_keys) + [best_metric]
    missing = [c for c in required if c not in full.columns]
    if missing:
        typer.echo(f"Missing required columns: {missing}", err=True)
        raise typer.Exit(code=1)

    full_sorted = full.sort_values(best_metric, ascending=False, na_position="last")
    best = full_sorted.groupby(list(group_keys), as_index=False).first()
    best = best.sort_values(list(group_keys), na_position="last").reset_index(drop=True)
    return best
