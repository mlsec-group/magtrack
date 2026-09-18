"""Generate metric heatmaps from evaluation CSV files."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib import patches

from magtrack.utils.plot import configure_output_format, PARAMS, CB_color_cycle, get_page_width
from magtrack.utils.results import load_best_tmd_results

app = typer.Typer(help="Generate heatmaps for multiple metrics from one evaluation CSV.")

DEFAULT_METRICS = [
    "accuracy", "f1", "mcc", "precision",
    "recall"]


def format_metric_name(metric: str) -> str:
    """Formats internal metric names for clean plot labels."""
    overrides = {
        "mcc": "MCC",
        "f1": "F1 Score",
        "auroc": "AUROC",
        "negative_predictive_value": "Negative Predictive Value (NPV)",
        "npv": "NPV",
        "tp": "True Positives",
        "fp": "False Positives",
        "tn": "True Negatives",
        "fn": "False Negatives"
    }
    # Fallback to replacing underscores and title casing (e.g., 'accuracy' -> 'Accuracy')
    return overrides.get(metric.lower(), metric.replace("_", " ").title())


def format_duration_label(duration: int | str) -> str:
    if duration == 0 or duration == "all":
        return "full length"
    return f"{duration} s"


def configure_matplotlib(output_format: str):
    configure_output_format(output_format)


def parse_meta(name: str):
    w = re.search(r"window(\d+)", name)
    span = re.search(r"first(\d+)_(\d+)s", name)
    ch_legacy = re.search(r"window\d+_(\d+)s", name)
    dur_legacy = re.search(r"first(\d+)s", name)
    hz_all = re.findall(r"(\d+)Hz", name)

    if span:
        duration = int(span.group(1))
        chunk = int(span.group(2))
    else:
        duration = int(dur_legacy.group(1)) if dur_legacy else None
        chunk = int(ch_legacy.group(1)) if ch_legacy else None

    return {
        "window": int(w.group(1)) if w else None,
        "chunk": chunk,
        "duration": duration,
        "hz": int(hz_all[-1]) if hz_all else None,
    }


def draw_box(ax, row, col, color, lw=2):
    rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=color, lw=lw, zorder=5)
    ax.add_patch(rect)


def prepare_pivot_data(df, metric, chunks, durations):
    """Extracts and pivots the data for the given chunks and durations."""
    pivot_store = {}
    for ch in chunks:
        for dur in durations:
            sub = df[(df["chunk"] == ch) & (df["duration"] == dur)]
            if sub.empty:
                pivot_store[(ch, dur)] = None
                continue
            agg = sub.groupby(["window", "hz"])[metric].agg(["mean", "std"]).reset_index()
            pivot_mean = agg.pivot(index="window", columns="hz", values="mean").sort_index(ascending=True)
            pivot_std = agg.pivot(index="window", columns="hz", values="std").reindex_like(pivot_mean)
            pivot_store[(ch, dur)] = (pivot_mean, pivot_std)
    return pivot_store


def find_best_cell(pivot_store):
    """Finds the chunk, duration, and cell indices that contain the maximum metric value."""
    best_val = -np.inf
    best_cell = None
    for (ch_k, dur_k), entry in pivot_store.items():
        if entry is None:
            continue
        pm, _ = entry
        data_k = pm.values
        if np.isnan(data_k).all():
            continue
        flat_idx = int(np.nanargmax(data_k))
        val = data_k.flat[flat_idx]
        if val > best_val:
            best_val = val
            best_cell = (ch_k, dur_k, flat_idx // data_k.shape[1], flat_idx % data_k.shape[1])
    return best_cell


def choose_compact_grid(n_plots, max_cols):
    """Choose a compact near-square grid, constrained by max_cols."""
    if n_plots <= 0:
        return 0, 0

    best_rows, best_cols = 1, n_plots
    best_score = None
    max_allowed_cols = max(1, max_cols)

    for ncols in range(1, min(max_allowed_cols, n_plots) + 1):
        nrows = int(np.ceil(n_plots / ncols))
        empty = nrows * ncols - n_plots
        # Prefer squarer layouts, then fewer empty cells.
        score = (abs(nrows - ncols), empty)
        if best_score is None or score < best_score:
            best_rows, best_cols = nrows, ncols
            best_score = score

    return best_rows, best_cols


@app.command("tmd")
def tmd_heatmap(
        input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                         help="Directory containing tmd_s*_d*.optuna_search_*.csv files."),
        metric: str = typer.Option("mcc_test", help="Metric column to visualise."),
        out_dir: Path = typer.Option(Path("results"), help="Output directory."),
        output_format: str = typer.Option("pgf", help="Output file format: pgf|pdf"),
) -> None:
    """Generate a heatmap (duration x sliding_window_length) per trainride_start_seconds for TMD results."""
    output_format = output_format.lower()
    if output_format not in {"pgf", "pdf"}:
        raise typer.BadParameter("--output-format must be one of: pgf, pdf")

    configure_matplotlib(output_format)

    files = sorted(input_dir.glob("tmd_s*_d*.optuna_search_*.csv"))
    if not files:
        typer.echo(f"No optuna-search CSV files found in {input_dir}", err=True)
        raise typer.Exit(code=1)

    best = load_best_tmd_results(files, best_metric=metric)

    if metric not in best.columns:
        typer.echo(f"Metric '{metric}' not found. Available: {list(best.columns)}", err=True)
        raise typer.Exit(code=1)

    trainride_starts = sorted(best["trainride_start_seconds"].dropna().unique())
    display_metric = format_metric_name(metric.replace("_test", ""))

    ncols = len(trainride_starts)
    fig_w = get_page_width()
    fig_h = fig_w * 0.45
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(fig_w, fig_h))
    if ncols == 1:
        axes = [axes]

    # Find global best cell for highlighting
    best_val = -np.inf
    best_info = None
    for ts in trainride_starts:
        sub = best[best["trainride_start_seconds"] == ts]
        pivot = sub.pivot_table(index="sliding_window_length", columns="duration", values=metric, aggfunc="first")
        pivot = pivot.sort_index(ascending=True)
        data = pivot.values
        if np.isnan(data).all():
            continue
        flat_idx = int(np.nanargmax(data))
        val = data.flat[flat_idx]
        if val > best_val:
            best_val = val
            best_info = (ts, flat_idx // data.shape[1], flat_idx % data.shape[1])

    im = None
    for col_idx, ts in enumerate(trainride_starts):
        ax = axes[col_idx]
        ax.grid(False)

        sub = best[best["trainride_start_seconds"] == ts]
        pivot = sub.pivot_table(index="sliding_window_length", columns="duration", values=metric, aggfunc="first")
        pivot = pivot.sort_index(ascending=True)
        data = pivot.values

        im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(
            [str(x) for x in pivot.columns],
            rotation=45, ha="left", rotation_mode="anchor",
        )
        ax.set_yticks(np.arange(data.shape[0]))

        if col_idx == 0:
            ax.set_yticklabels([str(x) for x in pivot.index])
            ax.set_ylabel("Majority vote length")
        else:
            ax.set_yticklabels([])

        if col_idx == ncols // 2:
            ax.set_xlabel("Recording length (s)")

        ax.set_title(f"Trainride start: {int(ts)} s")

        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                m = data[ii, jj]
                if not np.isnan(m):
                    ax.text(jj, ii, f"{m:.2f}", ha="center", va="center", color="white")

        if best_info and best_info[0] == ts:
            draw_box(ax, best_info[1], best_info[2], color=CB_color_cycle[1], lw=1.5)

    fig.suptitle("TMD - trainride start offset", y=1.01)
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))

    if im is not None:
        cax = fig.add_axes([0.92, 0.05, 0.02, 0.90])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(display_metric)

    out_path = out_dir / f"tmd_{metric}.{output_format}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved TMD heatmap to {out_path}")


if __name__ == "__main__":
    app()
