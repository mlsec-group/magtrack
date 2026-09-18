from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib import patches

from magtrack.utils.plot import configure_output_format, PARAMS, CB_color_cycle, get_page_width, get_column_width

app = typer.Typer(help="Generate heatmaps for multiple metrics from one evaluation CSV.")

DEFAULT_METRICS = [
    "accuracy", "f1", "mcc", "precision",
    "recall"
]


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
        return "full"
    return f"{duration} s"


def duration_sort_key(duration: int | str) -> tuple[int, float]:
    if isinstance(duration, str):
        if duration == "all":
            return (1, float("inf"))
        try:
            value = int(duration)
        except ValueError:
            return (1, float("inf"))
    else:
        value = int(duration)

    if value == 0:
        return (1, float("inf"))
    return (0, float(value))


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


def add_aligned_colorbar(fig, axes_list, im, label, label_size=None, tick_label_size=None):
    if im is None:
        return

    visible_axes = [ax for ax in axes_list if ax.get_visible()]
    if not visible_axes:
        return

    bboxes = [ax.get_position() for ax in visible_axes]
    right = max(bb.x1 for bb in bboxes)
    bottom = min(bb.y0 for bb in bboxes)
    top = max(bb.y1 for bb in bboxes)

    pad = 0.01
    width = 0.02
    cax_left = min(0.98 - width, right + pad)
    cax = fig.add_axes([cax_left, bottom, width, top - bottom])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    if label_size is None:
        label_size = matplotlib.rcParams.get("axes.labelsize", matplotlib.rcParams.get("font.size", 10))
    if tick_label_size is None:
        tick_label_size = matplotlib.rcParams.get("ytick.labelsize", matplotlib.rcParams.get("font.size", 10))
    cbar.set_label(label, fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)


def plot_metric_grid(df, metric, chunks, durations, args, out_path):
    """Standard exploratory plotting function (single plot or N x M grid)."""
    pivot_store = prepare_pivot_data(df, metric, chunks, durations)
    best_cell = find_best_cell(pivot_store)
    display_metric = format_metric_name(metric)

    # fig_w = get_page_width()
    # fig_h = get_page_height()

    valid_entries = [v for v in pivot_store.values() if v is not None]
    if not valid_entries:
        print(f"Skipping {display_metric} (Grid): no valid pivot data found")
        return

    valid_pairs = [
        (ch, dur)
        for dur in durations
        for ch in chunks
        if (pivot_store.get((ch, dur)) is not None and not pivot_store[(ch, dur)][0].isna().all().all())
    ]

    single_mode = (len(valid_pairs) == 1)

    if single_mode:
        ch, dur = valid_pairs[0]
        pivot_mean, pivot_std = pivot_store[(ch, dur)]
        n_rows, n_cols = pivot_mean.shape
        fig_w, fig_h = max(6, n_cols * 0.8), max(6, n_rows * 0.4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        axes = np.array([[ax]])
    else:
        n_plots = len(valid_pairs)
        nrows, ncols = choose_compact_grid(n_plots, args.max_cols)

        fig_w = get_page_width()
        fig_h = fig_w * 1.1 if nrows > 2 else fig_w * 0.6  # Adjust height dynamically based on rows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = axes.reshape(nrows, 1)

    im = None
    right_label = "MCC" if metric.lower() == "mcc" else f"{display_metric}"
    xtick_size = matplotlib.rcParams.get("xtick.labelsize")
    ytick_size = matplotlib.rcParams.get("ytick.labelsize")
    label_size = matplotlib.rcParams.get("axes.labelsize")
    title_size = matplotlib.rcParams.get("axes.titlesize")
    annot_size = matplotlib.rcParams.get("axes.labelsize")

    for plot_idx, (ch, dur) in enumerate(valid_pairs):
        row, col = plot_idx // axes.shape[1], plot_idx % axes.shape[1]

        ax = axes[row, col]
        entry = pivot_store[(ch, dur)]
        pivot_mean, pivot_std = entry
        data = pivot_mean.values
        im = ax.imshow(data, aspect="auto", origin="lower", cmap='YlGn', vmin=0.0, vmax=1.0)

        # Set major ticks for labels (centers of the cells)
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))

        # Set minor ticks for the grid (edges of the cells)
        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)

        # Draw a solid grid on the minor ticks, disable on major ticks
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.grid(which="major", visible=False)

        # Only print X tick numbers on the bottom row
        if row == axes.shape[0] - 1 or single_mode:
            ax.set_xticklabels(
                [str(x) for x in pivot_mean.columns],
                rotation=0,
                ha="center",
                fontsize=xtick_size,
            )
        else:
            ax.set_xticklabels([])

        # Only print Y tick numbers on the left-most column
        if col == 0 or single_mode:
            ax.set_yticklabels([str(x) for x in pivot_mean.index], fontsize=ytick_size)
        else:
            ax.set_yticklabels([])

        # Set a more compact title and reduce the font size
        duration_label = format_duration_label(dur)
        if not single_mode:
            title = f"Ch={ch}s, Dur={duration_label}"
        else:
            title = f"{display_metric}\n(Ch={ch}s, Dur={duration_label})"
        ax.set_title(title, fontsize="small", pad=4)  # 'small' relative to base font size

        # Conditionally display small annotations
        if args.show_scores:
            for ii in range(data.shape[0]):
                for jj in range(data.shape[1]):
                    m = pivot_mean.iat[ii, jj]
                    s = pivot_std.iat[ii, jj]
                    if not np.isnan(m):
                        text_color = "white" if m > 0.60 else "black"
                        txt = f"{m:.2f}"
                        ax.text(jj, ii, txt, ha="center", va="center", color=text_color, fontsize=annot_size)

        # Highlighting boxes
        if best_cell and best_cell[:2] == (ch, dur):
            draw_box(ax, best_cell[2], best_cell[3], color=CB_color_cycle[7])

    # Hide unused axes
    for idx in range(len(valid_pairs), axes.size):
        axes[idx // axes.shape[1], idx % axes.shape[1]].set_visible(False)

    # Set singular centralized labels for the entire figure grid
    if not single_mode:
        fig.supxlabel("Data points per second (Hz)", fontsize=label_size, y=0.02)
        fig.supylabel("Rolling window size", fontsize=label_size, x=0.02)
    else:
        axes[0, 0].set_xlabel("Data points per second (Hz)", fontsize=label_size)
        axes[0, 0].set_ylabel("Rolling window size", fontsize=label_size)

    # Adjust layout to make room for centralized labels and the colorbar
    fig.tight_layout(rect=(0.02, 0.02, 0.92, 1))
    add_aligned_colorbar(fig, list(axes.flat), im, right_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {display_metric} normal grid to {out_path}")


def plot_metric_paper(df, metric, chunk, durations, args, out_path):
    """Paper plotting function (1 Row x N Columns). Formats cleanly for LaTeX widths."""
    pivot_store = prepare_pivot_data(df, metric, [chunk], durations)
    best_cell = find_best_cell(pivot_store)
    display_metric = format_metric_name(metric)

    xtick_size = matplotlib.rcParams.get("xtick.labelsize")
    ytick_size = matplotlib.rcParams.get("ytick.labelsize")
    label_size = matplotlib.rcParams.get("axes.labelsize")
    title_size = matplotlib.rcParams.get("axes.titlesize")
    annot_size = matplotlib.rcParams.get("axes.labelsize")

    valid_entries = [v for v in pivot_store.values() if v is not None]
    if not valid_entries:
        print(f"Skipping {display_metric} (Paper): no valid pivot data found")
        return

    valid_durations = [
        dur for dur in durations
        if (pivot_store.get((chunk, dur)) is not None and not pivot_store[(chunk, dur)][0].isna().all().all())
    ]

    if not valid_durations:
        print(f"Skipping {display_metric} (Paper): no valid durations for chunk={chunk}")
        return

    ncols = len(valid_durations)
    fig_w = get_page_width()
    fig_h = fig_w * 0.45

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(fig_w, fig_h))
    if ncols == 1:
        axes = [axes]

    im = None
    right_label = "MCC" if metric.lower() == "mcc" else f"{display_metric}"
    for col, dur in enumerate(valid_durations):
        ax = axes[col]

        # Explicitly disable the global dotted grid lines for the heatmap
        ax.grid(False)

        entry = pivot_store.get((chunk, dur))

        pivot_mean, pivot_std = entry
        data = pivot_mean.values
        im = ax.imshow(data, aspect="auto", origin="lower", cmap='YlGn', vmin=0.0, vmax=1.0)

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(
            [str(x) for x in pivot_mean.columns],
            rotation=0,
            ha="center",
            fontsize=xtick_size,
        )
        ax.set_yticks(np.arange(data.shape[0]))

        # Only show Y-labels on the first column to prevent clutter
        if col == 0:
            ax.set_yticklabels([str(x) for x in pivot_mean.index], fontsize=ytick_size)
            ax.set_ylabel("Rolling window size", fontsize=label_size)
        else:
            ax.set_yticklabels([])

        if col == ncols // 2:
            ax.set_xlabel("Data points per second (Hz)", fontsize=label_size)

        ax.set_title(format_duration_label(dur), fontsize=title_size)

        # Dense, small annotations formatted to 2 decimals to fit
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                m = pivot_mean.iat[ii, jj]
                s = pivot_std.iat[ii, jj]
                if not np.isnan(m):
                    text_color = "white" if m > 0.60 else "black"
                    txt = f"{m:.2f}"
                    ax.text(jj, ii, txt, ha="center", va="center", color=text_color, fontsize=annot_size)

        # Highlight boxes
        # if best_cell and best_cell[:2] == (chunk, dur):
        #     draw_box(ax, best_cell[2], best_cell[3], color=CB_color_cycle[7], lw=0.5)

        # if chunk == args.train_chunk and dur == args.train_duration:
        #     tw, th = args.train_window, args.train_hz
        #     if (tw in pivot_mean.index) and (th in pivot_mean.columns):
        #         draw_box(ax, list(pivot_mean.index).index(tw), list(pivot_mean.columns).index(th), color="blue", lw=1.5)

    fig.suptitle(
        "Total recording length",
        y=0.91,
        fontsize=matplotlib.rcParams.get("font.size", 10),
    )
    fig.tight_layout(rect=(0, 0, 0.95, 0.96))
    fig.subplots_adjust(wspace=0.1)
    add_aligned_colorbar(fig, list(axes), im, right_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {display_metric} paper plot to {out_path}")


def plot_metric_paper_single(df, metric, chunk, durations, args, out_path):
    """Paper plotting function (single heatmap across duration; fixed 10 Hz)."""
    display_metric = format_metric_name(metric)
    duration_values = [d for d in durations if d != "all"]
    if not duration_values:
        print(f"Skipping {display_metric} (Paper Single): no duration values found")
        return

    sub = df[(df["chunk"] == chunk) & (df["hz"] == 10) & (df["duration"].isin(duration_values))]
    if sub.empty:
        print(f"Skipping {display_metric} (Paper Single): no data for chunk={chunk} s at 10 Hz")
        return

    agg = sub.groupby(["window", "duration"])[metric].agg(["mean", "std"]).reset_index()
    pivot_mean = agg.pivot(index="window", columns="duration", values="mean").sort_index(ascending=True)
    pivot_std = agg.pivot(index="window", columns="duration", values="std").reindex_like(pivot_mean)
    sorted_cols = sorted(pivot_mean.columns, key=duration_sort_key)
    pivot_mean = pivot_mean.reindex(columns=sorted_cols)
    pivot_std = pivot_std.reindex(columns=sorted_cols)
    if pivot_mean.isna().all().all():
        print(f"Skipping {display_metric} (Paper Single): pivot data empty")
        return

    xtick_size = matplotlib.rcParams.get("xtick.labelsize")
    ytick_size = matplotlib.rcParams.get("ytick.labelsize")
    label_size = matplotlib.rcParams.get("axes.labelsize")
    title_size = matplotlib.rcParams.get("axes.titlesize")
    annot_size = matplotlib.rcParams.get("axes.labelsize")

    fig_w = get_column_width() * 0.7
    fig_h = fig_w * 0.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.grid(False)

    data = pivot_mean.values
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="YlGn", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(
        [format_duration_label(x) for x in pivot_mean.columns],
        rotation=0,
        ha="center",
        fontsize=xtick_size,
    )
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([str(x) for x in pivot_mean.index], fontsize=ytick_size)
    ax.set_xlabel("Recording duration from train ride start (s)", fontsize=label_size)
    ax.set_ylabel("Rolling window size", fontsize=label_size)
    ax.set_title("Data points per second: 10 Hz", fontsize=xtick_size)

    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            m = pivot_mean.iat[ii, jj]
            s = pivot_std.iat[ii, jj]
            if not np.isnan(m):
                text_color = "white" if m > 0.60 else "black"
                txt = f"{m:.2f}"
                ax.text(jj, ii, txt, ha="center", va="center", color=text_color, fontsize=annot_size)

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    right_label = "MCC" if metric.lower() == "mcc" else f"{display_metric}"
    add_aligned_colorbar(fig, [ax], im, right_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {display_metric} paper single heatmap to {out_path}")


@app.command()
def main(
        csv: Path = typer.Argument(..., help="Path to results .csv"),
        metrics: str = typer.Option(",".join(DEFAULT_METRICS), help="Comma-separated metrics to plot"),
        out_dir: Path = typer.Option(Path("results/paper/figures"), help="Output directory for the figures"),
        chunks_raw: str | None = typer.Option(None, "--chunks", help="Comma-separated chunk sizes (e.g. 30,60)"),
        durations_raw: str | None = typer.Option(None, "--durations",
                                                 help="Comma-separated durations (e.g. 300,900) or 'all'"),
        train_window: int = typer.Option(5, help="Training window_size_seconds"),
        train_hz: int = typer.Option(40, help="Training hz"),
        train_chunk: int = typer.Option(20, help="Training chunk_size_seconds"),
        train_duration: int = typer.Option(0, help="Training first<duration> value in filename"),
        max_cols: int = typer.Option(4, help="Maximum number of columns in grid mode"),
        output_format: str = typer.Option("pgf", help="Output file format for heatmaps: pgf|pdf"),
        show_scores: bool = typer.Option(False, help="Print numeric scores inside the heatmap cells"),
        grid_heatmaps: bool = typer.Option(
            True, "--grid-heatmaps/--no-grid-heatmaps",
            help="Write the full grid heatmap per metric ('<metric>_<csv>').",
        ),
        paper_heatmaps: bool = typer.Option(
            True, "--paper-heatmaps/--no-paper-heatmaps",
            help="Write the horizontal paper heatmap per metric ('<metric>_paper_<csv>').",
        ),
        paper_single_heatmaps: bool = typer.Option(
            True, "--paper-single-heatmaps/--no-paper-single-heatmaps",
            help="Write the single-row paper heatmap per metric ('<metric>_paper_single_<csv>').",
        ),
) -> None:
    output_format = output_format.lower()
    if output_format not in {"pgf", "pdf"}:
        raise typer.BadParameter("--output-format must be one of: pgf, pdf")

    args = SimpleNamespace(
        train_window=train_window,
        train_hz=train_hz,
        train_chunk=train_chunk,
        train_duration=train_duration,
        max_cols=max_cols,
        show_scores=show_scores,
    )

    print("Show scores inside heatmap cells:", args.show_scores)

    configure_matplotlib(output_format)

    # Read CSV and handle duplicate header rows (common when CSVs are concatenated)
    # The duplicate headers have "seed" as the first value in the dataset_file column
    df = pd.read_csv(csv)

    # Filter out rows where dataset_file looks like a header row
    if "dataset_file" in df.columns:
        # Remove rows where dataset_file contains the header text or is "seed"
        df = df[~df["dataset_file"].astype(str).str.contains("dataset_file|seed", na=False)]

    # Convert metric columns to numeric, coercing errors to NaN
    metric_cols = [col for col in df.columns if col not in ["seed", "dataset_file"]]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    meta = df["dataset_file"].apply(parse_meta).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)

    has_duration = df["duration"].notna().any()
    if not has_duration:
        df["duration"] = "all"

    # Setup durations for standard plots
    if durations_raw:
        raw_durations = [x.strip() for x in durations_raw.split(",") if x.strip()]
        if has_duration:
            durations = [int(x) for x in raw_durations if x != "all"]
            if "all" in raw_durations:
                durations.append("all")
        else:
            durations = ["all"]
    else:
        if has_duration:
            durations = sorted(
                df["duration"].dropna().unique().astype(int).tolist(),
                key=duration_sort_key,
            )
        else:
            durations = ["all"]

    durations = sorted(durations, key=duration_sort_key)

    # Setup chunks for standard plots
    if chunks_raw:
        chunks = [int(x.strip()) for x in chunks_raw.split(",") if x.strip()]
    else:
        chunks = sorted(df["chunk"].dropna().unique().astype(int).tolist())

    if not chunks or not durations:
        raise SystemExit("No chunks or durations found to plot.")

    # Setup durations for the paper plot specifically
    paper_available_chunks = df["chunk"].dropna().unique().astype(int).tolist()

    paper_available_durations = sorted(
        df["duration"].dropna().unique().tolist(),
        key=duration_sort_key,
    )

    metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    csv_name = csv.stem

    for metric in metrics:
        if metric not in df.columns:
            display_name = format_metric_name(metric)
            print(f"Skipping {display_name}: column '{metric}' not found")
            continue

        # 1. Normal/Grid Plot
        if grid_heatmaps:
            out_path = out_dir / f"{metric}_{csv_name}.{output_format}"
            plot_metric_grid(df, metric, chunks, durations, args, out_path)

        # 2. Paper Plot (Fixed chunk size 30s, horizontal layout)
        paper_chunk = 60
        if paper_chunk in paper_available_chunks:
            if paper_heatmaps:
                paper_out_path = out_dir / f"{metric}_paper_{csv_name}.{output_format}"
                plot_metric_paper(df, metric, paper_chunk, paper_available_durations, args, paper_out_path)

            if paper_single_heatmaps:
                paper_single_out_path = out_dir / f"{metric}_paper_single_{csv_name}.{output_format}"
                plot_metric_paper_single(
                    df,
                    metric,
                    paper_chunk,
                    paper_available_durations,
                    args,
                    paper_single_out_path,
                )
        else:
            display_name = format_metric_name(metric)
            print(f"Skipping paper plot for {display_name}: chunk={paper_chunk} s not present in data")


if __name__ == "__main__":
    app()
