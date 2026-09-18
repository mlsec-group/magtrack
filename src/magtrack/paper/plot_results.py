from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import matplotlib
import pandas as pd
import matplotlib.lines
import matplotlib.pyplot as plt
import typer

from magtrack.utils.plot import (
    CB_color_cycle, configure_output_backend, get_presentation_size, tex_text,
)
from magtrack.utils.results import load_and_prepare_coloc_results, load_best_tmd_results

app = typer.Typer(help="Plot evaluation results.", invoke_without_command=False)


# ---------------------------------------------------------------------------
# Subcommand: coloc-distance
# ---------------------------------------------------------------------------

@app.command("coloc-distance")
def plot_coloc_distance(
        csv_path: Path = typer.Argument(..., help="Path to results CSV (e.g. data/results/dtw.csv)."),
        output: Path = typer.Argument(...,
                                      help="Output path; the extension picks the format (.pgf needs LaTeX, .png/.pdf do not)."),
        score: str = typer.Option("mcc", help="Score column to plot (without _test suffix), e.g. 'mcc', 'mrr', 'f1'."),
        title: str = typer.Option("", help="Optional plot title."),
        stdev: bool = typer.Option(True, help="Plot std deviation."),
        ylim: tuple[float, float] | None = typer.Option(None, help="Y-axis limits as (min, max)."),
) -> None:
    """Plot a distance-evaluation score vs. trainride duration, one line per sampling rate."""
    configure_output_backend(output)
    df = load_and_prepare_coloc_results(csv_path, score)
    score_col = f"{score}_test"
    std_col = f"{score}_test_std"
    has_std = std_col in df.columns

    fig_w, fig_h = get_presentation_size()
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sampling_rates = sorted(df["sampling_rate"].unique())
    for i, sr in enumerate(sampling_rates):
        subset = df[df["sampling_rate"] == sr].sort_values("trainride_start_seconds")
        color = CB_color_cycle[i % len(CB_color_cycle)]
        label = tex_text(f"{int(sr)}\\,Hz")
        ax.plot(subset["trainride_start_seconds"], subset[score_col], marker="x", markersize=2,
                color=color, label=label)
        if stdev and has_std:
            std = subset[std_col]
            if std.notna().any():
                ax.fill_between(subset["trainride_start_seconds"],
                                subset[score_col] - std,
                                subset[score_col] + std,
                                alpha=0.15, color=color)

    ax.set_xlabel("Trainride duration (s)")
    ax.set_ylabel(score.upper())
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlim(left=0, right=1000)
    if title:
        ax.set_title(title)
    ax.legend(ncol=2)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output))
    plt.close(fig)
    typer.echo(f"Saved plot to {output}")


# ---------------------------------------------------------------------------
# Subcommand: Train ride detection
# ---------------------------------------------------------------------------

@app.command("tmd-results")
def plot_tmd_results(
        input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                         help="Directory containing tmd_s*_d*.optuna_search_*.csv files."),
        output: Path = typer.Argument(...,
                                      help="Output path; the extension picks the format (.pgf needs LaTeX, .png/.pdf do not)."),
        metric: str = typer.Option("mcc_test", help="Metric column used to select the best row per group."),
        title: str = typer.Option("", help="Optional plot title."),
        ylim: tuple[float, float] = typer.Option((0.5, 1.0), help="Y-axis limits as (min, max)."),
        xlim: tuple[float, float] | None = typer.Option(None, help="X-axis limits as (min, max)."),
) -> None:
    configure_output_backend(output)
    files = sorted(input_dir.glob("tmd_s*_d*.optuna_search_*.csv"))
    if not files:
        typer.echo(f"No optuna-search CSV files found in {input_dir}", err=True)
        raise typer.Exit(code=1)

    best = load_best_tmd_results(files)
    best = best.copy()
    best["recording_time"] = best["duration"] * best["sliding_window_length"]

    trainride_starts = sorted(best["trainride_start_seconds"].unique())
    if 0 in trainride_starts:
        trainride_starts = [ts for ts in trainride_starts if ts != 0] + [0]
    windows = sorted(best["sliding_window_length"].unique())

    _MARKERS = ["x", "+", "o", "*", "^"]
    marker_map = {ts: _MARKERS[i % len(_MARKERS)] for i, ts in enumerate(trainride_starts)}
    ts_label = {ts: ("all" if ts == 0 else tex_text(f"{ts}\\,s")) for ts in trainride_starts}
    color_map = {w: CB_color_cycle[i % len(CB_color_cycle)] for i, w in enumerate(windows)}

    fig_w, fig_h = get_presentation_size()
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for ts in trainride_starts:
        for w in windows:
            subset = best[(best["trainride_start_seconds"] == ts) & (best["sliding_window_length"] == w)]
            if subset.empty:
                continue
            ax.scatter(
                subset["recording_time"], subset[metric],
                marker=marker_map[ts],
                color=color_map[w],
                s=20,
            )

    ax.set_xlabel("Used recording time (s)")
    ax.set_ylabel("MCC")
    ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)

    ts_handles = [
        matplotlib.lines.Line2D([], [], marker=marker_map[ts], color="grey",
                                linestyle="None", markersize=4, label=ts_label[ts])
        for ts in trainride_starts
    ]
    win_handles = [
        matplotlib.lines.Line2D([], [], marker="s", color=color_map[w],
                                linestyle="None", markersize=4, label=f"{w}")
        for w in windows
    ]

    legend_w = 0.2  # axes-fraction width
    leg1 = ax.legend(
        handles=ts_handles, title="First",
        fontsize="x-small", title_fontsize="x-small",
        loc="upper left", bbox_to_anchor=(1.02, 1.0, legend_w, 0.0),
        mode="expand", borderaxespad=0.0, handletextpad=0.5,
    )
    ax.add_artist(leg1)
    l2_y_position = 0.42

    ax.legend(
        handles=win_handles, title="Vote",
        fontsize="x-small", title_fontsize="x-small",
        loc="upper left", bbox_to_anchor=(1.02, l2_y_position, legend_w, 0.0),
        mode="expand", borderaxespad=0.0, handletextpad=0.5,
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Saved plot to {output}")


# ---------------------------------------------------------------------------
# Subcommand: runtime-comparison
# ---------------------------------------------------------------------------

_APPROACH_STYLE = {
    "dtw": {"label": "DTW", "linestyle": "--", "marker": "x", "alpha": 0.9},
    "ml_majority": {"label": "ML", "linestyle": ":", "marker": "+", "alpha": 0.5},
}


@app.command("runtime-comparison")
def plot_runtime_comparison(
        csv_path: Path = typer.Argument(..., help="CSV written by benchmark-inference."),
        output: Path = typer.Argument(...,
                                      help="Output path; the extension picks the format (.pgf needs LaTeX, .png/.pdf do not)."),
        include_vote: bool = typer.Option(
            True, help="Count the majority vote in the ML time; --no-include-vote times inference only."),
        stdev: bool = typer.Option(False, help="Draw std deviation as error bars."),
        title: str = typer.Option("", help="Optional plot title."),
        xlim: tuple[float, float] | None = typer.Option(None, help="X-axis limits as (min, max)."),
        ylim: tuple[float, float] | None = typer.Option(None, help="Y-axis limits as (min, max)."),
        log_x: bool = typer.Option(True,
                                   help="Log-scale the x axis; useful when the approaches differ by orders of magnitude."),
        log_y: bool = typer.Option(True,
                                   help="Log-scale the y axis; useful when the approaches differ by orders of magnitude."),
) -> None:
    """Plot per-pair computation time vs recording length, one colour per sampling rate."""
    configure_output_backend(output)

    df = pd.read_csv(csv_path)
    value_col = "seconds" if include_vote else "inference_seconds"
    for column in ("approach", "recording_length", "sampling_rate", value_col):
        if column not in df.columns:
            typer.echo(f"Column '{column}' not found in {csv_path}", err=True)
            raise typer.Exit(code=1)

    approaches = [a for a in _APPROACH_STYLE if a in set(df["approach"])]
    if not approaches:
        typer.echo(f"No known approaches in {csv_path}; expected any of "
                   f"{sorted(_APPROACH_STYLE)}.", err=True)
        raise typer.Exit(code=1)

    common_lengths = set.intersection(*[
        set(df.loc[df["approach"] == a, "recording_length"]) for a in approaches])
    common_rates = set.intersection(*[
        set(df.loc[df["approach"] == a, "sampling_rate"]) for a in approaches])
    if not common_lengths or not common_rates:
        typer.echo("No (recording_length, sampling_rate) cell is present for every "
                   "approach; nothing comparable to plot.", err=True)
        raise typer.Exit(code=1)

    df = df[df["recording_length"].isin(common_lengths) & df["sampling_rate"].isin(common_rates)]
    grouped = (df.groupby(["approach", "recording_length", "sampling_rate"])[value_col]
               .agg(["mean", "std"]).reset_index())

    fig_w, fig_h = get_presentation_size()
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sampling_rates = sorted(common_rates)
    colors = {sr: CB_color_cycle[i % len(CB_color_cycle)] for i, sr in enumerate(sampling_rates)}

    for approach in approaches:
        style = _APPROACH_STYLE[approach]
        for sr in sampling_rates:
            subset = grouped[(grouped["approach"] == approach)
                             & (grouped["sampling_rate"] == sr)].sort_values("recording_length")
            if subset.empty:
                continue
            ax.errorbar(
                subset["recording_length"], subset["mean"],
                yerr=subset["std"] if stdev else None,
                color=colors[sr], linestyle=style["linestyle"], marker=style["marker"],
                markersize=3, alpha=style["alpha"], capsize=2 if stdev else 0,
            )

    ax.set_xlabel("Recording length (s)")
    ax.set_ylabel("Computation per pair (s)")
    ax.set_yscale("log" if log_y else "linear")
    ax.set_xscale("log" if log_x else "linear")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    legend_w = 0.25

    approach_handles = [
        matplotlib.lines.Line2D([], [], color="grey", linestyle=_APPROACH_STYLE[a]["linestyle"],
                                marker=_APPROACH_STYLE[a]["marker"], markersize=1,
                                label=_APPROACH_STYLE[a]["label"])
        for a in approaches
    ]
    leg1 = ax.legend(
        handles=approach_handles, title="Approach",
        fontsize="x-small", title_fontsize="x-small",
        loc="upper left", bbox_to_anchor=(1.02, 1.0, legend_w, 0.0),
        mode="expand", borderaxespad=0.0, handletextpad=0.5,
        handlelength=1.0, alignment="center",
    )
    ax.add_artist(leg1)

    sr_handles = [
        matplotlib.lines.Line2D([], [], color=colors[sr], linestyle="-",
                                marker="s", markersize=1, label=tex_text(f"{int(sr)}\\,Hz"))
        for sr in sampling_rates
    ]
    ax.legend(
        handles=sr_handles, title="Data Points",
        fontsize="x-small", title_fontsize="x-small",
        loc="upper left", bbox_to_anchor=(1.02, 0.55, legend_w, 0.0),
        mode="expand", borderaxespad=0.0, handletextpad=0.5,
        handlelength=1.0, alignment="center",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Saved plot to {output}")


if __name__ == "__main__":
    app()
