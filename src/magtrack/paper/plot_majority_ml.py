"""CLI tool to plot majority vote metrics."""

import os

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import typer

from magtrack.utils.plot import configure_output_format, CB_color_cycle, PARAMS, get_column_width

# Initialize Typer app
app = typer.Typer(help="CLI tool to plot majority vote metrics.")


@app.command()
def plot_master_metrics(
        metrics_csv: str = typer.Option("results/coloc_ml/majority_vote/master_metrics_by_k.csv",
                                        help="Path to master metrics CSV"),
        chunk_size: int = typer.Option(60, help="Chunk size as int (e.g., 5, 10, 20, 30, 60)"),
        window_size: int = typer.Option(1, help="Window size (e.g., 1, 5, 10, 50, 100, 150)"),
        hz: int = typer.Option(10, help="Hz sampling rate (e.g., 10, 20, 40, 60)"),
        output_format: str = typer.Option("pdf", help="Output format for plots (pdf or pgf)"),
        out_dir: Path = typer.Option(Path("results/paper/figures"), help="Output directory for the figures"),
):
    """
    Reads the master metrics CSV and plots results for the specified configuration.
    X-axis: Recording Duration. Y-axis: Mean MCC.
    Lines: One for each majority_vote_length (k).
    """
    output_format = output_format.lower()
    if output_format not in ["pdf", "pgf"]:
        typer.echo("Invalid output format. Choose 'pdf' or 'pgf'.", err=True)
        raise typer.Exit(code=1)

    configure_output_format(output_format)

    df = pd.read_csv(metrics_csv)

    # Filter by exact configurations
    df = df[
        (df['chunk_size'] == chunk_size) &
        (df['window_size'] == window_size) &
        (df['hz'] == hz)
        ].copy()

    if df.empty:
        print(f"[{output_format.upper()}] No valid data found for Chunk={chunk_size}s, Window={window_size}, Hz={hz}")
        return

    # Replace 0 with 'full' and ensure string type for categorical plotting
    df['duration'] = df['duration'].replace({0: 'full'}).astype(str)

    unique_durations = df['duration'].unique()
    numeric_durations = sorted([int(d) for d in unique_durations if d != 'full'])

    duration_order = [str(d) for d in numeric_durations]
    has_full = 'full' in unique_durations

    if has_full:
        duration_order.append('full')

    k_values = sorted(df['k'].unique())

    fig_w = get_column_width()
    fig_h = fig_w * 0.7

    if has_full:
        n_num = len(numeric_durations)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharey=True, figsize=(fig_w, fig_h),
            gridspec_kw={'width_ratios': [n_num, 1]}
        )
    else:
        fig, ax1 = plt.subplots(figsize=(fig_w, fig_h))
        ax2 = None

    colors = CB_color_cycle[:len(k_values)]

    # Plot one line per k-value using explicit integer x-coordinates
    for j, k in enumerate(k_values):
        k_df = df[df['k'] == k]
        mean_mcc = k_df.groupby('duration')['mcc'].mean()

        x_vals = []
        y_vals = []
        for i, d_str in enumerate(duration_order):
            if d_str in mean_mcc and pd.notna(mean_mcc[d_str]):
                x_vals.append(i)
                y_vals.append(mean_mcc[d_str])

        if not x_vals:
            continue

        if has_full:
            # Plot on both axes to allow lines to cross the visual break seamlessly
            ax1.plot(
                x_vals, y_vals,
                marker='x', linestyle='-', color=colors[j % len(colors)], alpha=0.8
            )
            # Add labels only to the second axis to avoid legend duplication
            ax2.plot(
                x_vals, y_vals,
                marker='x', linestyle='-', color=colors[j % len(colors)], alpha=0.8,
                label=f'${int(k)}$'
            )
        else:
            ax1.plot(
                x_vals, y_vals,
                marker='x', linestyle='-', color=colors[j % len(colors)], alpha=0.8,
                label=f'${int(k)}$'
            )

    ax1.set_ylabel('MCC')
    ax1.set_ylim(0., 1.)
    ax1.grid(True, linestyle='--', alpha=0.6, color='gray')

    if has_full:
        ax2.grid(True, linestyle='--', alpha=0.6, color='gray')

        # zoom-in / limit the view to different portions of the data
        ax1.set_xlim(-0.5, n_num - 0.5)  # Most of the data
        ax2.set_xlim(n_num - 0.5, n_num + 0.5)  # The 'full' outlier

        # Apply custom categorical ticks
        ax1.set_xticks(range(n_num))
        ax1.set_xticklabels(duration_order[:n_num])
        ax2.set_xticks([n_num])
        ax2.set_xticklabels(['full'])

        # hide the spines between ax1 and ax2
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        ax1.yaxis.tick_left()
        ax2.tick_params(left=False, right=False)  # don't put tick labels on the break boundary

        # Now, let's turn towards the cut-out slanted lines (adapted for X-axis break)
        d = 0.5  # proportion of horizontal to vertical extent of the slanted line
        kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)

        # Draw on the right edge of ax1 and left edge of ax2
        ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

        # Set x label but ensure it is centered across both subplots
        fig.text(0.45, 0.02, 'Recording duration from train ride start (s)', ha='center',
                 fontsize=PARAMS['axes.labelsize'])
        ax2.legend(title='Vote', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    else:
        ax1.set_xticks(range(len(duration_order)))
        ax1.set_xticklabels(duration_order)
        ax1.set_xlim(-0.5, len(duration_order) - 0.5)
        ax1.set_xlabel('Recording duration from train ride start (s)')
        ax1.legend(title='Vote', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    # 1. Let Matplotlib do the automatic label spacing
    plt.tight_layout()

    if has_full:
        fig.subplots_adjust(wspace=0.05)

    # Ensure the output directory exists before saving
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = out_dir / f"mcc_chunk{chunk_size}s_win{window_size}_{hz}Hz.{output_format}"

    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {out_filename}")
    plt.close()


if __name__ == "__main__":
    app()
