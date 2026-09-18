from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from magtrack.utils.results import load_best_tmd_results

app = typer.Typer(help="Generate publication tables from experiment outputs.")


@app.callback()
def main() -> None:
    """Top-level CLI callback."""
    return None


def _label(v: object) -> str:
    if pd.isna(v):
        return ""
    try:
        fv = float(v)  # type: ignore[arg-type]
    except Exception:
        return str(v)
    if fv.is_integer():
        return str(int(fv))
    return str(fv)


def _first_label(v: object) -> str:
    """Format trainride_start_seconds: 0 → 'all', else integer string."""
    if pd.isna(v):
        return ""
    try:
        fv = float(v)  # type: ignore[arg-type]
    except Exception:
        return str(v)
    if fv.is_integer() and int(fv) == 0:
        return "all"
    if fv.is_integer():
        return str(int(fv))
    return str(fv)


def _fmt_metric(val: object, std: object, digits: int) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "---"
    try:
        v_f = float(val)  # type: ignore[arg-type]
    except Exception:
        return str(val)
    fmt = f"{{:.{digits}f}}"
    v_str = fmt.format(v_f)
    try:
        s_f = float(std)  # type: ignore[arg-type]
        if not pd.isna(s_f):
            return rf"{v_str}$\pm${fmt.format(s_f)}"
    except Exception:
        pass
    return v_str


@app.command("tmd-results")
def tmd_results(
        input_dir: Path = typer.Argument(
            ..., exists=True, file_okay=False, dir_okay=True, readable=True,
            help="Directory containing tmd_s*_d*.optuna_search_*.csv files.",
        ),
        output_tex: Path = typer.Argument(..., help="Output LaTeX table file path."),
        best_metric: str = typer.Option("mcc_test", help="Metric used to select the best row per group."),
        label: str = typer.Option("tab:tmd_summary", help="LaTeX \\label for the table."),
        caption: str = typer.Option(
            "TMD results for the best hyperparameters. M: number of majority-vote chunks. D: chunk duration in seconds. Start: seconds from trainride start used for training (all: full ride).",
            help="LaTeX caption.",
        ),
        digits: int = typer.Option(2, help="Number of decimal digits for metric values."),
) -> None:
    """Generate the TMD summary .tex table.

    Rows: one per (trainride_start_seconds, duration) combination.
    Columns: for each sliding_window_length (M) a pair of F1 and MCC.
    Output uses table* for double-column layout.
    """
    files = sorted(input_dir.glob("tmd_s*_d*.optuna_search_*.csv"))
    if not files:
        typer.echo(f"No optuna-search CSV files found in {input_dir}", err=True)
        raise typer.Exit(code=1)

    best = load_best_tmd_results(files, best_metric=best_metric)

    m_values = sorted(best["sliding_window_length"].unique())
    first_values = sorted(best["trainride_start_seconds"].unique())
    durations = sorted(best["duration"].unique())

    # Build pivot tables
    piv_f1 = best.pivot_table(
        index=["trainride_start_seconds", "duration"],
        columns="sliding_window_length",
        values="f1_test",
        aggfunc="first",
    )
    piv_mcc = best.pivot_table(
        index=["trainride_start_seconds", "duration"],
        columns="sliding_window_length",
        values="mcc_test",
        aggfunc="first",
    )
    piv_f1_std = (
        best.pivot_table(
            index=["trainride_start_seconds", "duration"],
            columns="sliding_window_length",
            values="f1_test_std",
            aggfunc="first",
        )
        if "f1_test_std" in best.columns else None
    )
    piv_mcc_std = (
        best.pivot_table(
            index=["trainride_start_seconds", "duration"],
            columns="sliding_window_length",
            values="mcc_test_std",
            aggfunc="first",
        )
        if "mcc_test_std" in best.columns else None
    )

    n_m = len(m_values)
    # 2 row-key columns (Start, D) + 2 metric columns per M value
    col_fmt = "rr" + "cc" * n_m

    lines: list[str] = [
        r"\begin{table*}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]

    # First header row: row-key columns span both header rows (bottom-aligned), then \multicolumn per M
    m_cells = [rf"\multicolumn{{2}}{{c}}{{M\,=\,{_label(m)}}}" for m in m_values]
    row_key_cells = [
        r"\multirow[b]{2}{*}{\textbf{First (s)}}",
        r"\multirow[b]{2}{*}{\textbf{D}}",
    ]
    lines.append(" & ".join(row_key_cells) + " & " + " & ".join(m_cells) + r" \\")

    # Cmidrule under each M pair (offset by 2 row-key columns)
    cmidrules = [rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(n_m)]
    lines.append(" ".join(cmidrules))

    # Second header row: empty row-key placeholders, then F1/MCC per M
    metric_cells = ["", ""] + [cell for _ in m_values for cell in (r"\textbf{F1}", r"\textbf{MCC}")]
    lines.append(" & ".join(metric_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows — one per (first, D) combination
    prev_first = None
    for first_val in first_values:
        for dur in durations:
            if (first_val, dur) not in piv_f1.index:
                continue

            # Show First (s) label only on the first row of each group
            if first_val != prev_first:
                first_str = _first_label(first_val)
                prev_first = first_val
            else:
                first_str = ""

            cells = [first_str, _label(dur)]
            for m in m_values:
                f1_val = piv_f1.at[(first_val, dur), m] if m in piv_f1.columns else float("nan")
                mcc_val = piv_mcc.at[(first_val, dur), m] if m in piv_mcc.columns else float("nan")
                f1_std = piv_f1_std.at[
                    (first_val, dur), m] if piv_f1_std is not None and m in piv_f1_std.columns else None
                mcc_std = piv_mcc_std.at[
                    (first_val, dur), m] if piv_mcc_std is not None and m in piv_mcc_std.columns else None
                cells.append(_fmt_metric(f1_val, f1_std, digits))
                cells.append(_fmt_metric(mcc_val, mcc_std, digits))
            lines.append(" & ".join(cells) + r" \\")

        # Separator between first-value groups (except after the last)
        if first_val != first_values[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    typer.echo(f"Wrote LaTeX table to {output_tex}")


if __name__ == "__main__":
    app()
