from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger

from magtrack.utils.results import load_and_prepare_coloc_results, load_best_tmd_results

app = typer.Typer(help="Magtrack Paper latex vars")


# ---------------------------------------------------------------------------
# Subcommand: ml-results
# ---------------------------------------------------------------------------

def _parse_meta(name: str):
    """Extract configuration parameters from the dataset file name."""
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


def _safe_format_param(val) -> str:
    """Format configuration parameters, handling None/NaN gracefully."""
    if pd.isna(val) or val is None:
        return "All"
    if isinstance(val, float):
        return str(int(val))
    return str(val)


@app.command("ml-results")
def ml_results(
        results_file: Path = typer.Argument(..., help="Path to evaluation_results.result_ml.csv"),
        output_file: str = typer.Argument(..., help="Path to output .tex file"),
        round_digits: int = typer.Option(2, "-r", "--round", help="Number of decimal places to round score values to"),
) -> None:
    """Find the overall best F1 and MCC scores, plus specific chunk/duration stats."""
    if not results_file.is_file():
        logger.error(f"File not found: {results_file}")
        raise typer.Exit(code=1)

    try:
        df = pd.read_csv(results_file)

        if "dataset_file" in df.columns:
            df = df[~df["dataset_file"].astype(str).str.contains("dataset_file|seed", na=False)]

        metric_cols = [col for col in df.columns if col not in ["seed", "dataset_file"]]
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise typer.Exit(code=1)

    if "dataset_file" not in df.columns:
        logger.error("Column 'dataset_file' missing from CSV. Cannot parse configurations.")
        raise typer.Exit(code=1)

    meta = df["dataset_file"].apply(_parse_meta).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)

    tex_commands: list[str] = []
    digits_score = max(0, int(round_digits))

    digit_word = {
        30: "Thirty",
        60: "Sixty",
        300: "ThreeHundred",
        600: "SixHundred",
        900: "NineHundred"
    }

    for metric, metric_label in [("f1", "F"), ("mcc", "MCC")]:
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in CSV, skipping.")
            continue

        agg_df = df.groupby(["window", "chunk", "duration", "hz"], dropna=False)[metric].agg(
            ["mean", "std"]).reset_index()

        if agg_df.empty or agg_df["mean"].isna().all():
            logger.warning(f"No valid numeric data calculated for metric '{metric}', skipping.")
            continue

        # ---------------------------------------------------------
        # 1. Overall Best Configuration
        # ---------------------------------------------------------
        best_idx = agg_df["mean"].idxmax()
        best_row = agg_df.loc[best_idx]

        best_window = _safe_format_param(best_row["window"])
        best_chunk = _safe_format_param(best_row["chunk"])
        best_duration = _safe_format_param(best_row["duration"])
        best_hz = _safe_format_param(best_row["hz"])

        formatted_value = format(best_row["mean"], f".{digits_score}f")

        tex_commands.append(
            f"\\newcommand{{\\MlResultBest{metric_label}Overall}}{{\\numprint{{{formatted_value}}}\\xspace}}")
        tex_commands.append(f"\\newcommand{{\\MlResultBest{metric_label}Duration}}{{{best_duration}\\xspace}}")
        tex_commands.append(f"\\newcommand{{\\MlResultBest{metric_label}Chunk}}{{{best_chunk}\\xspace}}")
        tex_commands.append(f"\\newcommand{{\\MlResultBest{metric_label}Window}}{{{best_window}\\xspace}}")
        tex_commands.append(f"\\newcommand{{\\MlResultBest{metric_label}Hz}}{{{best_hz}\\xspace}}")

        # ---------------------------------------------------------
        # 2. Specific Stats: Chunk 30s + Durations [60, 300, 900]
        # ---------------------------------------------------------
        if pd.notna(best_row["chunk"]):
            target_chunk = int(best_row["chunk"])
        else:
            target_chunk = 60

        target_durations = [60, 300, 600, 900]

        for dur in target_durations:
            sub_df = agg_df[(agg_df["chunk"] == target_chunk) & (agg_df["duration"] == dur)]

            if sub_df.empty or sub_df["mean"].isna().all():
                continue

            best_sub_idx = sub_df["mean"].idxmax()
            best_sub_row = sub_df.loc[best_sub_idx]

            mean_val = best_sub_row["mean"]

            mean_fmt = format(mean_val, f".{digits_score}f")

            chunk_word = digit_word.get(target_chunk, f"C{target_chunk}")
            dur_word = digit_word.get(dur, f"D{dur}")
            tex_name = f"MlResultBest{metric_label}Chunk{chunk_word}Duration{dur_word}"

            tex_commands.append(
                f"\\newcommand{{\\{tex_name}}}{{\\numprint{{{mean_fmt}}}\\xspace}}"
            )

    for cmd in tex_commands:
        print(cmd)

    with open(output_file, "w") as tex_file:
        tex_file.write("\n".join(tex_commands) + "\n")

    logger.info(f"LaTeX variables written to {output_file} ({len(tex_commands)} commands)")


# ---------------------------------------------------------------------------
# Subcommand: coloc-distances
# ---------------------------------------------------------------------------

_DIGIT_WORD: dict[int, str] = {
    0: "All", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
    11: "Eleven", 12: "Twelve", 13: "Thirteen", 15: "Fifteen", 20: "Twenty",
    30: "Thirty", 40: "Fourty", 50: "Fifty", 60: "Sixty", 100: "Hundred",
    300: "ThreeHundred", 600: "SixHundred", 900: "NineHundred",
}

_ACRONYMS = {"dtw", "ddtw", "mcc", "mrr"}


def _auto_unit(seconds: float) -> tuple[float, str]:
    """Return (value, unit_string) scaled to the most readable unit."""
    if seconds >= 3600:
        return seconds / 3600, "h"
    if seconds >= 60:
        return seconds / 60, "min"
    if seconds >= 1:
        return seconds, "s"
    if seconds >= 1e-3:
        return seconds * 1e3, "ms"
    if seconds >= 1e-6:
        return seconds * 1e6, r"\textmu s"
    return seconds * 1e9, "ns"


def _filename_to_tex_name(stem: str) -> str:
    parts = stem.split("_")
    result = []
    for part in parts:
        if part in _ACRONYMS:
            result.append(part.upper())
        elif part.startswith("r") and part[1:].isdigit():
            n = int(part[1:])
            result.append("R" + _DIGIT_WORD.get(n, f"N{n}"))
        else:
            result.append(part.capitalize())
    return "".join(result)


@app.command("coloc-distances")
def coloc_distances(
        results_file: Path = typer.Argument(..., help="Directory containing results CSV files."),
        output_file: str = typer.Argument(..., help="Path to output .tex file"),
        round_digits: int = typer.Option(2, "-r", "--round", help="Number of decimal places to round score values to"),
        round_digits_time: int = typer.Option(0, "-t", "--round-time",
                                              help="Number of decimal places to round time values to"),
        stdev: bool = typer.Option(True, help="Include ±std in output commands."),
) -> None:
    """Generate LaTeX \\newcommand definitions for best F1 and MCC per distance and trainride duration."""
    if not results_file.is_file():
        logger.error(f"No CSV files found in {results_file}")
        raise typer.Exit(code=1)

    tex_commands: list[str] = []

    distance_name = _filename_to_tex_name(results_file.stem)
    for metric, metric_label in [("f1", "F"), ("mcc", "MCC")]:
        try:
            df = load_and_prepare_coloc_results(results_file, metric)
        except SystemExit:
            logger.warning(f"Skipping {results_file.name} for metric '{metric}' (column not found).")
            continue

        score_col = f"{metric}_test"
        std_col = f"{metric}_test_std"
        has_std = std_col in df.columns
        digits_score = max(0, int(round_digits))

        best_idx = df.groupby("trainride_start_seconds")[score_col].idxmax()
        best_rows = df.loc[best_idx].set_index("trainride_start_seconds")

        per_sr_rows = []
        if "sampling_rate" in df.columns:
            per_sr_idx = df.groupby(["trainride_start_seconds", "sampling_rate"])[score_col].idxmax()
            per_sr_df = df.loc[per_sr_idx]
            for _, r in per_sr_df.iterrows():
                per_sr_rows.append((int(r["trainride_start_seconds"]), int(r["sampling_rate"]), r))

        for seconds, row in best_rows.iterrows():
            duration_word = _DIGIT_WORD.get(int(seconds), f"S{int(seconds)}")
            tex_name = f"ColocDistance{distance_name}Best{metric_label}First{duration_word}"
            digits_score = max(0, int(round_digits))
            value_num = float(row[score_col])
            value = format(value_num, f".{digits_score}f")
            if stdev and has_std and not math.isnan(row[std_col]):
                std_num = float(row[std_col])
                std = format(std_num, f".{digits_score}f")
                tex_commands.append(
                    f"\\newcommand{{\\{tex_name}}}{{\\numprint{{{value}}}$\\pm$\\numprint{{{std}}}\\xspace}}"
                )
            else:
                tex_commands.append(
                    f"\\newcommand{{\\{tex_name}}}{{\\numprint{{{value}}}\\xspace}}"
                )

            if "total_cpu_hours" in row.index and "n_calculated_pairs" in row.index:
                cpu_s = float(row["total_cpu_hours"]) * 3600.0
                n_pairs = float(row["n_calculated_pairs"])
                n_points = int(seconds) * float(row["sampling_rate"])

                time_per_pair_s = cpu_s / n_pairs
                time_per_point_s = time_per_pair_s / n_points

                for suffix, raw_s in [("TimePerPair", time_per_pair_s), ("TimePerPoint", time_per_point_s)]:
                    val, unit = _auto_unit(raw_s)
                    digits = max(0, int(round_digits_time))
                    val_r = format(val, f".{digits}f")
                    tname = f"{tex_name}{suffix}"
                    tex_commands.append(
                        f"\\newcommand{{\\{tname}}}{{\\numprint{{{val_r}}}\\xspace}}"
                    )
                    tex_commands.append(
                        f"\\newcommand{{\\{tname}Unit}}{{{unit}\\xspace}}"
                    )

        for seconds, sr, row in per_sr_rows:
            duration_word = _DIGIT_WORD.get(int(seconds), f"S{int(seconds)}")
            sr_word = _DIGIT_WORD.get(int(sr), f"N{int(sr)}")
            tex_name = f"ColocDistance{distance_name}{metric_label}First{duration_word}SR{sr_word}"
            value_num = float(row[score_col])
            value = format(value_num, f".{digits_score}f")
            if stdev and has_std and not math.isnan(row[std_col]):
                std_num = float(row[std_col])
                std = format(std_num, f".{digits_score}f")
                tex_commands.append(
                    f"\\newcommand{{\\{tex_name}}}{{\\numprint{{{value}}}$\\pm$\\numprint{{{std}}}\\xspace}}"
                )
            else:
                tex_commands.append(
                    f"\\newcommand{{\\{tex_name}}}{{\\numprint{{{value}}}\\xspace}}"
                )

            if "total_cpu_hours" in row.index and "n_calculated_pairs" in row.index:
                cpu_s = float(row["total_cpu_hours"]) * 3600.0
                n_pairs = float(row["n_calculated_pairs"])
                n_points = int(seconds) * float(sr)

                time_per_pair_s = cpu_s / n_pairs
                time_per_point_s = time_per_pair_s / n_points

                for suffix, raw_s in [("TimePerPair", time_per_pair_s), ("TimePerPoint", time_per_point_s)]:
                    val, unit = _auto_unit(raw_s)
                    digits = max(0, int(round_digits_time))
                    val_r = format(val, f".{digits}f")
                    tname = f"{tex_name}{suffix}"
                    tex_commands.append(
                        f"\\newcommand{{\\{tname}}}{{\\numprint{{{val_r}}}\\xspace}}"
                    )
                    tex_commands.append(
                        f"\\newcommand{{\\{tname}Unit}}{{{unit}\\xspace}}"
                    )

    for cmd in tex_commands:
        print(cmd)

    with open(output_file, "w") as tex_file:
        tex_file.write("\n".join(tex_commands) + "\n")

    logger.info(f"LaTeX variables written to {output_file} ({len(tex_commands)} commands)")


# ---------------------------------------------------------------------------
# Subcommand: coloc-distances
# ---------------------------------------------------------------------------

@app.command("tmd-results")
def tmd_results(
        input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                         help="Directory containing tmd_s*_d*.optuna_search_*.csv files."),
        output_file: str = typer.Argument(..., help="Path to output .tex file"),
        best_metric: str = typer.Option("mcc_test", help="Metric column used to select the best row per group."),
        round_digits: int = typer.Option(2, "-r", "--round", help="Number of decimal places to round score values to"),
) -> None:
    files = sorted(input_dir.glob("tmd_s*_d*.optuna_search_*.csv"))
    if not files:
        typer.echo(f"No optuna-search CSV files found in {input_dir}", err=True)
        raise typer.Exit(code=1)

    best = load_best_tmd_results(files, best_metric=best_metric)

    _METRIC_LABEL: dict[str, str] = {"f1_test": "F", "f1": "F"}
    digits_score = max(0, int(round_digits))
    tex_commands: list[str] = []

    metrics = [
        (best_metric, _METRIC_LABEL.get(best_metric, best_metric.replace("_test", "").replace("_", "").capitalize()))]
    if best_metric != "f1_test":
        metrics.append(("f1_test", "F"))

    for _, row in best.iterrows():
        first_word = _DIGIT_WORD.get(int(row["trainride_start_seconds"]), f"S{int(row['trainride_start_seconds'])}")
        d_word = _DIGIT_WORD.get(int(row["duration"]), f"N{int(row['duration'])}")
        w_word = _DIGIT_WORD.get(int(row["sliding_window_length"]), f"N{int(row['sliding_window_length'])}")

        for col, metric_label in metrics:
            if col not in row.index:
                continue
            tex_name = f"TmdResultBest{metric_label}First{first_word}D{d_word}W{w_word}"
            value = format(float(row[col]), f".{digits_score}f")
            tex_commands.append(
                f"\\newcommand{{\\{tex_name}}}{{\\numprint{{{value}}}\\xspace}}"
            )

    # Best scores with sliding_window_length == 1
    w1 = best[best["sliding_window_length"] == 1]
    for col, metric_label in metrics:
        if col not in best.columns or w1.empty:
            continue
        best_w1_row = w1.loc[w1[col].idxmax()]
        first_word = _DIGIT_WORD.get(int(best_w1_row["trainride_start_seconds"]),
                                     f"S{int(best_w1_row['trainride_start_seconds'])}")
        d_word = _DIGIT_WORD.get(int(best_w1_row["duration"]), f"N{int(best_w1_row['duration'])}")
        tex_name = f"TmdResultBest{metric_label}OneChunk"
        value = format(float(best_w1_row[col]), f".{digits_score}f")
        tex_commands.append(
            f"\\newcommand{{\\{tex_name}}}{{\\numprint{{{value}}}\\xspace}}"
        )

    # Best scores overall
    for col, metric_label in metrics:
        if col not in best.columns:
            continue
        tex_name = f"TmdResultBest{metric_label}Overall"
        value = format(float(best[col].max()), f".{digits_score}f")
        tex_commands.append(
            f"\\newcommand{{\\{tex_name}}}{{\\numprint{{{value}}}\\xspace}}"
        )

    for cmd in tex_commands:
        print(cmd)

    with open(output_file, "w") as tex_file:
        tex_file.write("\n".join(tex_commands) + "\n")

    logger.info(f"LaTeX variables written to {output_file} ({len(tex_commands)} commands)")


@app.command("runtime")
def runtime_vars(
        benchmark_csv: Path = typer.Argument(..., help="CSV written by benchmark-inference."),
        output_file: str = typer.Argument(..., help="Path to output .tex file"),
        results_csv: Optional[Path] = typer.Option(
            None, help="Distance results CSV (e.g. results/coloc_distance/dtw_r1.csv). When given, "
                       "the configuration with the best MCC is reported alongside its runtime."),
        include_vote: bool = typer.Option(
            True, help="Count the majority vote in the ML runtime."),
        round_digits: int = typer.Option(1, "-r", "--round", help="Decimal places for millisecond values"),
) -> None:
    """Generate LaTeX \\newcommand definitions for the inference-time benchmark."""
    if not benchmark_csv.is_file():
        logger.error(f"Benchmark CSV not found: {benchmark_csv}")
        raise typer.Exit(code=1)

    df = pd.read_csv(benchmark_csv)
    value_col = "seconds" if include_vote else "inference_seconds"
    dtw = df[df["approach"] == "dtw"]
    ml = df[df["approach"] == "ml_majority"]
    if dtw.empty or ml.empty:
        logger.error("The benchmark CSV needs both a 'dtw' and an 'ml_majority' approach.")
        raise typer.Exit(code=1)

    def ms(value: float) -> str:
        return format(float(value) * 1000.0, f".{max(0, round_digits)}f")

    def quantity(name: str, seconds: float) -> str:
        return f"\\newcommand{{\\{name}}}{{\\SI{{{ms(seconds)}}}{{\\milli\\second}}\\xspace}}"

    def count(name: str, value) -> str:
        return f"\\newcommand{{\\{name}}}{{\\numprint{{{int(value)}}}\\xspace}}"

    def number(name: str, value: float, digits: int = 0) -> str:
        return f"\\newcommand{{\\{name}}}{{\\numprint{{{format(float(value), f'.{digits}f')}}}\\xspace}}"

    cells = df.pivot_table(index=["recording_length", "sampling_rate"],
                           columns="approach", values=value_col, aggfunc="mean")
    lines: list[str] = []

    # How the measurement was set up.
    lines.append(count("RuntimeDatasets", df["dataset_file"].nunique()))
    lines.append(count("RuntimePairsPerDataset", df["pair_index"].nunique()))
    lines.append(count("RuntimeRepeats", df["repeat"].nunique()))
    lines.append(count("RuntimeComparisons", len(ml)))
    lines.append(count("RuntimeRecordingLengthMin", df["recording_length"].min()))
    lines.append(count("RuntimeRecordingLengthMax", df["recording_length"].max()))
    lines.append(count("RuntimeSamplingRateMin", df["sampling_rate"].min()))
    lines.append(count("RuntimeSamplingRateMax", df["sampling_rate"].max()))
    lines.append(count("RuntimeChunkSize", df["chunk_size"].iloc[0]))

    # Per-comparison runtime, as cell means so a single outlier cannot set the range.
    lines.append(quantity("RuntimeDtwMin", cells["dtw"].min()))
    lines.append(quantity("RuntimeDtwMax", cells["dtw"].max()))
    lines.append(quantity("RuntimeMlMin", cells["ml_majority"].min()))
    lines.append(quantity("RuntimeMlMax", cells["ml_majority"].max()))

    # The ML approach is insensitive to the sampling rate: report the spread at the
    # longest recording, where any dependence would be most visible.
    longest = df["recording_length"].max()
    at_longest = cells.xs(longest, level="recording_length")["ml_majority"]
    lines.append(quantity("RuntimeMlLongestMin", at_longest.min()))
    lines.append(quantity("RuntimeMlLongestMax", at_longest.max()))

    # A single chunk-pair classification against the marginal cost of one more chunk
    # in the batch; the ratio is the fixed per-call overhead.
    if "single_chunk_seconds" in ml.columns and ml["single_chunk_seconds"].notna().any():
        single = float(ml["single_chunk_seconds"].mean())
        amortised = float((ml["inference_seconds"] / ml["n_chunks"]).mean())
        lines.append(quantity("RuntimeMlSingleChunk", single))
        lines.append(quantity("RuntimeMlPerChunkAmortised", amortised))
        lines.append(number("RuntimeMlChunkOverheadFactor", single / amortised if amortised else 0.0))
        lines.append(number("RuntimeMlChunkThroughput", 1.0 / amortised if amortised else 0.0))

    # What the majority vote costs on top of the inference.
    if "vote_seconds" in ml.columns and ml["seconds"].sum() > 0:
        # Share of the total, not the worst single measurement: the latter is
        # dominated by whichever comparison happened to be interrupted.
        lines.append(number("RuntimeMlVoteSharePct",
                            100.0 * ml["vote_seconds"].sum() / ml["seconds"].sum(), 1))

    # Speed-up of the learning-based approach over the distance-based one.
    speedup = cells["dtw"] / cells["ml_majority"]
    lines.append(number("RuntimeSpeedupMean", dtw[value_col].mean() / ml[value_col].mean()))
    lines.append(number("RuntimeSpeedupMax", speedup.max()))
    slower = speedup[speedup < 1.0]
    lines.append(count("RuntimeCellsDtwFaster", len(slower)))
    lines.append(count("RuntimeCells", len(speedup)))
    if not slower.empty:
        lines.append(count("RuntimeDtwFasterUpToLength",
                           max(length for length, _ in slower.index)))

    # The configuration that actually performs best, and what it costs.
    if results_csv is not None:
        if not results_csv.is_file():
            logger.warning(f"Results CSV not found: {results_csv}; skipping the best-configuration commands.")
        else:
            res = pd.read_csv(results_csv)
            best = res.loc[res["mcc_test"].idxmax()]
            length = int(best["trainride_start_seconds"])
            rate = int(float(best["sampling_rate"]))
            lines.append(count("RuntimeBestLength", length))
            lines.append(count("RuntimeBestSamplingRate", rate))
            lines.append(number("RuntimeBestMcc", float(best["mcc_test"]), 2))
            if (length, rate) in cells.index:
                row = cells.loc[(length, rate)]
                lines.append(quantity("RuntimeBestDtw", row["dtw"]))
                lines.append(quantity("RuntimeBestMl", row["ml_majority"]))
                lines.append(number("RuntimeBestSpeedup", row["dtw"] / row["ml_majority"]))
                lines.append(number("RuntimeBestShareOfMax", 100.0 * row["dtw"] / cells["dtw"].max()))
            else:
                logger.warning(f"The best configuration ({length}s, {rate}Hz) was not benchmarked; "
                               f"skipping its runtime commands.")

    text = "\n".join(lines) + "\n"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    typer.echo(f".tex file saved to {output_path}")


if __name__ == "__main__":
    app()
