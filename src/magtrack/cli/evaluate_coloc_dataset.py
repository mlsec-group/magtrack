import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import typer
from sklearn.metrics import confusion_matrix

from magtrack.utils.distance import dtw_distance
from magtrack.utils.distance_dataset import load_and_stitch
from magtrack.utils.evaluation import find_best_threshold
from magtrack.utils.loader import read_pickle

app = typer.Typer(help="Evaluate colocation using pairwise distance metrics.")
list_app = typer.Typer(help="List dataset entities.")
app.add_typer(list_app, name="list")


@list_app.command("ids")
def list_ids(
        pkl_path: Path = typer.Argument(..., help="Path to a single colocation .pkl dataset."),
):
    """Load the dataset and print unique ids with their occurrence counts.

    Uses pandas built-ins (value_counts) to compute counts concisely.
    Returns a DataFrame with columns ['id', 'count'].
    """
    data = read_pickle(pkl_path)
    counts = data["id"].value_counts()
    typer.echo("IDs in dataset (id\tcount):")
    for id_, cnt in counts.items():
        typer.echo(f"\t{id_}\t{cnt}")


@list_app.command("segments")
def list_segments(
        pkl_path: Path = typer.Argument(..., help="Path to a single colocation .pkl dataset."),
):
    data = read_pickle(pkl_path)
    segments = data["segment_id"].unique()
    typer.echo("Segments in dataset (id\tcount):")
    for segment in segments:
        segments_df = data[data["segment_id"] == segment]
        source_files = segments_df["source_file"].unique()
        typer.echo(f"\t{segment}\t{len(source_files)}")


@app.command("detect-colocation-dtw")
def detect_colocation_dtw(
        pkl_path: Path = typer.Argument(..., help="Path to a single colocation .pkl dataset."),
        samples: int = typer.Option(0, help="Number of random pairs to evaluate (0 = all)."),
        threshold: float = typer.Option(..., help="Threshold for colocation detection."),
        radius: int = typer.Option(1, help="DTW Sakoe-Chiba radius."),
        seed: str = typer.Option("magtrack", help="Seed for random number generator."),
        verbose: bool = typer.Option(False, help="Print detailed output for each pair."),
):
    data, meta = load_and_stitch(pkl_path)
    required = {"segment_id", "data"}
    missing = required - set(data.columns)
    if missing:
        raise typer.BadParameter(f"Missing columns: {sorted(missing)}")
    if len(data) < 2:
        raise typer.BadParameter("Dataset must contain at least 2 rows")
    if samples < 0:
        raise typer.BadParameter("samples must be >= 0")
    if radius < 1:
        raise typer.BadParameter("radius must be >= 1")

    def _trace(row):
        v = row["data"]
        if hasattr(v, "columns") and "magnitude" in v.columns:
            return v["magnitude"].to_numpy(dtype=np.float32)
        return np.asarray(v, dtype=np.float32)

    pair_idx = list(combinations(range(len(data)), 2))
    total_possible = len(pair_idx)
    if samples > 0:
        rng = random.Random(seed)
        pair_idx = rng.sample(pair_idx, k=min(samples, total_possible))

    y_true: list[int] = []
    y_pred: list[int] = []
    times_s: list[float] = []
    pair_distances: list[float] = []
    pair_classes: list[int] = []
    per_anchor: dict[str, list[tuple[float, bool]]] = {}

    for i, j in pair_idx:
        row_i = data.iloc[i]
        row_j = data.iloc[j]
        trace_i = _trace(row_i)
        trace_j = _trace(row_j)

        t0 = time.perf_counter()
        distance = dtw_distance(trace_i, trace_j, radius=radius)

        same_segment = row_i["segment_id"] == row_j["segment_id"]
        predicted_same = distance <= threshold

        y_true.append(1 if same_segment else 0)
        y_pred.append(1 if predicted_same else 0)
        pair_distances.append(float(distance))
        # Threshold utility expects: 0 = same class, 1 = different class
        pair_classes.append(0 if same_segment else 1)

        id_i = str(row_i.get("id", i))
        id_j = str(row_j.get("id", j))
        per_anchor.setdefault(id_i, []).append((distance, bool(same_segment)))
        per_anchor.setdefault(id_j, []).append((distance, bool(same_segment)))

        elapsed = time.perf_counter() - t0
        times_s.append(elapsed)

        if verbose:
            typer.echo(
                f"{id_i} <-> {id_j} | dist={distance:.6f} | same={int(same_segment)} "
                f"| pred={int(predicted_same)} | time_s={elapsed:.6f}"
            )

    if not y_true:
        raise typer.BadParameter("No row pairs available for evaluation")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mcc_denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0.0

    rr_values: list[float] = []
    for entries in per_anchor.values():
        ranked = sorted(entries, key=lambda x: x[0])
        for rank, (_, is_positive) in enumerate(ranked, start=1):
            if is_positive:
                rr_values.append(1.0 / rank)
                break
    mrr = float(np.mean(rr_values)) if rr_values else 0.0

    d_flat = np.asarray(pair_distances, dtype=np.float64)
    g_flat = np.asarray(pair_classes, dtype=np.int64)
    best_threshold, best_info = find_best_threshold(d_flat, g_flat, "mcc")

    typer.echo(f"F1:  {f1:.4f}")
    typer.echo(f"MCC: {mcc:.4f}")
    typer.echo(f"MRR: {mrr:.4f}")
    typer.echo(f"Best threshold from evaluated pairs (MCC): {best_threshold:.6f}")

    if verbose:
        typer.echo(f"Pairs evaluated: {len(y_true)} / {total_possible}")
        typer.echo(f"Threshold: {threshold}")
        if isinstance(best_info, dict):
            typer.echo(
                "Best-threshold stats: "
                f"MCC={best_info['mcc']:.4f} F1={best_info['f1']:.4f} "
                f"Acc={best_info['accuracy']:.4f}"
            )
        else:
            typer.echo(f"Best-threshold MCC score: {float(best_info):.4f}")
        typer.echo(f"Precision={precision:.4f} Recall={recall:.4f}")
        typer.echo(f"TP={tp} FP={fp} TN={tn} FN={fn}")

    total_s = float(np.sum(times_s))
    avg_s = float(np.mean(times_s))
    min_s = float(np.min(times_s))
    max_s = float(np.max(times_s))
    typer.echo(
        f"Time per combination: avg={avg_s:.6f}s min={min_s:.6f}s max={max_s:.6f}s total={total_s:.3f}s"
    )


if __name__ == "__main__":
    app()
