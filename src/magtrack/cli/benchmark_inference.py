from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from tqdm import tqdm

from magtrack.cli.train_ml_model import _build_model_from_hparams
from magtrack.utils.distance import dtw_distance
from magtrack.utils.loader import read_pickle
from magtrack.utils.utils import resolve_seed

app = typer.Typer(help="Benchmark colocation inference time.")

_CSV_COLUMNS = [
    "approach", "dataset_file", "recording_length", "sampling_rate",
    "chunk_size", "window_size", "n_samples", "n_chunks",
    "repeat", "pair_index", "seconds", "inference_seconds", "vote_seconds",
    "single_chunk_seconds",
]

_NAME_RE = re.compile(r"_coloc_first(\d+)_(\d+)s_window(\d+)_(\d+)Hz$")


def _rolling_vote(predictions: np.ndarray, length: int) -> np.ndarray:
    """The majority vote evaluate-majority-ml applies to a recording pair.

    Mirrors ``process_k_length``: a centred rolling mean over the per-chunk
    predictions, thresholded at 0.5.
    """
    return (
        pd.Series(predictions.astype(int))
        .rolling(window=length, center=True, min_periods=1)
        .mean()
        .ge(0.5)
        .astype(int)
        .to_numpy()
    )


def _discover_datasets(data_dir: Path, first: int, chunk: int, hz: int) -> list[Path]:
    """Datasets for one (recording length, sampling rate), any rolling window."""
    return sorted(data_dir.glob(f"*_coloc_first{first}_{chunk}s_window*_{hz}Hz.pkl"))


def _window_of(path: Path) -> int:
    match = _NAME_RE.search(path.stem)
    return int(match.group(3)) if match else -1


def _load_recordings(pkl_path: Path, expected_chunks: int) -> list[np.ndarray]:
    """Per recording, its chunks as one ``(k, chunk_len)`` array.

    A recording is one (segment_id, source_file); its chunks are ordered by id,
    which is the order the majority vote rolls over.
    """
    data = read_pickle(pkl_path)
    required = {"id", "segment_id", "source_file", "data"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{pkl_path.name} is missing columns {missing}")

    recordings: list[np.ndarray] = []
    for _, group in data.groupby(["segment_id", "source_file"], sort=False, dropna=False):
        if len(group) != expected_chunks:
            continue
        group = group.sort_values("id")
        chunks = [t["magnitude"].to_numpy(dtype=np.float32) for t in group["data"]]
        lengths = {len(c) for c in chunks}
        if len(lengths) != 1:
            continue
        recordings.append(np.stack(chunks))
    return recordings


@app.command("run")
def run(
        data_dir: Path = typer.Option(
            Path("datasets/coloc_datasets/normalized_trace_all_trains"),
            help="Directory holding the colocation .pkl datasets."),
        output: Path = typer.Option(
            Path("results/benchmark/inference_times.csv"),
            help="CSV to write one row per timed pair to."),
        model_path: Path = typer.Option(
            Path("coloc_model/colocation_net.pth"), help="Trained colocation model."),
        hparams_path: Path = typer.Option(
            Path("coloc_model/model_hparams.yaml"), help="Hyperparameters of that model."),
        recording_length: List[int] = typer.Option(
            [10, 30, 60], help="Recording lengths in seconds. Repeatable."),
        sampling_rate: List[int] = typer.Option(
            [10, 20, 40, 60], help="Sampling rates in Hz. Repeatable."),
        chunk_duration: int = typer.Option(
            5, help="Chunk duration of the datasets to use; the vote runs over length/chunk chunks."),
        datasets_per_cell: int = typer.Option(
            5, help="Datasets sampled per (length, sampling rate), identical for both approaches."),
        pairs: int = typer.Option(20, help="Recording pairs timed per dataset per repeat."),
        repeats: int = typer.Option(5, help="Repetitions, to average over background CPU load."),
        warmup: int = typer.Option(
            3, help="Untimed iterations before measuring, so lazy initialisation "
                    "does not land in the first timing."),
        radius: int = typer.Option(1, help="DTW Sakoe-Chiba radius."),
        threshold: float = typer.Option(0.0, help="Logit threshold for the ML prediction."),
        seed: str = typer.Option("magtrack", help="Seed for dataset and pair selection."),
        approaches: List[str] = typer.Option(
            ["dtw", "ml_majority"], help="Approaches to time. Repeatable."),
):
    """Time both colocation approaches across recording lengths and sampling rates."""
    unknown = set(approaches) - {"dtw", "ml_majority"}
    if unknown:
        typer.echo(f"Unknown approach(es): {sorted(unknown)}", err=True)
        raise typer.Exit(code=1)
    if not data_dir.is_dir():
        typer.echo(f"Dataset directory not found: {data_dir}", err=True)
        raise typer.Exit(code=1)

    model = None
    if "ml_majority" in approaches:
        for path in (model_path, hparams_path):
            if not path.exists():
                typer.echo(f"Not found: {path}. Run scripts/coloc_ml/train.sh first.", err=True)
                raise typer.Exit(code=1)
        model, _ = _build_model_from_hparams(hparams_path)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to("cpu")
        model.eval()

    seed_int = resolve_seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=_CSV_COLUMNS).to_csv(output, index=False)

    typer.echo(f"Data dir:     {data_dir}")
    typer.echo(f"Lengths:      {sorted(set(recording_length))} s")
    typer.echo(f"Sampling:     {sorted(set(sampling_rate))} Hz")
    typer.echo(f"Chunks:       {chunk_duration}s -> k = length/{chunk_duration}")
    typer.echo(f"Per cell:     {datasets_per_cell} dataset(s) x {pairs} pair(s) x {repeats} repeat(s)")
    typer.echo(f"Approaches:   {list(approaches)}  (device: cpu)")
    typer.echo(f"Output:       {output}")

    n_cells = len(set(recording_length)) * len(set(sampling_rate))
    cell = 0
    missing_cells: list[str] = []

    for length in sorted(set(recording_length)):
        if length % chunk_duration != 0:
            logger.warning(f"Recording length {length}s is not a multiple of the "
                           f"{chunk_duration}s chunk; skipping.")
            continue
        n_chunks = length // chunk_duration

        for hz in sorted(set(sampling_rate)):
            cell += 1
            candidates = _discover_datasets(data_dir, length, chunk_duration, hz)
            if not candidates:
                typer.echo(f"\n[{cell}/{n_cells}] {length}s {hz}Hz: no datasets, skipping.")
                missing_cells.append(f"first{length}_{chunk_duration}s_*_{hz}Hz")
                continue

            rng = random.Random(f"{seed_int}-{length}-{hz}")
            chosen = sorted(rng.sample(candidates, min(datasets_per_cell, len(candidates))))
            typer.echo(f"\n[{cell}/{n_cells}] {length}s {hz}Hz, k={n_chunks}: "
                       f"{len(chosen)}/{len(candidates)} dataset(s)")

            rows: list[dict] = []
            for pkl_path in chosen:
                try:
                    recordings = _load_recordings(pkl_path, n_chunks)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    logger.warning(f"Failed reading {pkl_path.name}: {exc}")
                    continue
                if len(recordings) < 2:
                    logger.warning(f"{pkl_path.name}: fewer than 2 usable recordings, skipping.")
                    continue

                pair_rng = random.Random(f"{seed_int}-{pkl_path.name}")
                pair_idx = [
                    tuple(pair_rng.sample(range(len(recordings)), 2))
                    for _ in range(pairs)
                ]

                window_size = _window_of(pkl_path)
                n_samples = int(recordings[0].size)
                base = {
                    "dataset_file": pkl_path.name,
                    "recording_length": length,
                    "sampling_rate": hz,
                    "chunk_size": chunk_duration,
                    "window_size": window_size,
                    "n_samples": n_samples,
                    "n_chunks": n_chunks,
                }

                traces = [r.reshape(-1).astype(np.float64) for r in recordings]
                tensors = ([torch.from_numpy(r) for r in recordings]
                           if model is not None else None)

                if pair_idx:
                    wa, wb = pair_idx[0]
                    for _ in range(max(0, warmup)):
                        if "dtw" in approaches:
                            dtw_distance(traces[wa], traces[wb], radius=radius)
                        if model is not None:
                            with torch.no_grad():
                                model(tensors[wa], tensors[wb])
                                model(tensors[wa][:1], tensors[wb][:1])

                desc = f"  {pkl_path.name[:38]}"
                for repeat in tqdm(range(1, repeats + 1), desc=desc, leave=False):
                    for index, (a, b) in enumerate(pair_idx):
                        if "dtw" in approaches:
                            t0 = time.perf_counter()
                            dtw_distance(traces[a], traces[b], radius=radius)
                            elapsed = time.perf_counter() - t0
                            rows.append({**base, "approach": "dtw", "repeat": repeat,
                                         "pair_index": index, "seconds": elapsed,
                                         "inference_seconds": elapsed,
                                         "vote_seconds": 0.0,
                                         "single_chunk_seconds": float("nan")})

                        if model is not None:
                            x1, x2 = tensors[a], tensors[b]
                            t0 = time.perf_counter()
                            with torch.no_grad():
                                logits = model(x1, x2)
                                preds = (logits > threshold).long().squeeze(1).numpy()
                            t_inference = time.perf_counter() - t0

                            t1 = time.perf_counter()
                            _rolling_vote(preds, n_chunks)
                            t_vote = time.perf_counter() - t1

                            t2 = time.perf_counter()
                            with torch.no_grad():
                                model(x1[:1], x2[:1])
                            t_single = time.perf_counter() - t2

                            rows.append({**base, "approach": "ml_majority", "repeat": repeat,
                                         "pair_index": index,
                                         "seconds": t_inference + t_vote,
                                         "inference_seconds": t_inference,
                                         "vote_seconds": t_vote,
                                         "single_chunk_seconds": t_single})

            if rows:
                pd.DataFrame(rows, columns=_CSV_COLUMNS).to_csv(
                    output, mode="a", header=False, index=False)
                summary = (pd.DataFrame(rows).groupby("approach")["seconds"]
                           .agg(["mean", "std", "count"]))
                for approach, stats in summary.iterrows():
                    typer.echo(f"    {approach:<12} {stats['mean']:.6f}s "
                               f"+-{stats['std']:.6f} over {int(stats['count'])} timings")

    typer.echo(f"\nDone. {output}")
    if missing_cells:
        typer.echo(f"No datasets for {len(missing_cells)} cell(s): {', '.join(missing_cells)}")


if __name__ == "__main__":
    app()
