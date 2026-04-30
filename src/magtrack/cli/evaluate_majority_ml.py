import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer

from magtrack.cli.train_ml_model import _build_model_from_hparams
from magtrack.utils.coloc_dataset import ColocSequentialEvalDataset
from magtrack.utils.evaluation import train_test_split
from magtrack.utils.loader import read_pickle, get_metadata_from_dataset
from magtrack.utils.ml import (
    build_eval_pairs,
)
from magtrack.utils.utils import resolve_seed

app = typer.Typer(help="Evaluate majority-vote ML model.")

def _build_sequential_arrays(df):
    dataset = ColocSequentialEvalDataset(df)
    return dataset.signals, dataset.ids, dataset.positive_pairs

def _build_split_arrays(df, workers: int):
    if workers <= 1:
        return _build_sequential_arrays(df)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        array_future = executor.submit(_build_sequential_arrays, df)
        arrays = array_future.result()
    return arrays


def _get_gpu_split_from_dfs(df, device: str, seed_int: int, preload_workers: int, test=False):
    signals_np, ids_np, pos_pairs_np = _build_split_arrays(df, preload_workers)
    signals = torch.from_numpy(signals_np).to(device)
    ids = torch.from_numpy(ids_np).to(device)
    pos_pairs = torch.from_numpy(pos_pairs_np).to(device)
    torch.manual_seed(seed_int)
    if test:
        pairs, labels = build_eval_pairs(pos_pairs, ids, signals.shape[0])
        return signals, pairs, labels
    return signals, ids, pos_pairs



@app.command()
def main(
    dataset_path: Path = typer.Option(..., help="Path to the dataset .pkl file"),
    model_path: Path = typer.Option(..., help="Path to the trained PyTorch model (.pth)"),
    hparams_path: Path = typer.Option(Path("./coloc_model/model_hparams.yaml"), help="Path to the model hparams yaml"),
    test_fraction: float = typer.Option(0.3, help="Fraction of data to use for testing"),
    threshold: float = typer.Option(0.0, help="Logit threshold for binary classification"),
    batch_size: int = typer.Option(1024, help="Batch size for evaluation"),
    seed: str = typer.Option("magtrack", help="Random seed for fixing the train/test split"),
    n_jobs: int = typer.Option(32, help="Worker processes for parallel signal stacking when building the eval dataset (1 = serial)."),
    results_dir: Path = typer.Option(Path("./results/coloc_ml"), help="Directory to save evaluation results (e.g. metrics, plots)"),
):
    hparams_path = Path(hparams_path)

    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = results_dir / "evaluation_results.majority_vote.csv"
    write_header = not results_csv_path.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_int = resolve_seed(seed)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    g = torch.Generator()
    g.manual_seed(seed_int)

    data = read_pickle(dataset_path)
    metadata_df = get_metadata_from_dataset(dataset_path)
    chunk_size_seconds = metadata_df.duration.values[0]
    sampling_rate = float(metadata_df.sampling_rate.values[0])
    target_len = int(chunk_size_seconds * sampling_rate)

    codes, uniques = pd.factorize(data["id"])
    data["class"] = codes

    train_df, test_df = train_test_split(
        data,
        test_frac=test_fraction,
        random_state=seed_int,
    )

    train_ids = set(train_df["id"].unique())
    test_ids = set(test_df["id"].unique())
    overlap_ids = train_ids.intersection(test_ids)
    if overlap_ids:
        print(f"Found {len(overlap_ids)} overlapping ids between train and test (possible leakage).")
    else:
        print("No overlapping ids found between train and test.")

    test_signals, test_pairs, test_labels = _get_gpu_split_from_dfs(test_df, device, seed_int, n_jobs, test=True)

    if device == "cuda":
        torch.cuda.synchronize()
        vram_mb = (test_signals.numel() * test_signals.element_size()) / (1024 ** 2)
        print(f"Signals on GPU: ~{vram_mb:.1f} MB")

    model, model_hparams = _build_model_from_hparams(hparams_path)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for start in range(0, test_pairs.shape[0], batch_size):
            p = test_pairs[start:start + batch_size]
            logits = model(test_signals[p[:, 0]], test_signals[p[:, 1]])
            preds = (logits > threshold).long().squeeze(1)
            all_predictions.extend(preds.cpu().tolist())
    predictions_np = np.array(all_predictions)

    test_df_reset = test_df.reset_index(drop=True)
    pairs_np = test_pairs.cpu().numpy()
    a_idx = pairs_np[:, 0]
    b_idx = pairs_np[:, 1]
    ids_arr = test_df_reset["id"].to_numpy()
    seg_arr = test_df_reset["segment_id"].to_numpy()
    src_arr = test_df_reset["source_file"].to_numpy()
    pred_df = pd.DataFrame({
        "id_A": ids_arr[a_idx],
        "segment_id_A": seg_arr[a_idx],
        "source_file_A": src_arr[a_idx],
        "id_B": ids_arr[b_idx],
        "segment_id_B": seg_arr[b_idx],
        "source_file_B": src_arr[b_idx],
        "prediction": predictions_np,
    })

    Path(results_dir / "majority_vote").mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(results_dir / "majority_vote" / f"{dataset_path.stem}.csv", index=False)


if __name__ == "__main__":
    app()
