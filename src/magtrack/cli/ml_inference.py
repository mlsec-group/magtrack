import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import fcntl
import os

import pandas as pd
import torch

from magtrack.cli.train_ml_model import _build_model_from_hparams
from magtrack.utils.coloc_dataset import ColocSequentialEvalDataset
from magtrack.utils.evaluation import train_test_split
from magtrack.utils.loader import read_pickle, get_metadata_from_dataset
from magtrack.utils.ml import (
    build_eval_pairs,
    evaluate,
)
from magtrack.utils.utils import resolve_seed


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to the dataset .pkl file")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the trained PyTorch model (.pth)")
    parser.add_argument("--hparams-path", type=Path, default=Path("./coloc_model/model_hparams.yaml"),
                        help="Path to the model hparams yaml")
    parser.add_argument("--test-fraction", type=float, default=0.3, help="Fraction of data to use for testing")
    parser.add_argument("--threshold", type=float, default=0.0, help="Logit threshold for binary classification")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for evaluation")
    parser.add_argument("--seed", type=str, default="magtrack", help="Random seed for fixing the train/test split")
    parser.add_argument("--n-jobs", type=int, default=32,
                        help="Worker processes for parallel signal stacking when building the eval dataset (1 = serial).")
    parser.add_argument("--results-dir", type=Path, default=Path("./results"),
                        help="Directory to save evaluation results (e.g. metrics, plots)")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    model_path = args.model_path
    hparams_path = args.hparams_path
    test_fraction = args.test_fraction
    threshold = args.threshold
    batch_size = args.batch_size
    results_dir = args.results_dir
    seed = args.seed
    preload_workers = args.n_jobs

    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = results_dir / "evaluation_results.results_ml.csv"

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

    test_signals, test_pairs, test_labels = _get_gpu_split_from_dfs(test_df, device, seed_int, preload_workers,
                                                                    test=True)

    if device == "cuda":
        torch.cuda.synchronize()
        vram_mb = (test_signals.numel() * test_signals.element_size()) / (1024 ** 2)
        print(f"Signals on GPU: ~{vram_mb:.1f} MB")

    model, model_hparams = _build_model_from_hparams(hparams_path)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    metrics = evaluate(
        model,
        test_signals,
        test_pairs,
        test_labels,
        batch_size,
        threshold=0.0,
    )

    print(f"Evaluation results for {dataset_path.name} (Seed {seed_int}):")
    print(f"  Accuracy:      {metrics['acc']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1 Score:      {metrics['f1']:.4f}")
    print(f"  Prevalence:    {metrics['prevalence']:.4f}")
    print(f"  Specificity:   {metrics['specificity']:.4f}")
    print(f"  NPV:           {metrics['npv']:.4f}")
    print(f"  MCC:           {metrics['mcc']:.4f}")

    row = {
        'seed': seed_int,
        'dataset_file': dataset_path.name,
        'accuracy': metrics['acc'],
        'f1': metrics['f1'],
        'mcc': metrics['mcc'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'specificity': metrics['specificity'],
        'negative_predictive_value': metrics['npv'],
        'prevalence': metrics['prevalence'],
        'TP': metrics['tp'],
        'FP': metrics['fp'],
        'TN': metrics['tn'],
        'FN': metrics['fn'],
        'samples': metrics['total'],
    }

    # Parallel jobs all append to this one CSV. Decide the header inside an
    # exclusive lock and from the file's actual size: checking os.path.exists()
    # up front lets every job that starts before the first write conclude the
    # file is missing, and each writes its own header row.
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv_path, "a", newline="") as results_fh:
        fcntl.flock(results_fh.fileno(), fcntl.LOCK_EX)
        try:
            write_header = os.fstat(results_fh.fileno()).st_size == 0
            pd.DataFrame([row]).to_csv(results_fh, header=write_header, index=False)
        finally:
            fcntl.flock(results_fh.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
