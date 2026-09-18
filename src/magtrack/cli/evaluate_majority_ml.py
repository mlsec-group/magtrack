import glob
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from sklearn.metrics import f1_score, matthews_corrcoef

from magtrack.cli.train_ml_model import _build_model_from_hparams
from magtrack.utils.coloc_dataset import ColocSequentialEvalDataset
from magtrack.utils.evaluation import train_test_split
from magtrack.utils.loader import read_pickle, get_metadata_from_dataset
from magtrack.utils.ml import (
    build_eval_pairs,
)
from magtrack.utils.utils import resolve_seed

app = typer.Typer(help="Evaluate majority-vote ML model.")


# ---------------------------------------------------------------------------
# Majority-vote metrics pipeline
# ---------------------------------------------------------------------------

def _rolling_vote(codes: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Per-group centred rolling majority vote, computed from one prefix sum.

    Identical to ``groupby(group_cols)['prediction'].transform(lambda x:
    x.rolling(k, center=True, min_periods=1).mean().ge(0.5))`` -- the same window
    sums, just accumulated once for the whole column instead of running a Python
    rolling object per group. With one group per pair that inner loop dominated
    the step; this makes it a handful of array operations.
    """
    n = values.size
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.concatenate(([0], np.cumsum(sizes)))
    base = np.repeat(starts[:-1], sizes)
    within = np.arange(n) - base
    group_size = np.repeat(sizes, sizes)

    prefix = np.concatenate(([0], np.cumsum(values[order])))
    # pandas centres an even window one row below the row it belongs to.
    lo = base + np.maximum(within - k // 2, 0)
    hi = base + np.minimum(within + (k - 1) // 2 + 1, group_size)

    voted = (prefix[hi] - prefix[lo]) / (hi - lo) >= 0.5
    out = np.empty(n, dtype=np.int64)
    out[order] = voted.astype(np.int64)
    return out


def process_k_length(majority_vote_length, df, group_cols, codes=None):
    """Worker function to process a specific k-length for all groups."""
    required_cols = list(set(group_cols + ['prediction', 'segment_id_A', 'segment_id_B']))
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}. Available columns: {list(df.columns)}")

    if codes is None:
        codes = df.groupby(group_cols, sort=False).ngroup().to_numpy()
    majority_vote = _rolling_vote(codes, df['prediction'].to_numpy().astype(np.int64),
                                  majority_vote_length)

    y_true = (df['segment_id_A'] == df['segment_id_B']).astype(int)
    y_pred = 1 - majority_vote

    metrics = {
        'k': majority_vote_length,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'tp': ((y_true == 1) & (y_pred == 1)).sum(),
        'fp': ((y_true == 0) & (y_pred == 1)).sum(),
        'tn': ((y_true == 0) & (y_pred == 0)).sum(),
        'fn': ((y_true == 1) & (y_pred == 0)).sum(),
        'total': len(df)
    }
    return metrics


def run_pipeline(results_dir: Path = Path("./results/coloc_ml"), recompute: bool = False):
    """Run the majority-vote metrics aggregation pipeline."""
    majority_vote_dir = results_dir / "majority_vote"
    majority_vote_files = [
        f for f in glob.glob(str(majority_vote_dir / "all_coloc_first*.csv"))
        if not f.endswith("_metrics_by_k.csv")
    ]

    group_cols = ['segment_id_A', 'segment_id_B', 'source_file_A', 'source_file_B']
    majority_vote_lengths = [1, 3, 5, 7, 9, 11, 13]

    for file_path in majority_vote_files:
        output_name = str(Path(file_path).with_suffix("")) + "_metrics_by_k.csv"
        if os.path.exists(output_name) and not recompute:
            print(f"Metrics file {output_name} already exists. Skipping.")
            continue

        print(f"--- Processing {file_path} ---")
        df = pd.read_csv(file_path, index_col=False)

        if df.empty:
            print(f"Skipping {file_path}: File is empty.")
            continue

        # Group codes are the same for every k, so build them once per file.
        codes = df.groupby(group_cols, sort=False).ngroup().to_numpy()
        results = [process_k_length(k, df, group_cols, codes) for k in majority_vote_lengths]

        metrics_df = pd.DataFrame(results).sort_values('k')
        metrics_df.to_csv(output_name, index=False)
        print(f"  Saved metrics to {output_name}")


def build_master_csv(results_dir: Path = Path("./results/coloc_ml")):
    """Build the master metrics CSV from all *_metrics_by_k.csv files."""
    metric_files = glob.glob(str(results_dir / "majority_vote" / "*_metrics_by_k.csv"))
    all_metrics = []

    for f in metric_files:
        match = re.search(r'all_coloc_first(\d+)_(\d+)s_window(\d+)_(\d+)Hz', f)

        if match:
            f_duration = int(match.group(1))
            f_chunk = int(match.group(2))
            f_win = int(match.group(3))
            f_hz = int(match.group(4))

            try:
                df = pd.read_csv(f)
                if not df.empty and 'mcc' in df.columns and 'k' in df.columns:
                    for _, row in df.iterrows():
                        all_metrics.append({
                            'duration': f_duration,
                            'chunk_size': f_chunk,
                            'window_size': f_win,
                            'hz': f_hz,
                            'k': row['k'],
                            'mcc': row['mcc']
                        })
            except Exception as e:
                print(f"Skipping {f}: Error reading file - {e}")

    master_df = pd.DataFrame(all_metrics)

    if not master_df.empty:
        master_csv = results_dir / "majority_vote" / "master_metrics_by_k.csv"
        master_df.to_csv(master_csv, index=False)
        print(f"Master CSV saved to {master_csv}")
    else:
        print("No valid data found to save as master CSV.")

    return master_df


# ---------------------------------------------------------------------------
# ML evaluation (existing code)
# ---------------------------------------------------------------------------

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


@app.command("metrics")
def compute_metrics(
        results_dir: Path = typer.Option(Path("./results/coloc_ml"), help="Directory containing majority vote CSVs"),
        recompute: bool = typer.Option(False, "--recompute", help="Recompute metrics even if CSVs exist"),
):
    """Compute majority-vote metrics (MCC, F1, etc.) for all k-lengths."""
    run_pipeline(results_dir=results_dir, recompute=recompute)
    build_master_csv(results_dir=results_dir)


@app.command("master")
def compute_master(
        results_dir: Path = typer.Option(Path("./results/coloc_ml"), help="Directory containing metrics CSVs"),
):
    """Rebuild the master metrics CSV from all *_metrics_by_k.csv files."""
    build_master_csv(results_dir=results_dir)


@app.command("main")
def main(
        dataset_path: Path = typer.Option(..., help="Path to the dataset .pkl file"),
        model_path: Path = typer.Option(..., help="Path to the trained PyTorch model (.pth)"),
        hparams_path: Path = typer.Option(Path("./coloc_model/model_hparams.yaml"),
                                          help="Path to the model hparams yaml"),
        test_fraction: float = typer.Option(0.3, help="Fraction of data to use for testing"),
        threshold: float = typer.Option(0.0, help="Logit threshold for binary classification"),
        batch_size: int = typer.Option(1024, help="Batch size for evaluation"),
        seed: str = typer.Option("magtrack", help="Random seed for fixing the train/test split"),
        n_jobs: int = typer.Option(32,
                                   help="Worker processes for parallel signal stacking when building the eval dataset (1 = serial)."),
        results_dir: Path = typer.Option(Path("./results/coloc_ml"),
                                         help="Directory to save evaluation results (e.g. metrics, plots)"),
):
    hparams_path = Path(hparams_path)

    results_dir.mkdir(parents=True, exist_ok=True)

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
