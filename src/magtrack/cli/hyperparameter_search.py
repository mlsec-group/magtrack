import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import optuna
import torch
import typer
import yaml
from tqdm import trange, tqdm

from magtrack.utils.coloc_dataset import ColocDataset, ColocSequentialEvalDataset
from magtrack.utils.evaluation import train_test_split
from magtrack.utils.loader import read_pickle
from magtrack.utils.ml import (
    build_eval_pairs,
    evaluate,
    train_one_epoch_gpu,
)
from magtrack.utils.model import ColocationCNN
from magtrack.utils.utils import resolve_seed

app = typer.Typer(help="Hyperparameter optimization for SimpleCNN using Optuna.")

# Search-space constants shared between the trial body and the preload step.
HZ_CHOICES = [10, 20, 40, 60]
WINDOW_CHOICES = [5, 10, 100, 150]
CHUNK_CHOICES = [5, 10, 20, 30, 60]
START_CHOICES = [0, 60, 300, 600, 900]


def _dataset_path(base_data_dir: Path, start: int, chunk: int, window: int, hz: int) -> Path:
    return base_data_dir / f"all_coloc_first{start}_{chunk}s_window{window}_{hz}Hz.pkl"


# Process-shared array cache. Populated by the parallel preload step and read
# by trials in the main process; threads (Optuna n_jobs > 1) share it safely.
_array_cache: dict[tuple, tuple] = {}


def _load_split_arrays_impl(path_str: str, test_fraction: float, seed_int: int):
    """Pickle → split → ColocDataset → numpy arrays. Pure function; no caching.

    Suitable for use as a multiprocessing worker (top-level, picklable).
    """
    df = read_pickle(Path(path_str))
    train_df, _ = train_test_split(df, test_frac=test_fraction, random_state=seed_int)
    sampled_idx = train_df.sample(frac=test_fraction, random_state=seed_int, replace=False).index
    test_df = train_df.loc[sampled_idx].reset_index(drop=True)
    train_df = train_df.drop(index=sampled_idx).reset_index(drop=True)

    train_ds = ColocDataset(train_df, random_state=seed_int)
    test_ds = ColocDataset(test_df, random_state=seed_int)
    return (
        train_ds.signals, train_ds.ids, train_ds.positive_pairs,
        test_ds.signals, test_ds.ids, test_ds.positive_pairs,
    )


def _load_split_arrays(path_str: str, test_fraction: float, seed_int: int):
    """Cache-fronted loader. Returns the same tuple as `_load_split_arrays_impl`.

    Hits `_array_cache` after the parallel preload populates it; otherwise
    falls back to a serial in-process load (e.g. for paths missed by preload).
    """
    key = (path_str, test_fraction, seed_int)
    cached = _array_cache.get(key)
    if cached is not None:
        return cached
    result = _load_split_arrays_impl(path_str, test_fraction, seed_int)
    _array_cache[key] = result
    return result


def _preload_all_datasets(base_data_dir: Path, test_fraction: float, seed_int: int, n_jobs: int) -> None:
    """Parallel preload of every existing dataset file across the search space.

    Spawns up to `n_jobs` worker processes (spawn context, CUDA-safe) to
    decode the pickles and build the per-row signal arrays in parallel.
    Results are pickled back to the main process and stored in
    `_array_cache` so Optuna trials hit a warm cache.
    """
    paths = []
    for s in START_CHOICES:
        for c in CHUNK_CHOICES:
            for w in WINDOW_CHOICES:
                for hz in HZ_CHOICES:
                    p = _dataset_path(base_data_dir, s, c, w, hz)
                    if p.exists():
                        paths.append(p)
    if not paths:
        print(f"No datasets found under {base_data_dir}; skipping preload.")
        return

    args = [(str(p), test_fraction, seed_int) for p in paths]
    workers = max(1, min(n_jobs, len(args)))
    print(f"Preloading {len(args)} datasets with {workers} worker(s)...")

    if workers == 1:
        for a in tqdm(args, desc="Preloading datasets"):
            path_str, arrays = _preload_worker(a)
            _array_cache[(path_str, test_fraction, seed_int)] = arrays
        return

    # Spawn context so child processes don't inherit a CUDA-initialized parent.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for path_str, arrays in tqdm(
                ex.map(_preload_worker, args),
                total=len(args),
                desc="Preloading datasets",
        ):
            _array_cache[(path_str, test_fraction, seed_int)] = arrays


@lru_cache(maxsize=8)
def _load_split_arrays(path_str: str, test_fraction: float, seed_int: int):
    """Load + split + pair-build a dataset, cached across trials.

    Returns numpy arrays (signals/ids) and positive-pair index arrays for
    train/test. Identical (path, test_fraction, seed_int) triples reuse the
    cached result, skipping the expensive pickle/stack/groupby work.
    """
    df = read_pickle(Path(path_str))
    train_df, _ = train_test_split(df, test_frac=test_fraction, random_state=seed_int)
    sampled_idx = train_df.sample(frac=test_fraction, random_state=seed_int, replace=False).index
    test_df = train_df.loc[sampled_idx].reset_index(drop=True)
    train_df = train_df.drop(index=sampled_idx).reset_index(drop=True)

    train_ds = ColocSequentialEvalDataset(train_df)
    test_ds = ColocSequentialEvalDataset(test_df)
    return (
        train_ds.signals, train_ds.ids, train_ds.positive_pairs,
        test_ds.signals, test_ds.ids, test_ds.positive_pairs,
    )


def _preload_worker(args):
    path_str, test_fraction, seed_int = args
    return path_str, _load_split_arrays_impl(path_str, test_fraction, seed_int)


def _load_split_arrays(path_str: str, test_fraction: float, seed_int: int):
    """Cache-fronted loader. Returns the same tuple as `_load_split_arrays_impl`.

    Hits `_array_cache` after the parallel preload populates it; otherwise
    falls back to a serial in-process load (e.g. for paths missed by preload).
    """
    key = (path_str, test_fraction, seed_int)
    cached = _array_cache.get(key)
    if cached is not None:
        return cached
    result = _load_split_arrays_impl(path_str, test_fraction, seed_int)
    _array_cache[key] = result
    return result


def _prewarm_gpu_cache(test_fraction: float, seed_int: int, device_str: str) -> None:
    """Move every preloaded dataset to GPU once, in the main thread.

    Reason: `_get_gpu_split` is lru_cached, but functools.lru_cache uses a
    single global lock — N concurrent trials calling it serialize there even
    for different keys. Doing all H2D transfers up front avoids that lock
    contention and lets trials hit a fully-warm cache with zero data movement.
    """
    keys = [k for k in _array_cache if k[1] == test_fraction and k[2] == seed_int]
    if not keys:
        return
    print(f"Pre-warming GPU cache for {len(keys)} datasets on {device_str}...")
    for path_str, tf, si in tqdm(keys, desc="GPU prewarm"):
        _get_gpu_split(path_str, tf, si, device_str)
    if device_str.startswith("cuda"):
        torch.cuda.synchronize(device_str)


@lru_cache(maxsize=700)
def _get_gpu_split(path_str: str, test_fraction: float, seed_int: int, device_str: str):
    """GPU-resident view of the cached split, shared across parallel trials.

    Returns (train_signals, train_ids, train_pos_pairs, test_signals, test_pairs,
    test_labels) — all on `device_str`. test_pairs/test_labels are built once
    so all trials see the same fixed eval set.
    """
    arrays = _load_split_arrays(path_str, test_fraction, seed_int)
    train_signals = torch.from_numpy(arrays[0]).to(device_str)
    train_ids = torch.from_numpy(arrays[1]).to(device_str)
    train_pos_pairs = torch.from_numpy(arrays[2]).to(device_str)
    test_signals = torch.from_numpy(arrays[3]).to(device_str)
    test_ids = torch.from_numpy(arrays[4]).to(device_str)
    test_pos_pairs = torch.from_numpy(arrays[5]).to(device_str)
    test_pairs, test_labels = build_eval_pairs(
        test_pos_pairs, test_ids, test_signals.shape[0]
    )
    return (train_signals, train_ids, train_pos_pairs,
            test_signals, test_pairs, test_labels)


def _conv1d_out_length(length, kernel_size, stride, padding, dilation=1):
    return ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _pool1d_out_length(length, kernel_size, stride, padding=0, dilation=1):
    return ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


@app.command("optimize")
def optimize_simple_cnn(
        base_data_dir: Path = typer.Option(
            "/shares/research/magtrack2/anonym_coloc_datasets/normalized_trace_all_trains/",
            help="Base path for all datasets"),
        test_fraction: float = typer.Option(0.3, help="Fraction of data to use for testing"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducibility (converted to int)"),
        n_trials: int = typer.Option(1_000, help="Number of Optuna trials"),
        timeout: int = typer.Option(0, help="Timeout in seconds (0 = no timeout)"),
        max_epochs: int = typer.Option(200, help="Max epochs per trial"),
        num_gpus: int = typer.Option(1, help="Number of GPUs to parallelize trials across (0 for CPU)"),
        n_jobs: int = typer.Option(1,
                                   help="Number of concurrent Optuna trials (threads). On a single GPU, 2-4 typically saturates it."),
):
    seed_int = resolve_seed(seed)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    g = torch.Generator()
    g.manual_seed(seed_int)

    has_cuda = torch.cuda.is_available()
    available_gpus = torch.cuda.device_count() if has_cuda else 0

    if num_gpus < 0:
        raise typer.BadParameter("num_gpus must be >= 0")
    if n_jobs < 1:
        raise typer.BadParameter("n_jobs must be >= 1")
    if num_gpus > 0 and not has_cuda:
        raise RuntimeError("CUDA is not available but num_gpus > 0 was requested.")
    if num_gpus > available_gpus:
        raise RuntimeError(
            f"Requested num_gpus={num_gpus} but only {available_gpus} GPU(s) are available."
        )

    device = 'cuda' if has_cuda and num_gpus > 0 else 'cpu'

    # Parallel-preload every existing dataset before any CUDA work begins.
    # Workers fork off cleanly because CUDA hasn't been initialized yet here
    # (Optuna trials are what trigger the first CUDA allocation).
    _preload_all_datasets(base_data_dir, test_fraction, seed_int, n_jobs)

    # Move all preloaded datasets to GPU once, in the main thread, so concurrent
    # trials never have to do H2D transfers (or contend on the lru_cache lock).
    if device == "cuda":
        _prewarm_gpu_cache(test_fraction, seed_int, "cuda:0")

    def objective(trial: optuna.Trial):
        if device == 'cuda':
            gpu_id = trial.number % max(1, num_gpus)
            torch.cuda.set_device(gpu_id)
            trial_device = f"cuda:{gpu_id}"
        else:
            trial_device = 'cpu'

        # --- 1. Dataset Configuration Hyperparameters ---
        hz = trial.suggest_categorical("hz", [10, 20, 40, 60])
        window_size_seconds = trial.suggest_categorical("window_size_seconds", [5, 10, 100, 150])
        chunk_size_seconds = trial.suggest_categorical("chunk_size_seconds", [5, 10, 20, 30, 60])
        train_ride_start_seconds = trial.suggest_categorical("train_ride_start_seconds", [0, 60, 300, 600, 900])

        target_len = int(chunk_size_seconds * hz)

        full_path = base_data_dir / f"all_coloc_first{train_ride_start_seconds}_{chunk_size_seconds}s_window{window_size_seconds}_{hz}Hz.pkl"

        if not full_path.exists():
            print(f"Skipping trial: Dataset {full_path} not found.")
            raise optuna.TrialPruned()

        # --- 2. Load Data (GPU tensors shared across parallel trials) ---
        (
            train_signals, train_ids, train_pos_pairs,
            test_signals, test_pairs, test_labels,
        ) = _get_gpu_split(str(full_path), test_fraction, seed_int, str(trial_device))

        # --- 3. Model Architecture Hyperparameters ---
        num_conv_layers = trial.suggest_int("num_conv_layers", 0, 6)
        if num_conv_layers > 0:
            base_channels = trial.suggest_categorical("base_channels", [8, 16, 32, 64])
            conv_channels = [base_channels * (2 ** i) for i in range(num_conv_layers)]

            kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
            kernel_sizes = [kernel_size] * num_conv_layers

            stride = trial.suggest_categorical("stride", [1, 2])
            strides = [stride] * num_conv_layers

            pool_kernel = trial.suggest_categorical("pool_kernel", [2, 3])
            pool_kernel_sizes = [pool_kernel] * num_conv_layers
            pool_strides = [pool_kernel] * num_conv_layers
        else:
            conv_channels = []
            kernel_sizes = []
            strides = []
            pool_kernel_sizes = []
            pool_strides = []

        embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64, 128, 256])
        fc_layers = trial.suggest_int("fc_layers", 1, 6)
        fc_hidden_dims = [trial.suggest_categorical(f"fc_dim_{i}", [64, 128, 256, 512]) for i in range(fc_layers)]

        fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.5)

        lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        # --- 4. Prune Invalid Architectures ---
        length = target_len
        paddings = [k // 2 for k in kernel_sizes]
        if num_conv_layers > 0:
            for k, s, p, pk, ps in zip(kernel_sizes, strides, paddings, pool_kernel_sizes, pool_strides):
                length = _conv1d_out_length(length, k, s, p)
                if length < 2:
                    raise optuna.TrialPruned()
                length = _pool1d_out_length(length, pk, ps)
                if length < 2:
                    raise optuna.TrialPruned()

        try:
            model = ColocationCNN(
                input_channels=2,
                signal_length=target_len,
                embedding_dim=embedding_dim,
                num_conv_layers=num_conv_layers,
                conv_channels=conv_channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                pool_kernel_sizes=pool_kernel_sizes,
                pool_strides=pool_strides,
                fc_hidden_dims=fc_hidden_dims,
                fc_dropout=fc_dropout,
            ).to(trial_device)
        except RuntimeError as exc:
            if "max_pool1d() Invalid computed output size" in str(exc):
                raise optuna.TrialPruned() from exc
            raise

        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=1_000)

        best_mcc = 0.0
        stopped_epoch = float(max_epochs)
        epoch_loss = 0.0

        for epoch in trange(max_epochs, leave=False):
            epoch_loss = train_one_epoch_gpu(
                model, train_signals, train_ids, train_pos_pairs,
                batch_size, criterion, optimizer, scheduler, trial_device,
            )

            model.eval()
            acc, precision, recall, f1, prevalence, specificity, negative_predictive_value, mcc = evaluate(
                model, test_signals, test_pairs, test_labels, batch_size, threshold=0.0,
            )
            model.train()

            best_mcc = max(best_mcc, mcc)

            # Early stop if target reached
            if mcc >= 0.9 and epoch + 1 >= 10:
                print(f"Early stopping at epoch {epoch + 1} with mcc {mcc:.4f}")
                stopped_epoch = float(epoch + 1)
                save_model_dir = Path("optuna_models")
                save_model_dir.mkdir(exist_ok=True)
                model_save_path = save_model_dir / f"trial_{trial.number}_model_{mcc:.4f}.pth"
                torch.save(model.state_dict(), model_save_path)
                break

        trial.set_user_attr("best_mcc", float(best_mcc))
        print(f"Trial {trial.number} completed with best_mcc: {best_mcc:.4f}")

        # --- RETURN 4 OBJECTIVES ---
        # 1. Maximize MCC, 2. Minimize epoch loss, 3. Minimize epoch of early stopping
        return float(best_mcc), float(epoch_loss), stopped_epoch

    # Use NSGA-II for multi-objective optimization
    sampler = optuna.samplers.NSGAIISampler(seed=seed_int)

    # We must remove the MedianPruner. Genetic algorithms need to evaluate full populations 
    # and cannot randomly prune trials based on incomplete data without breaking the evolution.
    study = optuna.create_study(
        study_name="colocoation_cnn_hyperparameter_search",
        storage="sqlite:///ultra_parallel_paper_letsgo.db",
        load_if_exists=True,
        directions=["maximize", "minimize", "minimize"],
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None if timeout <= 0 else timeout,
        n_jobs=n_jobs,
    )

    best_trials = study.best_trials
    print(f"Number of trials on the Pareto front: {len(best_trials)}")

    if best_trials:
        # Pick the trial with the highest MCC (index 0 of values)
        best_trial = max(best_trials, key=lambda t: t.values[0])
        print(f"\nBest Pareto Trial based on MCC (ID: {best_trial.number}):")
        print(
            f"  MCC: {best_trial.values[0]:.4f} | epoch_loss: {best_trial.values[1]:.4f} | stopped_epoch: {best_trial.values[2]:.0f}")

        # Extract and format hyperparameters to save to YAML
        h = best_trial.params
        num_conv_layers = h['num_conv_layers']
        base_channels = h.get('base_channels')
        kernel_size = h.get('kernel_size')
        stride = h.get('stride')
        pool_kernel = h.get('pool_kernel')

        fc_layers = h['fc_layers']
        fc_hidden_dims = [h[f"fc_dim_{i}"] for i in range(fc_layers)]

        hparams_dict = {
            "input_channels": 2,
            "embedding_dim": h["embedding_dim"],
            "num_conv_layers": num_conv_layers,
            "base_channels": base_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "pool_kernel": pool_kernel,
            "fc_hidden_dims": fc_hidden_dims,
            "fc_dropout": h["fc_dropout"],

            # Dataset & Optimization parameters
            "hz": h["hz"],
            "window_size_seconds": h["window_size_seconds"],
            "chunk_size_seconds": h["chunk_size_seconds"],
            "train_ride_start_seconds": h["train_ride_start_seconds"],
            "lr": h["lr"],
            "batch_size": h["batch_size"],
            "weight_decay": h["weight_decay"],
        }

        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)
        hparams_path = results_dir / "model_hparams.yaml"

        with open(hparams_path, "w", encoding="utf-8") as f:
            yaml.dump(hparams_dict, f, sort_keys=False)

        print(f"Saved best hyperparameters to {hparams_path}")


if __name__ == "__main__":
    app()
