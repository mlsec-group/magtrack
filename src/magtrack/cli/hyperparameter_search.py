import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import optuna
import torch
import typer
import yaml
from tqdm import trange, tqdm

from magtrack.utils.coloc_dataset import ColocSequentialEvalDataset
from magtrack.utils.evaluation import consistent_train_test_split
from magtrack.utils.loader import read_pickle
from magtrack.utils.ml import (
    RecordingWindowSampler,
    build_eval_pairs,
    evaluate,
    train_one_epoch_gpu,
)
from magtrack.utils.model import ColocationCNN
from magtrack.utils.utils import resolve_seed

app = typer.Typer(help="Hyperparameter optimization for SimpleCNN using Optuna.")

# Search-space constants shared between the trial body and the preload step.
HZ_CHOICES = [10, 20, 40, 60]
WINDOW_CHOICES = [1, 5, 10, 50, 100, 150]
CHUNK_CHOICES = [5, 10, 20, 30, 60]
START_CHOICES = [0, 60, 300, 600, 900]

# A trial's score is the mean validation MCC over its last N epochs.
SCORE_LAST_N_EPOCHS = 10

# Every trial trains with this batch size; it is not part of the search space.
BATCH_SIZE = 128

# How long a trial waits for the SQLite lock of the Optuna DB before giving up.
SQLITE_LOCK_TIMEOUT_SECONDS = 300

# A storage error fails only the trial it hits; without this it would abort the whole search.
CAUGHT_TRIAL_ERRORS = (optuna.exceptions.StorageInternalError,)

# Trial parameters that select the dataset. They are searched in phase 1 and frozen in phase 2.
DATASET_PARAMS = ("hz", "window_size_seconds", "chunk_size_seconds", "train_ride_start_seconds")


def _dataset_path(base_data_dir: Path, start: int, chunk: int, window: int, hz: int) -> Path:
    return base_data_dir / f"all_coloc_first{start}_{chunk}s_window{window}_{hz}Hz.pkl"


_array_cache: dict[tuple, tuple] = {}
_skipped_datasets: dict[str, int] = {}


def _store_preloaded(path_str: str, arrays: tuple, test_fraction: float, seed_int: int,
                     min_val_positives: int) -> None:
    """Cache the arrays of a dataset, or mark it as skipped if its validation set is too small."""
    num_val_positives = int(arrays[5].shape[0])
    if num_val_positives < min_val_positives:
        _skipped_datasets[path_str] = num_val_positives
        return
    _array_cache[(path_str, test_fraction, seed_int)] = arrays


def _preload_all_datasets(base_data_dir: Path, test_fraction: float, seed_int: int, n_jobs: int,
                          min_val_positives: int = 0,
                          start_choices=START_CHOICES, chunk_choices=CHUNK_CHOICES,
                          window_choices=WINDOW_CHOICES, hz_choices=HZ_CHOICES) -> None:
    """Parallel preload of every existing dataset file across the search space.

    Spawns up to `n_jobs` worker processes (spawn context, CUDA-safe) to
    decode the pickles and build the per-row signal arrays in parallel.
    Results are pickled back to the main process and stored in
    `_array_cache` so Optuna trials hit a warm cache. Datasets with fewer than
    `min_val_positives` positive validation pairs are recorded in
    `_skipped_datasets` instead and never moved to the GPU.
    """
    paths = []
    for s in start_choices:
        for c in chunk_choices:
            for w in window_choices:
                for hz in hz_choices:
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
            _store_preloaded(path_str, arrays, test_fraction, seed_int, min_val_positives)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for path_str, arrays in tqdm(
                    ex.map(_preload_worker, args),
                    total=len(args),
                    desc="Preloading datasets",
            ):
                _store_preloaded(path_str, arrays, test_fraction, seed_int, min_val_positives)

    if _skipped_datasets:
        print(f"Skipping {len(_skipped_datasets)} of {len(args)} datasets with fewer than "
              f"{min_val_positives} validation positive pairs:")
        for path_str, num_val_positives in sorted(_skipped_datasets.items(), key=lambda kv: kv[1]):
            print(f"  {Path(path_str).name}: {num_val_positives}")


def _load_split_arrays_impl(path_str: str, test_fraction: float, seed_int: int):
    """Pickle → split → pair-build → numpy arrays. Pure function; no caching.

    The held-out test segments are discarded; the validation set is split off
    the remaining training data by segment, so no validation segment is seen
    during training and the test data never influences the search.

    Suitable for use as a multiprocessing worker (top-level, picklable).
    """
    df = read_pickle(Path(path_str))
    train_df, _ = consistent_train_test_split(df, test_frac=test_fraction, random_state=seed_int)
    # A different seed for the validation split: with the test seed, every remaining segment hashes above
    # test_fraction and the validation set would be empty.
    train_df, val_df = consistent_train_test_split(train_df, test_frac=test_fraction, random_state=seed_int + 1)

    train_ds = ColocSequentialEvalDataset(train_df)
    val_ds = ColocSequentialEvalDataset(val_df)

    train_meta = train_df[["id", "segment_id", "source_file"]].reset_index(drop=True)
    return (
        train_ds.signals, train_ds.ids, train_ds.positive_pairs,
        val_ds.signals, val_ds.ids, val_ds.positive_pairs,
        train_meta,
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


_recording_samplers: dict[tuple, object] = {}
_recording_sampler_lock = threading.Lock()


def _get_recording_sampler(path_str: str, test_fraction: float, seed_int: int, device_str: str):
    """Recording-window sampler of a dataset's training split, shared across parallel trials.

    Built on first use per (dataset, device): it keeps a padded copy of the training recordings on
    the GPU, so it is not prewarmed for every dataset of the search space. The lock keeps parallel
    trials from building the same sampler twice.
    """
    key = (path_str, test_fraction, seed_int, device_str)
    with _recording_sampler_lock:
        sampler = _recording_samplers.get(key)
        if sampler is None:
            train_signals = _get_gpu_split(path_str, test_fraction, seed_int, device_str)[0]
            train_meta = _load_split_arrays(path_str, test_fraction, seed_int)[6]
            sampler = RecordingWindowSampler(train_signals, train_meta)
            _recording_samplers[key] = sampler
        return sampler


def _conv1d_out_length(length, kernel_size, stride, padding, dilation=1):
    return ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _pool1d_out_length(length, kernel_size, stride, padding=0, dilation=1):
    return ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _hparams_from_trial(trial) -> dict:
    """Format one Optuna trial's parameters as the model hyperparameter dict.

    Shared by the search and by ``fast-evaluation`` so both derive the same
    YAML from the same trial.
    """
    h = trial.params
    fc_hidden_dims = [h[f"fc_dim_{i}"] for i in range(h["fc_layers"])]
    return {
        "input_channels": 2,
        "embedding_dim": h["embedding_dim"],
        "num_conv_layers": h["num_conv_layers"],
        "base_channels": h.get("base_channels"),
        "kernel_size": h.get("kernel_size"),
        "stride": h.get("stride"),
        "pool_kernel": h.get("pool_kernel"),
        "fc_hidden_dims": fc_hidden_dims,
        "fc_dropout": h["fc_dropout"],

        # Dataset & Optimization parameters
        "hz": h["hz"],
        "window_size_seconds": h["window_size_seconds"],
        "chunk_size_seconds": h["chunk_size_seconds"],
        "train_ride_start_seconds": h["train_ride_start_seconds"],
        "lr": h["lr"],
        "batch_size": BATCH_SIZE,
        "weight_decay": h["weight_decay"],
    }


def _best_trial_by_mcc(study):
    """The Pareto-front trial with the highest MCC — the search's own choice."""
    best_trials = study.best_trials
    if not best_trials:
        return None, 0
    return max(best_trials, key=lambda t: t.values[0]), len(best_trials)


def _dataset_config_ranking(study, top_k: int) -> list[tuple[float, int, dict]]:
    """Dataset configurations ranked by the mean score of their `top_k` best completed trials.

    Returns (score, number of completed trials, dataset params) per configuration, best first.
    Averaging the top trials instead of taking the single best one keeps a lucky trial on a
    rarely sampled dataset from deciding which dataset phase 2 is run on.
    """
    by_config: dict[tuple, list[float]] = {}
    for t in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]):
        if not all(p in t.params for p in DATASET_PARAMS):
            continue
        by_config.setdefault(tuple(t.params[p] for p in DATASET_PARAMS), []).append(t.value)

    ranking = [
        (float(np.mean(sorted(values, reverse=True)[:top_k])), len(values), dict(zip(DATASET_PARAMS, config)))
        for config, values in by_config.items()
    ]
    ranking.sort(key=lambda r: r[0], reverse=True)
    return ranking


def _choose_dataset_config(study, top_k: int) -> dict | None:
    """Pick the dataset for phase 2: best top-`top_k` mean among configs with at least `top_k` trials."""
    ranking = _dataset_config_ranking(study, top_k)
    if not ranking:
        return None
    print(f"\nDataset configurations after phase 1 (mean of the top {top_k} trials each):")
    for score, n, config in ranking[:10]:
        print(f"  {score:.4f}  n={n:4d}  first{config['train_ride_start_seconds']}_"
              f"{config['chunk_size_seconds']}s_window{config['window_size_seconds']}_{config['hz']}Hz")
    well_sampled = [r for r in ranking if r[1] >= top_k]
    return (well_sampled or ranking)[0][2]


def _best_trial_on_dataset(study, dataset_config: dict | None):
    """Best completed trial, restricted to `dataset_config` if the dataset was frozen."""
    completed = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    if dataset_config is not None:
        completed = [t for t in completed if all(t.params.get(p) == v for p, v in dataset_config.items())]
    return max(completed, key=lambda t: t.value) if completed else None


@app.command("optimize")
def optimize_simple_cnn(
        base_data_dir: Path = typer.Option(
            "/path/to/colocation/datasets/",
            help="Base path for all datasets"),
        test_fraction: float = typer.Option(0.3, help="Fraction of data to use for testing"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducibility (converted to int)"),
        n_trials: int = typer.Option(1_000, help="Number of Optuna trials"),
        timeout: int = typer.Option(0, help="Timeout in seconds (0 = no timeout)"),
        max_epochs: int = typer.Option(200, help="Max epochs per trial"),
        num_gpus: int = typer.Option(1, help="Number of GPUs to parallelize trials across (0 for CPU)"),
        n_jobs: int = typer.Option(1,
                                   help="Number of concurrent Optuna trials (threads). On a single GPU, 2-4 typically saturates it."),
        hparams_output: Path = typer.Option(
            Path("./results/model_hparams.yaml"),
            help="Path to save the best hyperparameters YAML file.",
        ),
        db_output: Path = typer.Option(
            Path("hyperparam_search.db"),
            help="Path to save the Optuna SQLite database.",
        ),
        min_val_positives: int = typer.Option(
            500,
            help="Exclude datasets whose validation split has fewer positive pairs than this "
                 "(MCC on small validation sets is too noisy to compare trials).",
        ),
        dataset_search_trials: int = typer.Option(
            500,
            help="Phase 1: number of completed trials in which the dataset parameters are searched "
                 "together with the model. After that the dataset is frozen to the best configuration "
                 "and only the model/optimizer parameters are searched (phase 2, same study and DB). "
                 "0 = never freeze the dataset.",
        ),
        dataset_top_k: int = typer.Option(
            5, help="Phase 2 dataset = configuration with the best mean score of its top-k trials."),
        hz_choices: list[int] = typer.Option(
            HZ_CHOICES, help="Sampling rates to search. Repeatable; must stay the same when resuming a DB."),
        window_choices: list[int] = typer.Option(
            WINDOW_CHOICES, help="Rolling windows to search. Repeatable; must stay the same when resuming a DB."),
        chunk_choices: list[int] = typer.Option(
            CHUNK_CHOICES, help="Chunk sizes (s) to search. Repeatable; must stay the same when resuming a DB."),
        start_choices: list[int] = typer.Option(
            START_CHOICES,
            help="Train-ride start offsets (s) to search. Repeatable; must stay the same when resuming a DB."),
        prune_below: float = typer.Option(
            0.1,
            help="Prune a trial whose mean validation MCC over the last epochs collapses below this value "
                 "after it had reached --prune-after-peak. 0 = no pruning."),
        prune_after_peak: float = typer.Option(
            0.3,
            help="A trial can only be pruned once its mean validation MCC over the last epochs has reached "
                 "this value, so trials that are slow to start learning are kept."),
        population_size: int = typer.Option(
            50,
            help="Population size of the NSGA-II sampler. A small population settles on one dataset and "
                 "architecture after the first generations."),
):
    if population_size < 2:
        raise typer.BadParameter("population_size must be >= 2")

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

    _preload_all_datasets(base_data_dir, test_fraction, seed_int, n_jobs, min_val_positives,
                          start_choices, chunk_choices, window_choices, hz_choices)

    if device == "cuda":
        _prewarm_gpu_cache(test_fraction, seed_int, "cuda:0")

    # 1 = dataset and model searched together, 2 = dataset frozen (set below before each optimize call).
    search_phase = {"phase": 1}

    def objective(trial: optuna.Trial):
        if device == 'cuda':
            gpu_id = trial.number % max(1, num_gpus)
            torch.cuda.set_device(gpu_id)
            trial_device = f"cuda:{gpu_id}"
        else:
            trial_device = 'cpu'

        trial.set_user_attr("search_phase", search_phase["phase"])

        # --- 1. Dataset Configuration Hyperparameters ---
        hz = trial.suggest_categorical("hz", hz_choices)
        window_size_seconds = trial.suggest_categorical("window_size_seconds", window_choices)
        chunk_size_seconds = trial.suggest_categorical("chunk_size_seconds", chunk_choices)
        train_ride_start_seconds = trial.suggest_categorical("train_ride_start_seconds", start_choices)

        target_len = int(chunk_size_seconds * hz)

        full_path = base_data_dir / f"all_coloc_first{train_ride_start_seconds}_{chunk_size_seconds}s_window{window_size_seconds}_{hz}Hz.pkl"

        if not full_path.exists():
            print(f"Skipping trial: Dataset {full_path} not found.")
            raise optuna.TrialPruned()

        if str(full_path) in _skipped_datasets:
            print(f"Skipping trial: Dataset {full_path} has only "
                  f"{_skipped_datasets[str(full_path)]} validation positive pairs (< {min_val_positives}).")
            raise optuna.TrialPruned()

        # --- 2. Load Data (GPU tensors shared across parallel trials) ---
        (
            train_signals, train_ids, train_pos_pairs,
            test_signals, test_pairs, test_labels,
        ) = _get_gpu_split(str(full_path), test_fraction, seed_int, str(trial_device))

        # --- 3. Model Architecture Hyperparameters ---
        num_conv_layers = trial.suggest_int("num_conv_layers", 0, 6)
        if num_conv_layers > 0:
            base_channels = trial.suggest_categorical("base_channels", [4, 8, 16, 32])
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
        batch_size = BATCH_SIZE

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

        mcc_history = []
        peak_score = 0.0

        window_sampler = _get_recording_sampler(str(full_path), test_fraction, seed_int, str(trial_device))

        for epoch in trange(max_epochs, leave=False):
            train_one_epoch_gpu(
                model, window_sampler, int(train_pos_pairs.shape[0]),
                batch_size, criterion, optimizer, scheduler, trial_device,
            )

            model.eval()
            metrics = evaluate(
                model, test_signals, test_pairs, test_labels, batch_size, threshold=0.0,
            )
            model.train()

            acc = metrics["acc"]
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1"]
            prevalence = metrics["prevalence"]
            specificity = metrics["specificity"]
            npv = metrics["npv"]
            mcc = metrics["mcc"]
            tp = metrics["tp"]
            fp = metrics["fp"]
            tn = metrics["tn"]
            fn = metrics["fn"]
            total = metrics["total"]

            mcc_history.append(float(mcc))
            trial.report(float(mcc), step=epoch)

            score = float(np.mean(mcc_history[-SCORE_LAST_N_EPOCHS:]))

            if epoch + 1 >= SCORE_LAST_N_EPOCHS:
                peak_score = max(peak_score, score)
                if 0 < prune_below and score < prune_below and peak_score >= prune_after_peak:
                    trial.set_user_attr("best_mcc", float(max(mcc_history)))
                    trial.set_user_attr("stopped_epoch", len(mcc_history))
                    trial.set_user_attr("pruned_reason", f"collapsed: peak {peak_score:.3f} -> {score:.3f}")
                    print(f"Pruning trial {trial.number} at epoch {epoch + 1}: mean mcc collapsed "
                          f"from {peak_score:.4f} to {score:.4f}")
                    raise optuna.TrialPruned()

            # Early stop if target reached
            if score >= 0.9 and epoch + 1 >= SCORE_LAST_N_EPOCHS:
                print(f"Early stopping at epoch {epoch + 1} with mean mcc {score:.4f}")
                save_model_dir = Path("optuna_models")
                save_model_dir.mkdir(exist_ok=True)
                model_save_path = save_model_dir / f"trial_{trial.number}_model_{score:.4f}.pth"
                torch.save(model.state_dict(), model_save_path)
                break

        score = float(np.mean(mcc_history[-SCORE_LAST_N_EPOCHS:]))
        trial.set_user_attr("best_mcc", float(max(mcc_history)))
        trial.set_user_attr("stopped_epoch", len(mcc_history))
        print(f"Trial {trial.number} completed with mean mcc (last {SCORE_LAST_N_EPOCHS} epochs): {score:.4f}")

        return score

    sampler = optuna.samplers.NSGAIISampler(
        population_size=population_size,
        mutation_prob=0.1,
        crossover_prob=0.9,
        seed=seed_int)

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_output}",
        engine_kwargs={"connect_args": {"timeout": SQLITE_LOCK_TIMEOUT_SECONDS}},
    )

    study = optuna.create_study(
        study_name="colocoation_cnn_param_search",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler
    )

    completed_states = (optuna.trial.TrialState.COMPLETE,)

    def _n_completed() -> int:
        return len(study.get_trials(deepcopy=False, states=completed_states))

    start_time = time.monotonic()
    trials_before = len(study.trials)

    def _budget() -> tuple[int, float | None]:
        """Trials and seconds left of this invocation's `n_trials` / `timeout`."""
        trials_left = n_trials - (len(study.trials) - trials_before)
        seconds_left = None if timeout <= 0 else timeout - (time.monotonic() - start_time)
        return trials_left, seconds_left

    fixed_dataset = study.user_attrs.get("fixed_dataset")

    # --- Phase 1: dataset and model parameters together ---
    if fixed_dataset is None and (dataset_search_trials <= 0 or _n_completed() < dataset_search_trials):
        callbacks = []
        if dataset_search_trials > 0:
            print(f"\nPhase 1: searching dataset + model until {dataset_search_trials} trials are completed "
                  f"({_n_completed()} so far).")
            callbacks.append(optuna.study.MaxTrialsCallback(dataset_search_trials, states=completed_states))
        search_phase["phase"] = 1
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=None if timeout <= 0 else timeout,
            n_jobs=n_jobs,
            callbacks=callbacks,
            catch=CAUGHT_TRIAL_ERRORS,
        )

    # --- Phase 2: dataset frozen, model/optimizer parameters only ---
    if fixed_dataset is None and 0 < dataset_search_trials <= _n_completed():
        fixed_dataset = _choose_dataset_config(study, dataset_top_k)
        if fixed_dataset is not None:
            study.set_user_attr("fixed_dataset", fixed_dataset)

    if fixed_dataset is not None:
        trials_left, seconds_left = _budget()
        if trials_left > 0 and (seconds_left is None or seconds_left > 0):
            print(f"\nPhase 2: dataset frozen to {fixed_dataset}; searching model parameters only "
                  f"({trials_left} trials left).")
            study.sampler = optuna.samplers.PartialFixedSampler(fixed_dataset, sampler)
            search_phase["phase"] = 2
            study.optimize(
                objective,
                n_trials=trials_left,
                timeout=seconds_left,
                n_jobs=n_jobs,
                catch=CAUGHT_TRIAL_ERRORS,
            )

    best_trial = _best_trial_on_dataset(study, fixed_dataset)
    if best_trial is not None:
        print(f"\nBest trial (ID: {best_trial.number}):")
        print(
            f"  mean MCC (last {SCORE_LAST_N_EPOCHS} epochs): {best_trial.value:.4f} | "
            f"best MCC: {best_trial.user_attrs.get('best_mcc', float('nan')):.4f} | "
            f"stopped_epoch: {best_trial.user_attrs.get('stopped_epoch')}")

        hparams_dict = _hparams_from_trial(best_trial)

        results_dir = hparams_output.parent
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(hparams_output, "w", encoding="utf-8") as f:
            yaml.dump(hparams_dict, f, sort_keys=False)

        print(f"Saved best hyperparameters to {hparams_output}")


if __name__ == "__main__":
    app()
