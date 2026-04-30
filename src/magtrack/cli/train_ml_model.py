import copy
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from sklearn.metrics import auc, roc_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from magtrack.utils.coloc_dataset import ColocSequentialEvalDataset
from magtrack.utils.evaluation import train_test_split
from magtrack.utils.loader import read_pickle, get_metadata_from_dataset
from magtrack.utils.ml import (
    build_eval_pairs,
    evaluate,
    sample_negatives_gpu,
)
from magtrack.utils.model import ColocationCNN
from magtrack.utils.utils import resolve_seed

app = typer.Typer(help="Train the ML Model for magtrack.")


def _resolve_conv_hparams(h: dict) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    num_conv_layers = int(h["num_conv_layers"])
    if num_conv_layers <= 0:
        return [], [], [], [], [], []

    if h.get("base_channels") is not None:
        base_channels = int(h["base_channels"])
        conv_channels = [base_channels * (2 ** i) for i in range(num_conv_layers)]
    else:
        conv_channels = list(map(int, h["conv_channels"]))

    if h.get("kernel_size") is not None:
        kernel_sizes = [int(h["kernel_size"]) for _ in range(num_conv_layers)]
    else:
        kernel_sizes = list(map(int, h["kernel_sizes"]))

    if h.get("stride") is not None:
        strides = [int(h["stride"]) for _ in range(num_conv_layers)]
    else:
        strides = list(map(int, h["strides"]))

    if h.get("pool_kernel") is not None:
        pool_kernel = int(h["pool_kernel"])
        pool_kernel_sizes = [pool_kernel for _ in range(num_conv_layers)]
        pool_strides = [pool_kernel for _ in range(num_conv_layers)]
    else:
        pool_kernel_sizes = list(map(int, h["pool_kernel_sizes"]))
        pool_strides = list(map(int, h["pool_strides"]))

    if h.get("paddings") is not None:
        paddings = list(map(int, h["paddings"]))
    else:
        paddings = [k // 2 for k in kernel_sizes]

    return conv_channels, kernel_sizes, strides, paddings, pool_kernel_sizes, pool_strides


def _to_tb_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        try:
            return value.item()
        except Exception:
            return value.detach().cpu().numpy()
    if isinstance(value, (int, float, str, bool)):
        return value
    # add_hparams only accepts int/float/str/bool/Tensor — coerce lists, None,
    # dicts, etc. to a string representation.
    return str(value)


def _build_model_from_hparams(hparams_path: Path) -> tuple[ColocationCNN, dict]:
    if not hparams_path.exists():
        raise FileNotFoundError(f"Model hyperparameters file not found: {hparams_path}")
    with open(hparams_path, encoding="utf-8") as fh:
        h = yaml.safe_load(fh) or {}

    target_len = 30*10#int(h.get("chunk_size_seconds", 20) * h.get("hz", 40))
    conv_channels, kernel_sizes, strides, paddings, pool_kernel_sizes, pool_strides = _resolve_conv_hparams(h)

    model = ColocationCNN(
        input_channels=2,
        signal_length=int(target_len),
        embedding_dim=int(h["embedding_dim"]),
        num_conv_layers=int(h["num_conv_layers"]),
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        pool_kernel_sizes=pool_kernel_sizes,
        pool_strides=pool_strides,
        fc_hidden_dims=list(map(int, h["fc_hidden_dims"])),
        fc_dropout=float(h["fc_dropout"]),
    )
    return model, h


def log_class_distribution(writer, labels, dataset_name, step=0):
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(unique, counts)
    writer.add_figure(f"Dataset/{dataset_name}_class_dist", fig, global_step=step)
    plt.close(fig)


def log_batch_labels(writer, batch_labels, phase, step):
    writer.add_histogram(f"Batch/{phase}_labels", batch_labels, global_step=step)


def log_signal_pairs(writer, x1, x2, labels, step, num_examples=4):
    num_examples = min(num_examples, x1.size(0))
    fig, axes = plt.subplots(num_examples, 1, figsize=(8, 2.5 * num_examples))
    if num_examples == 1:
        axes = [axes]
    for i in range(num_examples):
        axes[i].plot(x1[i].detach().cpu().numpy().flatten(), alpha=0.8)
        axes[i].plot(x2[i].detach().cpu().numpy().flatten(), alpha=0.6)
    writer.add_figure("Train/signal_pairs", fig, global_step=step)
    plt.close(fig)


def _log_roc_from_arrays(writer, labels_np, scores_np, epoch, tag="Test/ROC"):
    fpr, tpr, _ = roc_curve(labels_np, scores_np)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    writer.add_figure(tag, fig, global_step=epoch)
    writer.add_scalar(f"{tag}_auc", float(roc_auc), epoch)
    plt.close(fig)
    return float(roc_auc)


def _build_sequential_arrays(df):
    dataset = ColocSequentialEvalDataset(df)
    return dataset.signals, dataset.ids, dataset.positive_pairs


def _build_split_arrays(train_df, test_df, workers: int):
    if workers <= 1:
        train_arrays = _build_sequential_arrays(train_df)
        test_arrays = _build_sequential_arrays(test_df)
        return train_arrays, test_arrays

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        train_future = executor.submit(_build_sequential_arrays, train_df)
        test_future = executor.submit(_build_sequential_arrays, test_df)
        train_arrays = train_future.result()
        test_arrays = test_future.result()
    return train_arrays, test_arrays


def _get_gpu_split_from_dfs(train_df, test_df, device: str, seed_int: int, preload_workers: int):
    (train_signals_np, train_ids_np, train_pos_pairs_np), (test_signals_np, test_ids_np, test_pos_pairs_np) = (
        _build_split_arrays(train_df, test_df, preload_workers)
    )

    train_signals = torch.from_numpy(train_signals_np).to(device)
    train_ids = torch.from_numpy(train_ids_np).to(device)
    train_pos_pairs = torch.from_numpy(train_pos_pairs_np).to(device)

    test_signals = torch.from_numpy(test_signals_np).to(device)
    test_ids = torch.from_numpy(test_ids_np).to(device)
    test_pos_pairs = torch.from_numpy(test_pos_pairs_np).to(device)

    torch.manual_seed(seed_int)
    test_pairs, test_labels = build_eval_pairs(test_pos_pairs, test_ids, test_signals.shape[0])

    return (
        train_signals,
        train_ids,
        train_pos_pairs,
        test_signals,
        test_pairs,
        test_labels,
    )


def train(
        model,
        train_signals,
        train_ids,
        train_pos_pairs,
        test_signals,
        test_pairs,
        test_labels,
        batch_size,
        criterion,
        optimizer,
        writer,
        epochs=1_000,
        device="cuda",
        log_signal_every=10,
        log_batch_loss_every=10,
        eval_every=5,
        roc_fig_every=25,
):
    model.to(device)
    model.train()

    best_mcc = 0.0
    best_model = copy.deepcopy(model.state_dict())

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=1000,
    )

    global_step = 0
    smoothing = 0.1

    n_train = train_signals.shape[0]
    num_pos = train_pos_pairs.shape[0]
    total_pairs = num_pos * 2  # neg_pos_ratio = 1
    last_metrics = None

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        total_loss = torch.zeros((), device=device)
        n_batches = 0

        # Resample negatives once per epoch on the GPU.
        neg_i, neg_j = sample_negatives_gpu(n_train, num_pos, train_ids)
        # Build full pair table for this epoch: (positives | negatives), (2*P, 2)
        epoch_i = torch.cat([train_pos_pairs[:, 0], neg_i])
        epoch_j = torch.cat([train_pos_pairs[:, 1], neg_j])
        epoch_labels = torch.cat([
            torch.zeros(num_pos, dtype=torch.long, device=device),
            torch.ones(num_pos, dtype=torch.long, device=device),
        ])
        perm = torch.randperm(total_pairs, device=device)

        for batch_idx, start in enumerate(range(0, total_pairs, batch_size)):
            sel = perm[start:start + batch_size]
            i_idx = epoch_i[sel]
            j_idx = epoch_j[sel]
            label = epoch_labels[sel]

            optimizer.zero_grad(set_to_none=True)
            x1 = train_signals[i_idx]
            x2 = train_signals[j_idx]

            x1 = x1 + torch.normal(mean=0., std=1e-1, size=x1.shape, device=device)
            x2 = x2 + torch.normal(mean=0., std=1e-1, size=x2.shape, device=device)

            label_float = label.float().unsqueeze(1)
            label_float = label_float * (1.0 - smoothing) + (0.5 * smoothing)

            predicted_label = model(x1, x2)
            loss = criterion(predicted_label, label_float)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.detach()
            n_batches += 1

            if global_step % max(1, int(log_batch_loss_every)) == 0:
                writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)

            if batch_idx == 0:
                log_batch_labels(writer, label, phase="Train", step=global_step)
                if epoch % max(1, int(log_signal_every)) == 0:
                    log_signal_pairs(writer, x1, x2, label, step=epoch)

            global_step += 1

        epoch_loss = (total_loss / max(n_batches, 1)).item()
        epoch_losses.append(epoch_loss)
        losses.append(epoch_loss)
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)

        do_eval = (epoch % max(1, int(eval_every)) == 0) or (epoch == epochs - 1)
        if do_eval:
            model.eval()
            need_scores = (epoch % max(1, int(roc_fig_every)) == 0) or (epoch == epochs - 1)
            if need_scores:
                metrics, labels_np, scores_np = evaluate(
                    model, test_signals, test_pairs, test_labels, batch_size,
                    threshold=0.0, return_scores=True,
                )
                _log_roc_from_arrays(writer, labels_np, scores_np, epoch)
            else:
                metrics = evaluate(
                    model, test_signals, test_pairs, test_labels, batch_size,
                    threshold=0.0, return_scores=False,
                )
            (
                test_accuracy,
                precision,
                recall,
                f1,
                prevalence,
                specificity,
                negative_predictive_value,
                mcc,
            ) = metrics
            last_metrics = metrics
            test_mcc = mcc

            writer.add_scalar("Test/Accuracy", test_accuracy, epoch)
            writer.add_scalar("Test/Precision", precision, epoch)
            writer.add_scalar("Test/Recall", recall, epoch)
            writer.add_scalar("Test/F1", f1, epoch)
            writer.add_scalar("Test/Prevalence", prevalence, epoch)
            writer.add_scalar("Test/Specificity", specificity, epoch)
            writer.add_scalar("Test/Negative_Predictive_value", negative_predictive_value, epoch)
            writer.add_scalar("Test/MCC", mcc, epoch)
            model.train()
        else:
            if last_metrics is None:
                continue
            (
                test_accuracy,
                precision,
                recall,
                f1,
                prevalence,
                specificity,
                negative_predictive_value,
                mcc,
            ) = last_metrics

        if test_mcc > best_mcc:
            best_mcc = test_mcc
            torch.save(model.state_dict(), "model_checkpoints/coloc_net_best_model.pth")
            best_model = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(), f"model_checkpoints/coloc_net_epoch_{epoch + 1}.pth")

        model.train()
        if test_mcc > 0.9:
            print("Early stopping at epoch", epoch + 1, "with MCC:", test_mcc)
            break

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
            print(
                "Test Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1_val}, "
                "Prevalence: {prev}, Specificity: {spec}, NPV: {npv}, MCC: {mcc_val}".format(
                    acc=test_accuracy,
                    prec=precision,
                    rec=recall,
                    f1_val=f1,
                    prev=prevalence,
                    spec=specificity,
                    npv=negative_predictive_value,
                    mcc_val=test_mcc,
                )
            )

    return losses, epoch_losses, best_model


@app.command("main")
def train_ml(
        dataset_path: Path = typer.Argument(..., help="One path to ml dataset (pkl)"),
        test_fraction: float = typer.Option(0.3, help="Fraction of data to use for testing"),
        log_signal_every: int = typer.Option(10, help="Log sample signal plots every N epochs"),
        log_batch_loss_every: int = typer.Option(10, help="Log batch loss every N global steps"),
        save: bool = typer.Option(True, help="Whether to save final model state_dict"),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducibility (converted to int)"),
        epochs: int = typer.Option(1_000, help="Maximum number of training epochs"),
        batch_size: int = typer.Option(4096, help="Batch size for training"),
        learning_rate: float = typer.Option(0.001, help="Learning rate for training"),
        weight_decay: float = typer.Option(1e-5, help="Weight decay for training"),
        eval_every: int = typer.Option(10, help="Run test-set evaluation every N epochs"),
        roc_fig_every: int = typer.Option(50, help="Log ROC figure every N epochs (AUC scalar always logged on eval)"),
        hparams_path: Path = typer.Option('./coloc_model/model_hparams.yaml',
                                          help="Optional path to model hyperparameters YAML file (overrides defaults)"),
        preload_workers: int = typer.Option(
            0,
            help=(
                "Number of worker processes to build train/test arrays in parallel. "
                "Use 0 or 1 to disable."
            ),
        ),
):
    hparams_path = Path(hparams_path)
    if hparams_path.exists():
        with open(hparams_path, encoding="utf-8") as fh:
            h = yaml.safe_load(fh) or {}
        batch_size = h.get("batch_size", batch_size)
        learning_rate = h.get("lr", learning_rate)
        weight_decay = h.get("weight_decay", weight_decay)

    # Training/optimizer settings (selected from prior tuning).
    epochs = epochs
    batch_size = batch_size
    learning_rate = learning_rate
    weight_decay = weight_decay

    # Logging

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"runs/exp_{timestamp}_lr{learning_rate:.5f}_bs{batch_size}"
    os.makedirs(run_name, exist_ok=True)
    writer = SummaryWriter(run_name)
    logger.info("Logging TensorBoard to: {}", run_name)

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
    n_classes = len(uniques)

    logger.info("Loaded {} samples, {} classes.", len(data), n_classes)
    logger.info("Using chunk_size_seconds={}s", chunk_size_seconds)

    train_df, test_df = train_test_split(
        data,
        test_frac=test_fraction,
        random_state=seed_int,
    )

    logger.info("Total classes: {}, test classes: {}", n_classes, test_df["id"].nunique())
    logger.info("Samples -> train: {}, test: {}", len(train_df), len(test_df))

    train_ids = set(train_df["id"].unique())
    test_ids = set(test_df["id"].unique())
    overlap_ids = train_ids.intersection(test_ids)
    if overlap_ids:
        logger.warning("Found {} overlapping ids between train and test (possible leakage).", len(overlap_ids))
    else:
        logger.info("No overlapping ids found between train and test.")

    os.makedirs("model_checkpoints", exist_ok=True)

    log_class_distribution(writer, train_df["class"].to_numpy(), "Train")
    log_class_distribution(writer, test_df["class"].to_numpy(), "Test")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Len train dataset: {}", len(train_df))
    logger.info("Len test dataset: {}", len(test_df))

    # Move all tensors to device once. Frees the DataLoader / worker overhead.
    (
        train_signals,
        train_ids,
        train_pos_pairs,
        test_signals,
        test_pairs,
        test_labels,
    ) = _get_gpu_split_from_dfs(train_df, test_df, device, seed_int, preload_workers)

    if device == "cuda":
        torch.cuda.synchronize()
        vram_mb = (
                          train_signals.numel() * train_signals.element_size()
                          + test_signals.numel() * test_signals.element_size()
                  ) / (1024 ** 2)
        logger.info("Signals on GPU: ~{:.1f} MB", vram_mb)

    model, model_hparams = _build_model_from_hparams(hparams_path)
    logger.info(f"Loaded model hyperparameters from {hparams_path}")
    logger.info("Starting training...")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    _, _, trained_model_state = train(
        model,
        train_signals,
        train_ids,
        train_pos_pairs,
        test_signals,
        test_pairs,
        test_labels,
        batch_size,
        criterion,
        optimizer,
        writer,
        epochs=epochs,
        device=device,
        log_signal_every=log_signal_every,
        log_batch_loss_every=log_batch_loss_every,
        eval_every=eval_every,
        roc_fig_every=roc_fig_every,
    )

    if save:
        torch.save(trained_model_state, "coloc_model/colocation_net.pth")
        logger.info("Saved final model to coloc_model/colocation_net.pth")

    eval_model, _ = _build_model_from_hparams(hparams_path)
    eval_model = eval_model.to(device)
    eval_model.load_state_dict(torch.load("coloc_model/colocation_net.pth", map_location=device))
    eval_model.eval()

    acc, precision, recall, f1, prevalence, specificity, negative_predictive_value, mcc = evaluate(
        eval_model,
        test_signals,
        test_pairs,
        test_labels,
        batch_size,
        threshold=0.0,
    )  # type: ignore[misc]

    logger.info("Final metrics: accuracy={}, precision={}, recall={}, f1={}, mcc={}", acc, precision, recall, f1, mcc)

    hparam_dict = {
        "chunk_size_seconds": chunk_size_seconds,
        "lr": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "test_fraction": test_fraction,
        **model_hparams,
    }

    metric_dict = {
        "hparam/final_accuracy": acc,
        "hparam/final_f1": f1,
        "hparam/final_precision": precision,
        "hparam/final_recall": recall,
        "hparam/final_prevalence": prevalence,
        "hparam/final_specificity": specificity,
        "hparam/final_negative_predictive_value": negative_predictive_value,
        "hparam/final_mcc": mcc,
    }

    writer.add_hparams(
        {k: _to_tb_scalar(v) for k, v in hparam_dict.items()},
        {k: _to_tb_scalar(v) for k, v in metric_dict.items()},
    )
    writer.close()


if __name__ == "__main__":
    app()
