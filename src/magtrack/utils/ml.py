"""Shared ML training/evaluation helpers used by both the trainer and the
hyperparameter search."""

from __future__ import annotations

import numpy as np
import torch
from time import perf_counter
from tqdm import trange

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def seed_worker(worker_id):
    """DataLoader worker init for deterministic numpy/dataset RNG seeding."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(int(worker_seed))
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "rng"):
        worker_info.dataset.rng = np.random.default_rng(int(worker_seed))


def sample_negatives_gpu(n: int, num: int, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `num` index pairs (i, j) with ids[i] != ids[j]. All on `ids.device`."""
    device = ids.device
    if num == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    i = torch.randint(0, n, (num,), device=device)
    j = torch.randint(0, n, (num,), device=device)
    bad = ids[i] == ids[j]
    for _ in range(32):
        if not bool(bad.any()):
            break
        m = int(bad.sum().item())
        j_new = torch.randint(0, n, (m,), device=device)
        j = j.masked_scatter(bad, j_new)
        bad = ids[i] == ids[j]
    return i, j


def build_eval_pairs(
        positive_pairs: torch.Tensor, ids: torch.Tensor, n_signals: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate positive pairs with once-sampled negatives into a fixed eval set.

    Returns (pairs (P+N, 2), labels (P+N,)). Label 0 = positive, 1 = negative.
    """
    device = positive_pairs.device
    num_pos = positive_pairs.shape[0]
    neg_i, neg_j = sample_negatives_gpu(n_signals, num_pos, ids)
    pairs = torch.stack(
        [
            torch.cat([positive_pairs[:, 0], neg_i]),
            torch.cat([positive_pairs[:, 1], neg_j]),
        ],
        dim=1,
    )
    labels = torch.cat(
        [
            torch.zeros(num_pos, dtype=torch.long, device=device),
            torch.ones(neg_i.shape[0], dtype=torch.long, device=device),
        ]
    )
    return pairs, labels


def train_one_epoch_gpu(
        model,
        signals: torch.Tensor,
        ids: torch.Tensor,
        positive_pairs: torch.Tensor,
        batch_size: int,
        criterion,
        optimizer,
        scheduler,
        device,
        smoothing: float = 0.1,
        noise_std: float = 1e-1,
) -> float:
    """One training epoch using GPU-resident signals and pair indices.

    Negatives are sampled once per epoch on `device`; batches are formed by
    index-permuting the (positives | negatives) pair table. Returns mean loss.
    """
    model.train()
    n = signals.shape[0]
    num_pos = positive_pairs.shape[0]
    total = num_pos * 2  # 1:1 negative ratio

    neg_i, neg_j = sample_negatives_gpu(n, num_pos, ids)
    epoch_i = torch.cat([positive_pairs[:, 0], neg_i])
    epoch_j = torch.cat([positive_pairs[:, 1], neg_j])
    epoch_labels = torch.cat([
        torch.zeros(num_pos, dtype=torch.long, device=device),
        torch.ones(num_pos, dtype=torch.long, device=device),
    ])
    perm = torch.randperm(total, device=device)

    total_loss = torch.zeros((), device=device)
    n_batches = 0

    for start in range(0, total, batch_size):
        sel = perm[start:start + batch_size]
        x1 = signals[epoch_i[sel]]
        x2 = signals[epoch_j[sel]]
        label = epoch_labels[sel]

        if noise_std > 0:
            x1 = x1 + torch.normal(0., noise_std, size=x1.shape, device=device)
            x2 = x2 + torch.normal(0., noise_std, size=x2.shape, device=device)

        label_float = label.float().unsqueeze(1)
        label_float = label_float * (1.0 - smoothing) + (0.5 * smoothing)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x1, x2)
        loss = criterion(pred, label_float)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.detach()
        n_batches += 1

    return float((total_loss / max(n_batches, 1)).item())


def evaluate(model, signals, pairs, labels, batch_size, threshold=0.0):
    """GPU-resident evaluation.

    signals: (N, L) on device. pairs: (P, 2) long on device. labels: (P,) int on device.
    Returns a metrics dict.
    """
    all_preds: list[int] = []
    with torch.no_grad():
        for start in trange(0, pairs.shape[0], batch_size):
            p = pairs[start:start + batch_size]
            x1 = signals[p[:, 0]]
            x2 = signals[p[:, 1]]
            logits = model(x1, x2)
            preds = (logits > threshold).long().squeeze(1)
            all_preds.extend(preds.cpu().tolist())

    all_preds = np.array(all_preds)
    total = labels.shape[0]
    y_true = np.array([1 if l == 0 else 0 for l in labels.detach().cpu().numpy()])
    y_pred = np.array([1 if p == 0 else 0 for p in all_preds])

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(y_true) > 0 else 0.0

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
    else:
        TN = FP = FN = TP = 0

    prevalence = (TP + FN) / total if total > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    negative_predictive_value = TN / (TN + FN) if (TN + FN) > 0 else 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prevalence": prevalence,
        "specificity": specificity,
        "npv": negative_predictive_value,
        "mcc": mcc,
        "tp": TP,
        "fp": FP,
        "tn": TN,
        "fn": FN,
        "total": total,
    }


def evaluate_with_extras(
        model,
        signals: torch.Tensor,
        pairs: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        threshold: float = 0.0,
        time_per_step: bool = False,
        show_progress: bool = False,
) -> dict:
    """Like `evaluate(...)` but also returns AUROC, TP/FP/TN/FN counts, and
    optionally per-step (per-batch) wall-clock timing.

    Returns a dict containing every metric `evaluate` produces plus:
      - auroc: float (NaN if only one class is present)
      - tp/fp/tn/fn/total: int
      - step_times_s: list[float] (empty unless `time_per_step=True`)
      - labels_np / scores_np: np.ndarray (full label/score arrays)

    `time_per_step=True` synchronizes CUDA before/after each batch so the
    recorded wall-clock reflects real GPU execution time. This serializes the
    work so leave it off for max throughput.
    """
    device = signals.device
    sync_cuda = (device.type == "cuda" and torch.cuda.is_available())

    all_preds: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    step_times: list[float] = []

    was_training = model.training
    model.eval()
    n_pairs = pairs.shape[0]
    starts = range(0, n_pairs, batch_size)
    if show_progress:
        from tqdm import tqdm  # local import keeps the default path import-free
        starts = tqdm(
            starts, total=(n_pairs + batch_size - 1) // batch_size,
            desc="Evaluating", unit="batch", mininterval=0.5,
        )
    with torch.no_grad():
        for start in starts:
            p = pairs[start:start + batch_size]
            x1 = signals[p[:, 0]]
            x2 = signals[p[:, 1]]

            if time_per_step and sync_cuda:
                torch.cuda.synchronize()
            t0 = perf_counter() if time_per_step else 0.0

            logits = model(x1, x2)

            if time_per_step and sync_cuda:
                torch.cuda.synchronize()
            if time_per_step:
                step_times.append(perf_counter() - t0)

            preds = (logits > threshold).long().squeeze(1)
            all_preds.append(preds)
            all_scores.append(torch.sigmoid(logits).squeeze(1))

    if was_training:
        model.train()

    preds_np = torch.cat(all_preds).cpu().numpy() if all_preds else np.empty(0, dtype=np.int64)
    scores_np = torch.cat(all_scores).cpu().numpy() if all_scores else np.array([])
    labels_np = labels.cpu().numpy()
    total = int(labels_np.shape[0])

    y_true = (labels_np == 0).astype(int)
    y_pred = (preds_np == 0).astype(int)

    if total > 0:
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        # AUROC for the "match" class. Score for match = 1 - sigmoid(logit).
        try:
            auroc = float(roc_auc_score(y_true, 1.0 - scores_np))
        except ValueError:
            auroc = float("nan")
    else:
        acc = precision = recall = f1 = mcc = 0.0
        auroc = float("nan")
        tn = fp = fn = tp = 0

    prevalence = (tp + fn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prevalence": prevalence,
        "specificity": specificity,
        "npv": npv,
        "mcc": mcc,
        "auroc": auroc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "step_times_s": step_times,
        "labels_np": labels_np,
        "scores_np": scores_np,
    }
