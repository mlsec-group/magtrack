from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import torch
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


class RecordingWindowSampler:
    """Training pairs of random time-aligned windows cut from whole recordings.

    A recording is all chunks of one device in one segment ((segment_id, source_file)), stitched in
    chunk order. A pair is two recordings plus one offset; both windows are cut at the same offset,
    so they cover the same time span. Positive = two recordings of the same segment_id (every such
    pair equally likely), negative = recordings of different segments. Offsets are not tied to chunk
    boundaries, and windows are one chunk long and only cover chunks present in both recordings.

    A negative reuses a recording and the offset of a positive sample and only replaces the other
    recording by one from another segment, so negatives have the same distribution of ride positions
    as positives (no "where in the ride is this window" shortcut).

    `signals` holds the chunks in the row order of `df.reset_index()`.
    """

    def __init__(self, signals: torch.Tensor, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        device = signals.device
        chunk_len = signals.shape[1]
        chunk_idx = torch.tensor(df["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy(), device=device)
        rec_codes_np, rec_keys = pd.factorize(df["segment_id"].astype(str) + "::" + df["source_file"].astype(str))
        rec_codes = torch.tensor(rec_codes_np, device=device)
        n_rec, n_chunks = len(rec_keys), int(chunk_idx.max()) + 1

        self.chunk_len = chunk_len
        self.recordings = torch.zeros(n_rec, n_chunks, chunk_len, device=device, dtype=signals.dtype)
        self.recordings[rec_codes, chunk_idx] = signals
        self.recordings = self.recordings.view(n_rec, n_chunks * chunk_len)
        self.chunk_valid = torch.zeros(n_rec, n_chunks, device=device, dtype=torch.bool)
        self.chunk_valid[rec_codes, chunk_idx] = True
        self.n_chunks = torch.zeros(n_rec, device=device, dtype=torch.long).scatter_reduce(
            0, rec_codes, chunk_idx + 1, reduce="amax")

        segment_codes = torch.tensor(
            pd.factorize(df.groupby(rec_codes_np)["segment_id"].first())[0], device=device)
        self.segment = segment_codes
        same = (segment_codes.unsqueeze(0) == segment_codes.unsqueeze(1)).triu(diagonal=1)
        self.pos_pairs = torch.nonzero(same)  # (P, 2) recording pairs of the same segment
        if self.pos_pairs.shape[0] == 0:
            raise ValueError("No segment has more than one recording; cannot build positive pairs.")
        self.pos_weights = torch.ones(self.pos_pairs.shape[0], device=device)

    def _offsets(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Random offsets inside the common length, resampled until all covered chunks exist in both."""
        span = torch.minimum(self.n_chunks[a], self.n_chunks[b]) * self.chunk_len - self.chunk_len + 1
        offsets = torch.zeros_like(a)
        todo = torch.ones_like(a, dtype=torch.bool)
        for _ in range(16):
            cand = (torch.rand(a.shape[0], device=a.device) * span).long()
            first, last = cand // self.chunk_len, (cand + self.chunk_len - 1) // self.chunk_len
            ok = (self.chunk_valid[a, first] & self.chunk_valid[a, last]
                  & self.chunk_valid[b, first] & self.chunk_valid[b, last])
            take = todo & ok
            offsets[take] = cand[take]
            todo &= ~ok
            if not bool(todo.any()):
                break
        return offsets  # leftovers keep offset 0: chunk 0 exists in every recording

    def sample_positives(self, n: int):
        sel = self.pos_pairs[torch.multinomial(self.pos_weights, n, replacement=True)]
        a, b = sel[:, 0], sel[:, 1]
        return a, b, self._offsets(a, b)

    def _covers(self, rec: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Whether recording `rec` has data for the whole window starting at `offsets`."""
        first, last = offsets // self.chunk_len, (offsets + self.chunk_len - 1) // self.chunk_len
        n_chunks = self.chunk_valid.shape[1]
        inside = last < n_chunks
        last = last.clamp(max=n_chunks - 1)
        return inside & self.chunk_valid[rec, first] & self.chunk_valid[rec, last]

    def sample_negatives(self, n: int):
        n_rec = self.segment.shape[0]
        device = self.segment.device
        # Recording a and the offset follow the positive distribution (either side of a positive pair);
        # b is any recording of another segment that has data for the same window.
        pa, pb, offsets = self.sample_positives(n)
        a = torch.where(torch.rand(n, device=device) < 0.5, pa, pb)
        b = torch.randint(0, n_rec, (n,), device=device)
        bad = (self.segment[a] == self.segment[b]) | ~self._covers(b, offsets)
        for _ in range(64):
            if not bool(bad.any()):
                break
            b = b.masked_scatter(bad, torch.randint(0, n_rec, (int(bad.sum()),), device=device))
            bad = (self.segment[a] == self.segment[b]) | ~self._covers(b, offsets)
        if bool(bad.any()):
            # Rare leftovers (no long-enough recording found): fall back to an offset inside the common length.
            b = b.masked_scatter(bad, torch.randint(0, n_rec, (int(bad.sum()),), device=device))
            still = self.segment[a] == self.segment[b]
            for _ in range(32):
                if not bool((bad & still).any()):
                    break
                b = b.masked_scatter(bad & still, torch.randint(0, n_rec, (int((bad & still).sum()),), device=device))
                still = self.segment[a] == self.segment[b]
            offsets = torch.where(bad, self._offsets(a, b), offsets)
        return a, b, offsets

    def windows(self, a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor):
        idx = offsets.unsqueeze(1) + torch.arange(self.chunk_len, device=offsets.device).unsqueeze(0)
        return self.recordings[a.unsqueeze(1), idx], self.recordings[b.unsqueeze(1), idx]


def train_one_epoch_gpu(
        model,
        sampler: RecordingWindowSampler,
        num_pos: int,
        batch_size: int,
        criterion,
        optimizer,
        scheduler,
        device,
        smoothing: float = 0.1,
) -> float:
    """One training epoch over `num_pos` positive and `num_pos` negative window pairs from `sampler`.

    Pairs are sampled once per epoch on `device` and the order within each pair is swapped with
    probability 0.5; batches are formed by index-permuting the (positives | negatives) pair table.
    Returns mean loss.
    """
    model.train()
    total = num_pos * 2  # 1:1 negative ratio

    pos_i, pos_j, pos_off = sampler.sample_positives(num_pos)
    neg_i, neg_j, neg_off = sampler.sample_negatives(num_pos)
    epoch_i = torch.cat([pos_i, neg_i])
    epoch_j = torch.cat([pos_j, neg_j])
    epoch_off = torch.cat([pos_off, neg_off])
    epoch_labels = torch.cat([
        torch.zeros(num_pos, dtype=torch.long, device=device),
        torch.ones(num_pos, dtype=torch.long, device=device),
    ])
    swap = torch.rand(total, device=device) < 0.5
    epoch_i, epoch_j = torch.where(swap, epoch_j, epoch_i), torch.where(swap, epoch_i, epoch_j)
    perm = torch.randperm(total, device=device)

    total_loss = torch.zeros((), device=device)
    n_batches = 0

    for start in range(0, total, batch_size):
        sel = perm[start:start + batch_size]
        x1, x2 = sampler.windows(epoch_i[sel], epoch_j[sel], epoch_off[sel])
        label = epoch_labels[sel]

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


def evaluate(model, signals, pairs, labels, batch_size, threshold=0.0, return_scores=False):
    """GPU-resident evaluation.

    signals: (N, L) on device. pairs: (P, 2) long on device. labels: (P,) int on device.
    Returns metrics tuple, optionally with (labels_np, scores_np).
    """
    all_preds: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = [] if return_scores else None  # type: ignore[assignment]
    with torch.no_grad():
        for start in range(0, pairs.shape[0], batch_size):
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

    metrics = {
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

    if return_scores:
        labels_np = labels.cpu().numpy()
        scores_np = torch.cat(all_scores).cpu().numpy() if all_scores else np.array([])
        return metrics, labels_np, scores_np
    else:
        return metrics


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
