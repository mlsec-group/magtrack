from typing import Dict, Tuple

import numpy as np
import pandas as pd


def find_best_threshold_snr(
        scores: np.ndarray,
        y_true: np.ndarray,
        metric: str = "mcc",
        eps: float = 1e-9,
) -> Tuple[float, float]:
    """Find the threshold maximising *metric* under the rule
    ``score >= threshold → predict positive``.

    Expects ``y_true`` as boolean or 0/1 where ``True``/``1`` = positive class.
    Vectorised single-pass sweep: sorts finite scores once and uses cumulative
    sums of the labels to derive ``TP``/``FP`` at every candidate threshold,
    then evaluates MCC or F1 in closed form.  Replaces an ``O(N·K)`` Python
    loop calling ``sklearn`` per candidate with an ``O(N log N)`` numpy pass.

    Candidates are the midpoints between sorted unique finite scores plus a
    boundary at each extreme — the same set produced by
    :func:`compute_threshold_candidates`.

    NaN / inf scores are treated as non-detections (always predicted negative).
    They count toward ``FN`` / ``TN`` via the totals from the full *y_true*.

    Parameters
    ----------
    scores : np.ndarray
        Per-sample score; higher = more positive.  Same length as *y_true*.
    y_true : np.ndarray
        Ground-truth labels (boolean or 0/1), where True/1 = positive.
    metric : {"mcc", "f1"}
        Metric to maximise.
    eps : float
        Boundary offset added at each end of the candidate range.

    Returns
    -------
    (best_threshold, best_metric_value)

    Falls back to ``(1.0, 0.0)`` when no finite scores are present.
    """
    if metric not in ("mcc", "f1"):
        metric = "mcc"

    y_true_bool = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)

    finite_mask = np.isfinite(scores)
    scores_finite = scores[finite_mask]
    y_finite = y_true_bool[finite_mask]

    if scores_finite.size == 0:
        return 1.0, 0.0

    total_pos = int(y_true_bool.sum())
    total_neg = int(y_true_bool.size - total_pos)

    order = np.argsort(scores_finite, kind="mergesort")
    scores_sorted = scores_finite[order]
    y_sorted = y_finite[order].astype(np.int64)

    cum_pos = np.concatenate(([0], np.cumsum(y_sorted))).astype(np.int64)
    cum_neg = np.concatenate(([0], np.cumsum(1 - y_sorted))).astype(np.int64)
    p_finite = int(cum_pos[-1])
    n_finite = int(cum_neg[-1])
    n_sorted = scores_sorted.size

    uniq_vals, counts = np.unique(scores_sorted, return_counts=True)
    group_ends = np.cumsum(counts).astype(np.int64)

    if uniq_vals.size == 1:
        cuts = np.array([0, 0, n_sorted], dtype=np.int64)
        candidates = np.array(
            [uniq_vals[0] - eps, uniq_vals[0], uniq_vals[0] + eps],
            dtype=np.float64,
        )
    else:
        cuts = np.concatenate(([0], group_ends)).astype(np.int64)
        midpoints = (uniq_vals[:-1] + uniq_vals[1:]) / 2.0
        candidates = np.concatenate(
            ([uniq_vals[0] - eps], midpoints, [uniq_vals[-1] + eps])
        ).astype(np.float64)

    tp = (p_finite - cum_pos[cuts]).astype(np.float64)
    fp = (n_finite - cum_neg[cuts]).astype(np.float64)
    fn = float(total_pos) - tp
    tn = float(total_neg) - fp

    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "mcc":
            numer = tp * tn - fp * fn
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            vals = np.where(denom > 0, numer / denom, 0.0)
        else:  # f1
            denom = 2.0 * tp + fp + fn
            vals = np.where(denom > 0, (2.0 * tp) / denom, 0.0)

    best_idx = int(np.argmax(vals))
    return float(candidates[best_idx]), float(vals[best_idx])


def train_test_split(data: pd.DataFrame, test_frac: float, random_state: int, split_column: str = 'segment_id') -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing sets based on unique IDs.
    Parameters:
    - data: A pandas DataFrame containing the data to split.
    - test_frac: The fraction of unique IDs to include in the test set (between 0 and 1).
    - random_state: An integer seed for reproducibility of the random sampling.
    - split_column: Column name to use for splitting (default is 'segment_id').
    Returns:
    - A tuple containing the training set and the testing set as pandas DataFrames.
    """
    unique_ids = pd.Series(data[split_column].unique())
    if len(unique_ids) == 0:
        raise ValueError(f"No unique IDs found in the data. Ensure '{split_column}' column exists and is not empty.")
    if len(unique_ids) == 1:
        raise ValueError("Only one unique ID found. Cannot perform train/test split with a single class.")
    sampled_ids = unique_ids.sample(frac=test_frac, random_state=random_state)
    test_set = data[data[split_column].isin(sampled_ids)]
    train_set = data[~data[split_column].isin(sampled_ids)]
    return train_set, test_set


def flatten_pairwise_exclude_diag(distances: np.ndarray, classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Flatten square pairwise matrices excluding diagonal.

    Returns (dist_flat, gt_flat, n) where n is original matrix size.
    Raises ValueError for shape issues.
    """
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Distances must be a square matrix.")
    if classes.shape != distances.shape:
        raise ValueError("Classes matrix must match distances shape.")
    n = distances.shape[0]
    mask = ~np.eye(n, dtype=bool)
    dist_flat = distances[mask]
    gt_flat = classes[mask]
    return dist_flat, gt_flat, n


def filter_finite(dist_flat: np.ndarray, gt_flat: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, bool]:
    """Filter out non-finite distances.

    Returns (dist_f, gt_f, has_infinite) where has_infinite is True if any non-finite values were present.
    """
    finite_mask = np.isfinite(dist_flat)
    has_infinite = not np.all(finite_mask)
    dist_f = dist_flat[finite_mask]
    gt_f = gt_flat[finite_mask]
    return dist_f, gt_f, has_infinite


def compute_threshold_candidates(dist_f_train: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Compute candidate thresholds from finite train distances.

    Mirrors the logic in the original CLI: if only one unique value, build [v-eps, v, v+eps],
    otherwise use midpoints between sorted unique values and extend at ends by -eps/+eps.
    Raises ValueError if input is empty.
    """
    finite_dists = np.unique(dist_f_train)
    if finite_dists.size == 0:
        raise ValueError("No finite distances in train set; cannot determine threshold.")
    if finite_dists.size == 1:
        candidates = np.array([finite_dists[0] - eps, finite_dists[0], finite_dists[0] + eps], dtype=float)
    else:
        midpoints = (finite_dists[:-1] + finite_dists[1:]) / 2.0
        candidates = np.concatenate(([finite_dists[0] - eps], midpoints, [finite_dists[-1] + eps])).astype(float)
    return candidates


def find_best_threshold(dist_flat: np.ndarray, gt_flat: np.ndarray, metric: str) -> tuple[float, dict]:
    """Find the threshold maximising *metric* for pairwise distance data.

    Uses ``gt == 0`` as the positive-class convention (same-class pairs).
    Returns ``(best_threshold, stats_dict)`` where stats_dict contains
    TP/FP/TN/FN/mcc/f1/accuracy/precision/recall at the best threshold.
    """
    candidates = compute_threshold_candidates(dist_flat)
    m = compute_threshold_metrics_for_train(dist_flat, gt_flat, candidates)
    TP = m["TP"].astype(np.float64)
    FP = m["FP"].astype(np.float64)
    TN = m["TN"].astype(np.float64)
    FN = m["FN"].astype(np.float64)
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc_arr = np.where(denom > 0.0, (TP * TN - FP * FN) / denom, 0.0)
    score_arr = m["f1_pos_arr"] if metric == "f1" else mcc_arr
    idx = int(np.argmax(score_arr))
    stats = {
        "TP": int(m["TP"][idx]), "FP": int(m["FP"][idx]),
        "TN": int(m["TN"][idx]), "FN": int(m["FN"][idx]),
        "mcc": float(mcc_arr[idx]),
        "f1": float(m["f1_pos_arr"][idx]),
        "accuracy": float(m["acc_arr"][idx]),
        "precision": float(m["precision_pos"][idx]),
        "recall": float(m["recall_pos"][idx]),
    }
    return float(m["candidates_arr"][idx]), stats


def compute_threshold_metrics_for_train(dist_f_train: np.ndarray, gt_f_train: np.ndarray, candidates: np.ndarray) -> \
        Dict[str, np.ndarray]:
    """Compute TP/FP/TN/FN and derived metrics for each candidate threshold.

    Returns a dict containing arrays: TP, FP, TN, FN, precision_pos, recall_pos, f1_pos_arr, precision_neg, recall_neg, f1_neg_arr,
    f1_macro_arr, acc_arr, candidates_arr (same as candidates).
    """
    if dist_f_train.ndim != 1 or gt_f_train.ndim != 1:
        raise ValueError("dist_f_train and gt_f_train must be 1D arrays")
    if dist_f_train.shape[0] != gt_f_train.shape[0]:
        raise ValueError("dist_f_train and gt_f_train must have same length")

    order = np.argsort(dist_f_train)
    sorted_d = dist_f_train[order]
    # gt: 0 means same/positive class for "same"
    sorted_pos = (gt_f_train[order] == 0).astype(np.int64)

    cum_pos = np.cumsum(sorted_pos)
    cum_neg = np.cumsum(1 - sorted_pos)
    total_pos = int(sorted_pos.sum())
    total_neg = int(len(sorted_pos) - total_pos)

    uniq_vals, counts = np.unique(sorted_d, return_counts=True)
    group_end_indices = np.cumsum(counts) - 1

    if uniq_vals.size == 1:
        pos_indices = np.array([-1, group_end_indices[0], group_end_indices[0]], dtype=int)
    else:
        mid_indices = group_end_indices[:-1]
        pos_indices = np.concatenate(
            (np.array([-1], dtype=int), mid_indices, np.array([group_end_indices[-1]], dtype=int)))

    k = len(pos_indices)
    TP = np.empty(k, dtype=np.int64)
    FP = np.empty(k, dtype=np.int64)
    for idx_i, pos in enumerate(pos_indices):
        if pos == -1:
            tp = 0
            fp = 0
        else:
            tp = int(cum_pos[pos])
            fp = int(cum_neg[pos])
        TP[idx_i] = tp
        FP[idx_i] = fp

    FN = total_pos - TP
    TN = total_neg - FP

    with np.errstate(invalid="ignore", divide="ignore"):
        precision_pos = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
        recall_pos = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
        f1_pos_arr = np.where((precision_pos + recall_pos) > 0,
                              (2 * precision_pos * recall_pos) / (precision_pos + recall_pos), 0.0)

        precision_neg = np.where((TN + FN) > 0, TN / (TN + FN), 0.0)
        recall_neg = np.where((TN + FP) > 0, TN / (TN + FP), 0.0)
        f1_neg_arr = np.where((precision_neg + recall_neg) > 0,
                              (2 * precision_neg * recall_neg) / (precision_neg + recall_neg), 0.0)

        f1_macro_arr = (f1_pos_arr + f1_neg_arr) / 2.0
        denom = (TP + TN + FP + FN)
        acc_arr = np.where(denom > 0, (TP + TN) / denom, 0.0)

    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'precision_pos': precision_pos,
        'recall_pos': recall_pos,
        'f1_pos_arr': f1_pos_arr,
        'precision_neg': precision_neg,
        'recall_neg': recall_neg,
        'f1_neg_arr': f1_neg_arr,
        'f1_macro_arr': f1_macro_arr,
        'acc_arr': acc_arr,
        'candidates_arr': candidates,
    }
