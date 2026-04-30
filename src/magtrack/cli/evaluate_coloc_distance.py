from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from magtrack.utils.distance_dataset import load_and_stitch
from magtrack.utils.evaluation import (
    filter_finite,
    flatten_pairwise_exclude_diag,
    train_test_split, find_best_threshold,
)
from magtrack.utils.utils import resolve_seed

app = typer.Typer(help="Evaluate colocation using pairwise distance metrics.")


# ---------------------------------------------------------------------------
# Top-level multiprocessing workers (must be importable at module level)
# ---------------------------------------------------------------------------

def _compute_dtw_pair(args: tuple) -> tuple[str, str, float, float]:
    """Return (id_a, id_b, distance, elapsed_seconds)."""
    id_a, id_b, a, b, radius = args
    from magtrack.utils.distance import dtw_distance
    t0 = time.perf_counter()
    d = dtw_distance(a, b, radius=radius)
    return id_a, id_b, d, time.perf_counter() - t0


def _compute_euclidean_pair(args: tuple) -> tuple[str, str, float, float]:
    """Return (id_a, id_b, distance, elapsed_seconds)."""
    id_a, id_b, a, b = args
    t0 = time.perf_counter()
    d = float(np.linalg.norm(a - b))
    return id_a, id_b, d, time.perf_counter() - t0


def _compute_cosine_pair(args: tuple) -> tuple[str, str, float, float]:
    """Return (id_a, id_b, distance, elapsed_seconds)."""
    id_a, id_b, a, b = args
    from magtrack.utils.distance import cosine_distance
    t0 = time.perf_counter()
    d = cosine_distance(a, b)
    return id_a, id_b, d, time.perf_counter() - t0


def _compute_ddtw_pair(args: tuple) -> tuple[str, str, float, float]:
    """Return (id_a, id_b, distance, elapsed_seconds)."""
    id_a, id_b, a, b, K = args
    from magtrack.utils.distance import ddtw_distance
    t0 = time.perf_counter()
    d = ddtw_distance(a, b, K=K)
    return id_a, id_b, d, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# DuckDB distance cache (with per-pair timing)
# ---------------------------------------------------------------------------

_TABLE_DTW = "dtw_distances"
_TABLE_EUCLIDEAN = "euclidean_distances"
_TABLE_COSINE = "cosine_distances"
_TABLE_DDTW = "ddtw_distances"


def _open_cache(pkl_path: Path) -> tuple[duckdb.DuckDBPyConnection, Path]:
    """Open / create a .cache.duckdb next to the .pkl file."""
    cache_path = pkl_path.with_suffix(".cache.duckdb")
    con = duckdb.connect(str(cache_path))
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE_DTW} (
            id_a     TEXT  NOT NULL,
            id_b     TEXT  NOT NULL,
            radius   INTEGER NOT NULL,
            distance DOUBLE NOT NULL,
            calc_time_s DOUBLE NOT NULL DEFAULT 0,
            PRIMARY KEY (id_a, id_b, radius)
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE_EUCLIDEAN} (
            id_a     TEXT  NOT NULL,
            id_b     TEXT  NOT NULL,
            distance DOUBLE NOT NULL,
            calc_time_s DOUBLE NOT NULL DEFAULT 0,
            PRIMARY KEY (id_a, id_b)
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE_COSINE} (
            id_a     TEXT  NOT NULL,
            id_b     TEXT  NOT NULL,
            distance DOUBLE NOT NULL,
            calc_time_s DOUBLE NOT NULL DEFAULT 0,
            PRIMARY KEY (id_a, id_b)
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE_DDTW} (
            id_a     TEXT  NOT NULL,
            id_b     TEXT  NOT NULL,
            K        INTEGER NOT NULL,
            distance DOUBLE NOT NULL,
            calc_time_s DOUBLE NOT NULL DEFAULT 0,
            PRIMARY KEY (id_a, id_b, K)
        )
    """)
    n_dtw = con.execute(f"SELECT COUNT(*) FROM {_TABLE_DTW}").fetchone()[0]
    n_euc = con.execute(f"SELECT COUNT(*) FROM {_TABLE_EUCLIDEAN}").fetchone()[0]
    n_cos = con.execute(f"SELECT COUNT(*) FROM {_TABLE_COSINE}").fetchone()[0]
    n_ddtw = con.execute(f"SELECT COUNT(*) FROM {_TABLE_DDTW}").fetchone()[0]
    typer.echo(f"  Cache: {cache_path}  (dtw={n_dtw:,}, euclidean={n_euc:,}, cosine={n_cos:,}, ddtw={n_ddtw:,} entries)")
    return con, cache_path


# --- DTW cache helpers ---

def _get_cached_dtw(
        con: duckdb.DuckDBPyConnection,
        pairs: list[tuple[str, str]],
        radius: int,
) -> dict[tuple[str, str], float]:
    if not pairs:
        return {}
    pdf = pd.DataFrame(pairs, columns=["id_a", "id_b"])
    con.register("_qp", pdf)
    try:
        rows = con.execute(f"""
            SELECT d.id_a, d.id_b, d.distance
            FROM {_TABLE_DTW} d
            INNER JOIN _qp q ON d.id_a = q.id_a AND d.id_b = q.id_b
            WHERE d.radius = ?
        """, [radius]).fetchall()
    finally:
        con.unregister("_qp")
    return {(r[0], r[1]): float(r[2]) for r in rows}


def _insert_dtw(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    """rows: list of (id_a, id_b, radius, distance, calc_time_s)."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["id_a", "id_b", "radius", "distance", "calc_time_s"])
    con.register("_ir", df)
    try:
        con.execute(f"""
            INSERT INTO {_TABLE_DTW}
            SELECT id_a, id_b, radius, distance, calc_time_s FROM _ir
            ON CONFLICT DO NOTHING
        """)
    finally:
        con.unregister("_ir")


# --- Euclidean cache helpers ---

def _get_cached_euclidean(
        con: duckdb.DuckDBPyConnection,
        pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    if not pairs:
        return {}
    pdf = pd.DataFrame(pairs, columns=["id_a", "id_b"])
    con.register("_qp", pdf)
    try:
        rows = con.execute(f"""
            SELECT d.id_a, d.id_b, d.distance
            FROM {_TABLE_EUCLIDEAN} d
            INNER JOIN _qp q ON d.id_a = q.id_a AND d.id_b = q.id_b
        """).fetchall()
    finally:
        con.unregister("_qp")
    return {(r[0], r[1]): float(r[2]) for r in rows}


def _insert_euclidean(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    """rows: list of (id_a, id_b, distance, calc_time_s)."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["id_a", "id_b", "distance", "calc_time_s"])
    con.register("_ir", df)
    try:
        con.execute(f"""
            INSERT INTO {_TABLE_EUCLIDEAN}
            SELECT id_a, id_b, distance, calc_time_s FROM _ir
            ON CONFLICT DO NOTHING
        """)
    finally:
        con.unregister("_ir")


# --- Cosine cache helpers ---

def _get_cached_cosine(
        con: duckdb.DuckDBPyConnection,
        pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    if not pairs:
        return {}
    pdf = pd.DataFrame(pairs, columns=["id_a", "id_b"])
    con.register("_qp", pdf)
    try:
        rows = con.execute(f"""
            SELECT d.id_a, d.id_b, d.distance
            FROM {_TABLE_COSINE} d
            INNER JOIN _qp q ON d.id_a = q.id_a AND d.id_b = q.id_b
        """).fetchall()
    finally:
        con.unregister("_qp")
    return {(r[0], r[1]): float(r[2]) for r in rows}


def _insert_cosine(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    """rows: list of (id_a, id_b, distance, calc_time_s)."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["id_a", "id_b", "distance", "calc_time_s"])
    con.register("_ir", df)
    try:
        con.execute(f"""
            INSERT INTO {_TABLE_COSINE}
            SELECT id_a, id_b, distance, calc_time_s FROM _ir
            ON CONFLICT DO NOTHING
        """)
    finally:
        con.unregister("_ir")


# --- DDTW cache helpers ---

def _get_cached_ddtw(
        con: duckdb.DuckDBPyConnection,
        pairs: list[tuple[str, str]],
        K: int,
) -> dict[tuple[str, str], float]:
    if not pairs:
        return {}
    pdf = pd.DataFrame(pairs, columns=["id_a", "id_b"])
    con.register("_qp", pdf)
    try:
        rows = con.execute(f"""
            SELECT d.id_a, d.id_b, d.distance
            FROM {_TABLE_DDTW} d
            INNER JOIN _qp q ON d.id_a = q.id_a AND d.id_b = q.id_b
            WHERE d.K = ?
        """, [K]).fetchall()
    finally:
        con.unregister("_qp")
    return {(r[0], r[1]): float(r[2]) for r in rows}


def _insert_ddtw(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    """rows: list of (id_a, id_b, K, distance, calc_time_s)."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["id_a", "id_b", "K", "distance", "calc_time_s"])
    con.register("_ir", df)
    try:
        con.execute(f"""
            INSERT INTO {_TABLE_DDTW}
            SELECT id_a, id_b, K, distance, calc_time_s FROM _ir
            ON CONFLICT DO NOTHING
        """)
    finally:
        con.unregister("_ir")


# ---------------------------------------------------------------------------
# Parallel pairwise distance computation (with cache & resume)
# ---------------------------------------------------------------------------

def _compute_pairwise_dtw(
        data: pd.DataFrame,
        radius: int,
        n_workers: int,
        con: duckdb.DuckDBPyConnection,
        chunksize: int = 1,
) -> tuple[dict[tuple[str, str], float], float]:
    """Returns (distances_dict, total_cpu_seconds_from_cache)."""
    ids = data["id"].tolist()
    N = len(ids)
    all_pairs = [(ids[i], ids[j]) for i in range(N) for j in range(i + 1, N)]
    if not all_pairs:
        return {}, 0.0

    cached = _get_cached_dtw(con, all_pairs, radius)
    missing = [p for p in all_pairs if p not in cached]

    if missing:
        trace = {row["id"]: row["data"]["magnitude"].to_numpy(dtype=np.float32) for _, row in data.iterrows()}
        args = [(a, b, trace[a], trace[b], radius) for a, b in missing]
        batch: list[tuple] = []
        with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
            with tqdm(total=len(all_pairs), initial=len(cached), desc=f"  DTW r={radius}", leave=False) as pbar:
                for result in pool.map(_compute_dtw_pair, args, chunksize=chunksize):
                    a, b, d, t = result
                    cached[(a, b)] = d
                    batch.append((a, b, radius, d, t))
                    pbar.update(1)
                    if len(batch) >= chunksize:
                        _insert_dtw(con, batch)
                        con.execute("CHECKPOINT")
                        batch.clear()
        if batch:
            _insert_dtw(con, batch)
            con.execute("CHECKPOINT")

    # Query total CPU time from cache for these pairs
    total_cpu_s = con.execute(f"SELECT COALESCE(SUM(calc_time_s), 0) FROM {_TABLE_DTW} WHERE radius = ?", [radius]).fetchone()[0]
    return cached, float(total_cpu_s)


def _compute_pairwise_euclidean(
        data: pd.DataFrame,
        n_workers: int,
        con: duckdb.DuckDBPyConnection,
        chunksize: int = 1,
) -> tuple[dict[tuple[str, str], float], float]:
    """Returns (distances_dict, total_cpu_seconds_from_cache)."""
    ids = data["id"].tolist()
    N = len(ids)
    all_pairs = [(ids[i], ids[j]) for i in range(N) for j in range(i + 1, N)]
    if not all_pairs:
        return {}, 0.0

    cached = _get_cached_euclidean(con, all_pairs)
    missing = [p for p in all_pairs if p not in cached]

    if missing:
        trace = {row["id"]: row["data"]["magnitude"].to_numpy(dtype=np.float64) for _, row in data.iterrows()}
        args = [(a, b, trace[a], trace[b]) for a, b in missing]
        batch: list[tuple] = []
        with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
            with tqdm(total=len(all_pairs), initial=len(cached), desc=f"  Euclidean", leave=False) as pbar:
                for result in pool.map(_compute_euclidean_pair, args, chunksize=chunksize):
                    a, b, d, t = result
                    cached[(a, b)] = d
                    batch.append((a, b, d, t))
                    pbar.update(1)
                    if len(batch) >= chunksize:
                        _insert_euclidean(con, batch)
                        con.execute("CHECKPOINT")
                        batch.clear()
        if batch:
            _insert_euclidean(con, batch)
            con.execute("CHECKPOINT")

    total_cpu_s = con.execute(f"SELECT COALESCE(SUM(calc_time_s), 0) FROM {_TABLE_EUCLIDEAN}").fetchone()[0]
    return cached, float(total_cpu_s)


def _compute_pairwise_cosine(
        data: pd.DataFrame,
        n_workers: int,
        con: duckdb.DuckDBPyConnection,
        chunksize: int = 1,
) -> tuple[dict[tuple[str, str], float], float]:
    """Returns (distances_dict, total_cpu_seconds_from_cache)."""
    ids = data["id"].tolist()
    N = len(ids)
    all_pairs = [(ids[i], ids[j]) for i in range(N) for j in range(i + 1, N)]
    if not all_pairs:
        return {}, 0.0

    cached = _get_cached_cosine(con, all_pairs)
    missing = [p for p in all_pairs if p not in cached]

    if missing:
        trace = {row["id"]: row["data"]["magnitude"].to_numpy(dtype=np.float64) for _, row in data.iterrows()}
        args = [(a, b, trace[a], trace[b]) for a, b in missing]
        batch: list[tuple] = []
        with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
            with tqdm(total=len(all_pairs), initial=len(cached), desc="  Cosine", leave=False) as pbar:
                for result in pool.map(_compute_cosine_pair, args, chunksize=chunksize):
                    a, b, d, t = result
                    cached[(a, b)] = d
                    batch.append((a, b, d, t))
                    pbar.update(1)
                    if len(batch) >= chunksize:
                        _insert_cosine(con, batch)
                        con.execute("CHECKPOINT")
                        batch.clear()
        if batch:
            _insert_cosine(con, batch)
            con.execute("CHECKPOINT")

    total_cpu_s = con.execute(f"SELECT COALESCE(SUM(calc_time_s), 0) FROM {_TABLE_COSINE}").fetchone()[0]
    return cached, float(total_cpu_s)


def _compute_pairwise_ddtw(
        data: pd.DataFrame,
        K: int,
        n_workers: int,
        con: duckdb.DuckDBPyConnection,
        chunksize: int = 1,
) -> tuple[dict[tuple[str, str], float], float]:
    """Returns (distances_dict, total_cpu_seconds_from_cache)."""
    ids = data["id"].tolist()
    N = len(ids)
    all_pairs = [(ids[i], ids[j]) for i in range(N) for j in range(i + 1, N)]
    if not all_pairs:
        return {}, 0.0

    cached = _get_cached_ddtw(con, all_pairs, K)
    missing = [p for p in all_pairs if p not in cached]

    if missing:
        trace = {row["id"]: row["data"]["magnitude"].to_numpy(dtype=np.float64) for _, row in data.iterrows()}
        args = [(a, b, trace[a], trace[b], K) for a, b in missing]
        batch: list[tuple] = []
        with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
            with tqdm(total=len(all_pairs), initial=len(cached), desc=f"  DDTW K={K}", leave=False) as pbar:
                for result in pool.map(_compute_ddtw_pair, args, chunksize=chunksize):
                    a, b, d, t = result
                    cached[(a, b)] = d
                    batch.append((a, b, K, d, t))
                    pbar.update(1)
                    if len(batch) >= chunksize:
                        _insert_ddtw(con, batch)
                        con.execute("CHECKPOINT")
                        batch.clear()
        if batch:
            _insert_ddtw(con, batch)
            con.execute("CHECKPOINT")

    total_cpu_s = con.execute(f"SELECT COALESCE(SUM(calc_time_s), 0) FROM {_TABLE_DDTW} WHERE K = ?", [K]).fetchone()[0]
    return cached, float(total_cpu_s)


# ---------------------------------------------------------------------------
# Matrix building, threshold search, metrics
# ---------------------------------------------------------------------------

def _build_split_matrices(
        split_data: pd.DataFrame,
        all_distances: dict[tuple[str, str], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build pairwise distance + class matrices.  0 = co-located, 1 = different."""
    ids = split_data["id"].tolist()
    segs = split_data["segment_id"].tolist()
    N = len(ids)
    dist = np.full((N, N), np.nan, dtype=np.float64)
    cls = np.ones((N, N), dtype=np.int64)
    for i in range(N):
        dist[i, i] = 0.0
        cls[i, i] = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = all_distances.get((ids[i], ids[j]), all_distances.get((ids[j], ids[i]), np.nan))
            dist[i, j] = dist[j, i] = d
            if segs[i] == segs[j]:
                cls[i, j] = cls[j, i] = 0
    return dist, cls


def _metrics_at_threshold(dist: np.ndarray, gt: np.ndarray, threshold: float) -> dict:
    pred = np.where(dist <= threshold, 0, 1).astype(np.int64)
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    tp, fn, fp, tn = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "mcc": float(matthews_corrcoef(gt, pred)),
        "f1": float(f1_score(gt, pred, pos_label=0, zero_division=0.0)),
        "accuracy": float(accuracy_score(gt, pred)),
        "precision": float(precision_score(gt, pred, pos_label=0, zero_division=0.0)),
        "recall": float(recall_score(gt, pred, pos_label=0, zero_division=0.0)),
    }


def _compute_mrr(dist_mat: np.ndarray, cls_mat: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.

    For each sample, rank all others by ascending distance, find the rank of the
    first co-located sample (cls == 0), and return the mean of 1/rank.
    Samples with no co-located partner are excluded from the mean.
    """
    N = dist_mat.shape[0]
    rr_sum = 0.0
    count = 0
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        dists = dist_mat[i, mask]
        classes = cls_mat[i, mask]

        if not np.any(classes == 0):
            continue

        sorted_idx = np.argsort(dists)
        sorted_classes = classes[sorted_idx]
        first_pos = np.argmax(sorted_classes == 0)  # index of first 0
        rr_sum += 1.0 / (first_pos + 1)
        count += 1

    return rr_sum / count if count > 0 else 0.0


def _stats_for_split(dist_mat: np.ndarray, cls_mat: np.ndarray, thresh: float) -> dict:
    d, g, _ = flatten_pairwise_exclude_diag(dist_mat, cls_mat)
    d, g, _ = filter_finite(d, g)
    s = _metrics_at_threshold(d, g, thresh)
    s["n_pairs"] = len(d)
    s["mrr"] = _compute_mrr(dist_mat, cls_mat)
    return s


# ---------------------------------------------------------------------------
# Multi-run evaluation
# ---------------------------------------------------------------------------

def _evaluate_multi_run(
        data: pd.DataFrame,
        all_distances: dict[tuple[str, str], float],
        split_seeds: list[int],
        test_frac: float,
        metric: str,
) -> tuple[dict, dict, float]:
    """Run threshold-optimised evaluation across multiple train/test splits.

    Returns (test_stats, train_stats, mean_threshold).
    """
    all_test: list[dict] = []
    all_train: list[dict] = []
    all_thresh: list[float] = []

    for seed_i in split_seeds:
        tr, te = train_test_split(data, test_frac, seed_i)
        if len(tr) < 2 or len(te) < 2:
            raise ValueError("Insufficient recordings after train/test split.")
        tr_dist, tr_cls = _build_split_matrices(tr, all_distances)
        te_dist, te_cls = _build_split_matrices(te, all_distances)

        d_flat, g_flat, _ = flatten_pairwise_exclude_diag(tr_dist, tr_cls)
        d_flat, g_flat, _ = filter_finite(d_flat, g_flat)
        if len(d_flat) == 0:
            raise ValueError("No finite distances in train set.")

        thresh, tr_stats = find_best_threshold(d_flat, g_flat, metric)
        te_stats = _stats_for_split(te_dist, te_cls, thresh)

        all_test.append(te_stats)
        all_train.append(tr_stats)
        all_thresh.append(thresh)

    def _agg(stats_list: list[dict]) -> dict:
        agg: dict = {}
        for k in stats_list[0]:
            vals = [float(s[k]) for s in stats_list]
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        return agg

    if len(split_seeds) > 1:
        te_agg = _agg(all_test)
        tr_agg = _agg(all_train)
        mean_t = float(np.mean(all_thresh))
        te_agg["threshold_std"] = float(np.std(all_thresh))
    else:
        te_agg = all_test[0]
        tr_agg = all_train[0]
        mean_t = all_thresh[0]

    return te_agg, tr_agg, mean_t


# ---------------------------------------------------------------------------
# CSV results (append, skip-if-exists)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "dataset_file", "distance_function", "radius",
    "n_segment_ids", "n_ids", "n_calculated_pairs",
    "duration", "trainride_start_seconds", "window_size", "sampling_rate",
    "threshold",
    "test_frac", "seed", "metric", "n_runs", "n_workers",
    # train
    "mcc_train", "f1_train", "accuracy_train", "precision_train", "recall_train",
    "TP_train", "FP_train", "TN_train", "FN_train",
    # test
    "mcc_test", "f1_test", "accuracy_test", "precision_test", "recall_test",
    "TP_test", "FP_test", "TN_test", "FN_test", "n_pairs_test",
    "mrr_test",
    # std (multi-run)
    "mcc_train_std", "f1_train_std", "accuracy_train_std", "precision_train_std", "recall_train_std",
    "TP_train_std", "FP_train_std", "TN_train_std", "FN_train_std",
    "mcc_test_std", "f1_test_std", "accuracy_test_std", "precision_test_std", "recall_test_std",
    "TP_test_std", "FP_test_std", "TN_test_std", "FN_test_std", "n_pairs_test_std",
    "mrr_test_std",
    "threshold_std",
    "total_distance_time_s",
    "total_cpu_hours",
]


def _result_exists(csv_path: Path, dataset_file: str, distance_fn: str, radius: Optional[int] = None) -> bool:
    """Check whether a result row already exists in the CSV."""
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    mask = (df["dataset_file"] == dataset_file) & (df["distance_function"] == distance_fn)
    if radius is not None and "radius" in df.columns:
        mask = mask & (df["radius"] == radius)
    return bool(mask.any())


def _build_row(
        *,
        dataset_file: str,
        distance_fn: str,
        radius: Optional[int],
        n_segment_ids: int,
        n_ids: int,
        n_calculated_pairs: int,
        duration: Optional[int],
        trainride_start_seconds: Optional[int],
        window_size: Optional[int],
        sampling_rate: Optional[int],
        threshold: float,
        test_frac: float,
        seed: str,
        metric: str,
        n_runs: int,
        n_workers: int,
        test_stats: dict,
        train_stats: dict,
        total_time: float,
        total_cpu_hours: float = 0.0,
) -> dict:
    row: dict = {
        "dataset_file": dataset_file,
        "distance_function": distance_fn,
        "threshold": threshold,
        "n_segment_ids": n_segment_ids,
        "n_ids": n_ids,
        "n_calculated_pairs": n_calculated_pairs,
        "duration": duration,
        "trainride_start_seconds": trainride_start_seconds,
        "window_size": window_size,
        "sampling_rate": sampling_rate,
        "test_frac": test_frac,
        "seed": seed,
        "metric": metric,
        "n_runs": n_runs,
        "n_workers": n_workers,
        "total_distance_time_s": round(total_time, 3),
        "total_cpu_hours": round(total_cpu_hours, 4),
    }
    if radius is not None:
        row["radius"] = radius
    for suffix, stats in [("train", train_stats), ("test", test_stats)]:
        for k in ("mcc", "f1", "accuracy", "precision", "recall", "TP", "FP", "TN", "FN"):
            row[f"{k}_{suffix}"] = stats.get(k, 0)
            std_key = f"{k}_std"
            if std_key in stats:
                row[f"{k}_{suffix}_std"] = stats[std_key]
        if suffix == "test":
            row["n_pairs_test"] = stats.get("n_pairs", 0)
            if "n_pairs_std" in stats:
                row["n_pairs_test_std"] = stats["n_pairs_std"]
            row["mrr_test"] = stats.get("mrr", 0)
            if "mrr_std" in stats:
                row["mrr_test_std"] = stats["mrr_std"]
    if "threshold_std" in test_stats:
        row["threshold_std"] = test_stats["threshold_std"]
    return row


def _print_results(test_stats: dict, train_stats: dict, threshold: float, metric: str, multi_run: bool) -> None:
    typer.echo(f"\n{'='*50}")
    typer.echo(f"  threshold      : {threshold:.6f}")
    for split_name, stats in [("train", train_stats), ("test", test_stats)]:
        for m in ("mcc", "f1", "accuracy", "precision", "recall"):
            val = stats.get(m, 0.0)
            std = stats.get(f"{m}_std")
            if multi_run and std is not None:
                typer.echo(f"  {m:12s} {split_name:5s}: {val:.4f} ± {std:.4f}")
            else:
                typer.echo(f"  {m:12s} {split_name:5s}: {val:.4f}")
    mrr_val = test_stats.get("mrr", 0.0)
    mrr_std = test_stats.get("mrr_std")
    if multi_run and mrr_std is not None:
        typer.echo(f"  {'mrr':12s} {'test':5s}: {mrr_val:.4f} ± {mrr_std:.4f}")
    else:
        typer.echo(f"  {'mrr':12s} {'test':5s}: {mrr_val:.4f}")
    typer.echo(f"{'='*50}")

# ---------------------------------------------------------------------------
# CLI: DTW subcommand
# ---------------------------------------------------------------------------

@app.command("dtw")
def evaluate_dtw(
        pkl_path: Path = typer.Argument(..., help="Path to a single colocation .pkl dataset."),
        results_csv: Path = typer.Argument(..., help="CSV file to append results to."),
        sample: Optional[float] = typer.Option(None, help="Fraction of recordings to sample (0–1). Uses seed for reproducibility."),
        # Parallelism
        workers: Optional[int] = typer.Option(None, help="Parallel workers (default: all CPU cores)."),
        chunksize: int = typer.Option(32, help="Chunk size for parallel worker dispatch (higher = less IPC overhead, coarser progress)."),
        # DTW-specific
        radius: int = typer.Option(1, help="DTW Sakoe-Chiba radius."),
        # Evaluation
        test_frac: float = typer.Option(0.3, help="Fraction of segment_ids held out as test set (0–1)."),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split."),
        metric: str = typer.Option("mcc", help="Metric to optimise threshold: 'mcc' or 'f1'."),
        runs: int = typer.Option(10, help="Number of evaluation runs (>1 = average over N splits)."),
) -> None:
    """Evaluate DTW distance for colocation verification on a single dataset."""
    if metric not in ("mcc", "f1"):
        typer.echo(f"Unknown metric '{metric}'.", err=True)
        raise typer.Exit(code=1)
    if sample is not None and not (0.0 < sample <= 1.0):
        typer.echo("--sample must be between 0 (exclusive) and 1 (inclusive).", err=True)
        raise typer.Exit(code=1)

    n_workers_val = workers if workers is not None else multiprocessing.cpu_count()

    if _result_exists(results_csv, pkl_path.name, "dtw", radius):
        typer.echo(f"Result already exists for {pkl_path.name} (dtw, r={radius}). Printing cached result.")
        df = pd.read_csv(results_csv)
        row = df[(df["dataset_file"] == pkl_path.name) & (df["distance_function"] == "dtw") & (df["radius"] == radius)].iloc[-1]
        typer.echo(row.to_string())
        return

    data, meta = load_and_stitch(pkl_path)

    if sample is not None and sample < 1.0:
        seed_int_sample = int.from_bytes(seed.encode(), "big") % 9_999_999
        _, data = train_test_split(data, sample, seed_int_sample)
        typer.echo(f"  Sampled {len(data)} recordings ({sample:.0%} of total).")

    con, _ = _open_cache(pkl_path)

    typer.echo(f"\nComputing DTW distances (radius={radius}, n_workers={n_workers_val}) …")
    t0 = time.perf_counter()
    all_distances, total_cpu_s = _compute_pairwise_dtw(data, radius, n_workers_val, con, chunksize=chunksize)
    total_time = time.perf_counter() - t0
    total_cpu_h = total_cpu_s / 3600.0
    typer.echo(f"  Distance computation: {total_time:.1f}s wall, {total_cpu_h:.2f} CPU-hours ({len(all_distances):,} pairs total)")

    seed_int = resolve_seed(seed)
    multi_run = runs > 1
    split_seeds = [seed_int + i for i in range(runs)] if multi_run else [seed_int]

    test_stats, train_stats, threshold = _evaluate_multi_run(data, all_distances, split_seeds, test_frac, metric)

    _print_results(test_stats, train_stats, threshold, metric, multi_run)

    if not results_csv.exists():
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=_CSV_COLUMNS).to_csv(results_csv, index=False)

    row = _build_row(
        dataset_file=pkl_path.name,
        distance_fn="dtw",
        radius=radius,
        n_segment_ids=data["segment_id"].nunique(),
        n_ids=data["id"].nunique(),
        n_calculated_pairs=len(all_distances),
        duration=meta.get("duration"),
        trainride_start_seconds=meta.get("trainride_start_seconds"),
        window_size=meta.get("rolling_window"),
        sampling_rate=meta.get("sampling_rate"),
        threshold=threshold,
        test_frac=test_frac,
        seed=seed,
        metric=metric,
        n_runs=runs,
        n_workers=n_workers_val,
        test_stats=test_stats,
        train_stats=train_stats,
        total_time=total_time,
        total_cpu_hours=total_cpu_h,
    )
    pd.DataFrame([row], columns=_CSV_COLUMNS).to_csv(results_csv, index=False, mode="a", header=False)
    typer.echo(f"  Results appended to {results_csv}")

    con.close()


# ---------------------------------------------------------------------------
# CLI: Euclidean subcommand
# ---------------------------------------------------------------------------

@app.command("euclidean")
def evaluate_euclidean(
        pkl_path: Path = typer.Argument(..., help="Path to a single colocation .pkl dataset."),
        results_csv: Path = typer.Argument(..., help="CSV file to append results to."),
        sample: Optional[float] = typer.Option(None, help="Fraction of recordings to sample (0–1). Uses seed for reproducibility."),
        # Parallelism
        workers: Optional[int] = typer.Option(None, help="Parallel workers (default: all CPU cores)."),
        chunksize: int = typer.Option(1000,
                                      help="Chunk size for parallel worker dispatch (higher = less IPC overhead, coarser progress)."),
        # Evaluation
        test_frac: float = typer.Option(0.3, help="Fraction of segment_ids held out as test set (0–1)."),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split."),
        metric: str = typer.Option("mcc", help="Metric to optimise threshold: 'mcc' or 'f1'."),
        runs: int = typer.Option(10, help="Number of evaluation runs (>1 = average over N splits)."),
) -> None:
    """Evaluate Euclidean distance for colocation verification on a single dataset."""
    if metric not in ("mcc", "f1"):
        typer.echo(f"Unknown metric '{metric}'.", err=True)
        raise typer.Exit(code=1)
    if sample is not None and not (0.0 < sample <= 1.0):
        typer.echo("--sample must be between 0 (exclusive) and 1 (inclusive).", err=True)
        raise typer.Exit(code=1)

    n_workers_val = workers if workers is not None else multiprocessing.cpu_count()

    if _result_exists(results_csv, pkl_path.name, "euclidean"):
        typer.echo(f"Result already exists for {pkl_path.name} (euclidean). Printing cached result.")
        df = pd.read_csv(results_csv)
        row = df[(df["dataset_file"] == pkl_path.name) & (df["distance_function"] == "euclidean")].iloc[-1]
        typer.echo(row.to_string())
        return

    data, meta = load_and_stitch(pkl_path)

    if sample is not None and sample < 1.0:
        seed_int_sample = int.from_bytes(seed.encode(), "big") % 9_999_999
        _, data = train_test_split(data, sample, seed_int_sample)
        typer.echo(f"  Sampled {len(data)} recordings ({sample:.0%} of total).")

    con, _ = _open_cache(pkl_path)

    typer.echo(f"\nComputing Euclidean distances (n_workers={n_workers_val}) …")
    t0 = time.perf_counter()
    all_distances, total_cpu_s = _compute_pairwise_euclidean(data, n_workers_val, con, chunksize=chunksize)
    total_time = time.perf_counter() - t0
    total_cpu_h = total_cpu_s / 3600.0
    typer.echo(f"  Distance computation: {total_time:.1f}s wall, {total_cpu_h:.2f} CPU-hours ({len(all_distances):,} pairs total)")

    seed_int = resolve_seed(seed)
    multi_run = runs > 1
    split_seeds = [seed_int + i for i in range(runs)] if multi_run else [seed_int]

    test_stats, train_stats, threshold = _evaluate_multi_run(data, all_distances, split_seeds, test_frac, metric)

    _print_results(test_stats, train_stats, threshold, metric, multi_run)

    if not results_csv.exists():
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=_CSV_COLUMNS).to_csv(results_csv, index=False)

    row = _build_row(
        dataset_file=pkl_path.name,
        distance_fn="euclidean",
        radius=None,
        n_segment_ids=data["segment_id"].nunique(),
        n_ids=data["id"].nunique(),
        n_calculated_pairs=len(all_distances),
        duration=meta.get("duration"),
        trainride_start_seconds=meta.get("trainride_start_seconds"),
        window_size=meta.get("rolling_window"),
        sampling_rate=meta.get("sampling_rate"),
        threshold=threshold,
        test_frac=test_frac,
        seed=seed,
        metric=metric,
        n_runs=runs,
        n_workers=n_workers_val,
        test_stats=test_stats,
        train_stats=train_stats,
        total_time=total_time,
        total_cpu_hours=total_cpu_h,
    )
    pd.DataFrame([row], columns=_CSV_COLUMNS).to_csv(results_csv, index=False, mode="a", header=False)
    typer.echo(f"  Results appended to {results_csv}")

    con.close()


# ---------------------------------------------------------------------------
# CLI: Cosine subcommand
# ---------------------------------------------------------------------------

@app.command("cosine")
def evaluate_cosine(
        pkl_path: Path = typer.Argument(..., help="Path to a single colocation .pkl dataset."),
        results_csv: Path = typer.Argument(..., help="CSV file to append results to."),
        sample: Optional[float] = typer.Option(None, help="Fraction of recordings to sample (0–1). Uses seed for reproducibility."),
        # Parallelism
        workers: Optional[int] = typer.Option(None, help="Parallel workers (default: all CPU cores)."),
        chunksize: int = typer.Option(1000,
                                      help="Chunk size for parallel worker dispatch (higher = less IPC overhead, coarser progress)."),
        # Evaluation
        test_frac: float = typer.Option(0.3, help="Fraction of segment_ids held out as test set (0–1)."),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split."),
        metric: str = typer.Option("mcc", help="Metric to optimise threshold: 'mcc' or 'f1'."),
        runs: int = typer.Option(10, help="Number of evaluation runs (>1 = average over N splits)."),
) -> None:
    """Evaluate Cosine distance for colocation verification on a single dataset."""
    if metric not in ("mcc", "f1"):
        typer.echo(f"Unknown metric '{metric}'.", err=True)
        raise typer.Exit(code=1)
    if sample is not None and not (0.0 < sample <= 1.0):
        typer.echo("--sample must be between 0 (exclusive) and 1 (inclusive).", err=True)
        raise typer.Exit(code=1)

    n_workers_val = workers if workers is not None else multiprocessing.cpu_count()

    if _result_exists(results_csv, pkl_path.name, "cosine"):
        typer.echo(f"Result already exists for {pkl_path.name} (cosine). Printing cached result.")
        df = pd.read_csv(results_csv)
        row = df[(df["dataset_file"] == pkl_path.name) & (df["distance_function"] == "cosine")].iloc[-1]
        typer.echo(row.to_string())
        return

    data, meta = load_and_stitch(pkl_path)

    if sample is not None and sample < 1.0:
        seed_int_sample = int.from_bytes(seed.encode(), "big") % 9_999_999
        _, data = train_test_split(data, sample, seed_int_sample)
        typer.echo(f"  Sampled {len(data)} recordings ({sample:.0%} of total).")

    con, _ = _open_cache(pkl_path)

    typer.echo(f"\nComputing Cosine distances (n_workers={n_workers_val}) …")
    t0 = time.perf_counter()
    all_distances, total_cpu_s = _compute_pairwise_cosine(data, n_workers_val, con, chunksize=chunksize)
    total_time = time.perf_counter() - t0
    total_cpu_h = total_cpu_s / 3600.0
    typer.echo(f"  Distance computation: {total_time:.1f}s wall, {total_cpu_h:.2f} CPU-hours ({len(all_distances):,} pairs total)")

    seed_int = resolve_seed(seed)
    multi_run = runs > 1
    split_seeds = [seed_int + i for i in range(runs)] if multi_run else [seed_int]

    test_stats, train_stats, threshold = _evaluate_multi_run(data, all_distances, split_seeds, test_frac, metric)

    _print_results(test_stats, train_stats, threshold, metric, multi_run)

    if not results_csv.exists():
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=_CSV_COLUMNS).to_csv(results_csv, index=False)

    row = _build_row(
        dataset_file=pkl_path.name,
        distance_fn="cosine",
        radius=None,
        n_segment_ids=data["segment_id"].nunique(),
        n_ids=data["id"].nunique(),
        n_calculated_pairs=len(all_distances),
        duration=meta.get("duration"),
        trainride_start_seconds=meta.get("trainride_start_seconds"),
        window_size=meta.get("rolling_window"),
        sampling_rate=meta.get("sampling_rate"),
        threshold=threshold,
        test_frac=test_frac,
        seed=seed,
        metric=metric,
        n_runs=runs,
        n_workers=n_workers_val,
        test_stats=test_stats,
        train_stats=train_stats,
        total_time=total_time,
        total_cpu_hours=total_cpu_h,
    )
    pd.DataFrame([row], columns=_CSV_COLUMNS).to_csv(results_csv, index=False, mode="a", header=False)
    typer.echo(f"  Results appended to {results_csv}")

    con.close()


# ---------------------------------------------------------------------------
# CLI: DDTW subcommand
# ---------------------------------------------------------------------------

@app.command("ddtw")
def evaluate_ddtw(
        pkl_path: Path = typer.Argument(..., help="Path to a single colocation .pkl dataset."),
        results_csv: Path = typer.Argument(..., help="CSV file to append results to."),
        sample: Optional[float] = typer.Option(None, help="Fraction of recordings to sample (0–1). Uses seed for reproducibility."),
        # Parallelism
        workers: Optional[int] = typer.Option(None, help="Parallel workers (default: all CPU cores)."),
        chunksize: int = typer.Option(32, help="Chunk size for parallel worker dispatch (higher = less IPC overhead, coarser progress)."),
        # DDTW-specific
        K: int = typer.Option(10, "--K", "-K", help="DDTW window size parameter (window = 2*K)."),
        # Evaluation
        test_frac: float = typer.Option(0.3, help="Fraction of segment_ids held out as test set (0–1)."),
        seed: str = typer.Option("magtrack", help="Random seed string for reproducible train/test split."),
        metric: str = typer.Option("mcc", help="Metric to optimise threshold: 'mcc' or 'f1'."),
        runs: int = typer.Option(10, help="Number of evaluation runs (>1 = average over N splits)."),
) -> None:
    """Evaluate Derivative DTW (DDTW) distance for colocation verification on a single dataset."""
    if metric not in ("mcc", "f1"):
        typer.echo(f"Unknown metric '{metric}'.", err=True)
        raise typer.Exit(code=1)
    if sample is not None and not (0.0 < sample <= 1.0):
        typer.echo("--sample must be between 0 (exclusive) and 1 (inclusive).", err=True)
        raise typer.Exit(code=1)

    n_workers_val = workers if workers is not None else multiprocessing.cpu_count()

    if _result_exists(results_csv, pkl_path.name, "ddtw", K):
        typer.echo(f"Result already exists for {pkl_path.name} (ddtw, K={K}). Printing cached result.")
        df = pd.read_csv(results_csv)
        row = df[(df["dataset_file"] == pkl_path.name) & (df["distance_function"] == "ddtw") & (df["radius"] == K)].iloc[-1]
        typer.echo(row.to_string())
        return

    data, meta = load_and_stitch(pkl_path)

    if sample is not None and sample < 1.0:
        seed_int_sample = int.from_bytes(seed.encode(), "big") % 9_999_999
        _, data = train_test_split(data, sample, seed_int_sample)
        typer.echo(f"  Sampled {len(data)} recordings ({sample:.0%} of total).")

    con, _ = _open_cache(pkl_path)

    typer.echo(f"\nComputing DDTW distances (K={K}, n_workers={n_workers_val}) …")
    t0 = time.perf_counter()
    all_distances, total_cpu_s = _compute_pairwise_ddtw(data, K, n_workers_val, con, chunksize=chunksize)
    total_time = time.perf_counter() - t0
    total_cpu_h = total_cpu_s / 3600.0
    typer.echo(f"  Distance computation: {total_time:.1f}s wall, {total_cpu_h:.2f} CPU-hours ({len(all_distances):,} pairs total)")

    seed_int = resolve_seed(seed)
    multi_run = runs > 1
    split_seeds = [seed_int + i for i in range(runs)] if multi_run else [seed_int]

    test_stats, train_stats, threshold = _evaluate_multi_run(data, all_distances, split_seeds, test_frac, metric)

    _print_results(test_stats, train_stats, threshold, metric, multi_run)

    # K is stored in the "radius" column to reuse the shared CSV schema
    if not results_csv.exists():
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=_CSV_COLUMNS).to_csv(results_csv, index=False)

    row = _build_row(
        dataset_file=pkl_path.name,
        distance_fn="ddtw",
        radius=K,  # store K in the radius column
        n_segment_ids=data["segment_id"].nunique(),
        n_ids=data["id"].nunique(),
        n_calculated_pairs=len(all_distances),
        duration=meta.get("duration"),
        trainride_start_seconds=meta.get("trainride_start_seconds"),
        window_size=meta.get("rolling_window"),
        sampling_rate=meta.get("sampling_rate"),
        threshold=threshold,
        test_frac=test_frac,
        seed=seed,
        metric=metric,
        n_runs=runs,
        n_workers=n_workers_val,
        test_stats=test_stats,
        train_stats=train_stats,
        total_time=total_time,
        total_cpu_hours=total_cpu_h,
    )
    pd.DataFrame([row], columns=_CSV_COLUMNS).to_csv(results_csv, index=False, mode="a", header=False)
    typer.echo(f"  Results appended to {results_csv}")

    con.close()

if __name__ == "__main__":
    app()
