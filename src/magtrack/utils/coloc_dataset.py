import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def _extract_magnitude_chunk(args):
    """Top-level worker for parallel signal stacking (must be picklable)."""
    rows, magnitude_key = args
    return np.stack([r[magnitude_key].values for r in rows]).astype(np.float32)


def _stack_magnitudes(rows: list, magnitude_key: str, n_jobs: int) -> np.ndarray:
    """Stack per-row magnitude vectors into a contiguous (N, L) float32 array.

    With `n_jobs > 1` and a sufficiently large dataset, splits the rows into
    `n_jobs` chunks and processes them in a spawn-based ProcessPoolExecutor.
    Spawn (not fork) is used because callers may have already initialized
    CUDA in the parent process.
    """
    n = len(rows)
    if n_jobs <= 1 or n < 1000:
        return np.ascontiguousarray(
            np.stack([r[magnitude_key].values for r in rows]).astype(np.float32)
        )

    chunk_size = max(1, (n + n_jobs - 1) // n_jobs)
    args = [(rows[i:i + chunk_size], magnitude_key) for i in range(0, n, chunk_size)]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        pieces = list(ex.map(_extract_magnitude_chunk, args))
    return np.ascontiguousarray(np.concatenate(pieces, axis=0))


class ColocDataset(Dataset):
    """Build chunk-aligned recording pairs with colocated labels.

    Label semantics:
      - 0: colocated/similar (same id_col)
      - 1: non-colocated/dissimilar (different id_col)
    """

    def __init__(
            self,
            df: pd.DataFrame,
            data_col: str = "data",
            magnitude_key: str = "magnitude",
            id_col: str = "id",
            random_state: int = 2323,
            neg_pos_ratio: int = 1,  # Number of negative pairs per positive pair
    ):
        df = df.reset_index(drop=True)
        self.rng = np.random.default_rng(random_state)

        # Pre-stack all magnitude vectors into one contiguous float32 array so
        # __getitem__ is pure numpy indexing (no pandas access in workers).
        sigs = [row[magnitude_key].values for row in df[data_col].to_list()]
        self.signals = np.ascontiguousarray(np.stack(sigs).astype(np.float32))
        self.ids = pd.factorize(df[id_col], sort=False)[0].astype(np.int64)
        self.n = len(df)

        grouped: dict[int, list[int]] = {}
        for idx, cid in enumerate(self.ids):
            grouped.setdefault(int(cid), []).append(idx)
        self.unique_ids = list(grouped.keys())

        positives = []
        for arr in grouped.values():
            if len(arr) > 1:
                positives.extend(combinations(arr, 2))
        self.positive_pairs = np.asarray(positives, dtype=np.int64) if positives else np.empty((0, 2), dtype=np.int64)
        self.num_positives = len(self.positive_pairs)

        self.num_negatives = self.num_positives * neg_pos_ratio

    def __len__(self):
        return self.num_positives + self.num_negatives

    def __getitem__(self, idx):
        if idx < self.num_positives:
            i, j = self.positive_pairs[idx]
            label = np.int8(0)
        else:
            i = int(self.rng.integers(self.n))
            j = int(self.rng.integers(self.n))
            while self.ids[i] == self.ids[j]:
                j = int(self.rng.integers(self.n))
            label = np.int8(1)

        return self.signals[i], self.signals[j], label


class ColocSequentialEvalDataset(Dataset):
    """Deterministic evaluation dataset over all sample pairs (not just ID groups).

    The dataset iterates sequentially through all pairs (i, j) with i < j for all samples.
    This ensures comprehensive evaluation across the entire dataset.
    
    Label semantics:
      - 0: same id_col (positive/colocated)
      - 1: different id_col (negative/non-colocated)
    """

    def __init__(
            self,
            df: pd.DataFrame,
            data_col: str = "data",
            magnitude_key: str = "magnitude",
            id_col: str = "id",
            n_jobs: int = 1,
    ):
        df = df.reset_index(drop=True)
        self.n = len(df)

        # Stack signals — optionally in parallel for large datasets.
        rows = df[data_col].to_list()
        self.signals = _stack_magnitudes(rows, magnitude_key, n_jobs)

        self.ids = pd.factorize(df[id_col], sort=False)[0].astype(np.int64)

        # Vectorized: all (i, j) pairs with i <= j (combinations_with_replacement).
        i_idx, j_idx = np.triu_indices(self.n, k=1)
        self.pairs = np.stack([i_idx, j_idx], axis=1).astype(np.int64)

        #  Vectorized labels: 0 if same id (positive), 1 if different (negative).
        self.labels = (self.ids[i_idx] != self.ids[j_idx]).astype(np.int8)
       
        positive_mask = (self.labels == 0)
        self.positive_pairs = self.pairs[positive_mask]
        self.num_positives = len(self.positive_pairs)

        negative_mask = (self.labels == 1)
        self.negative_pairs = self.pairs[negative_mask]
        self.num_negatives = len(self.negative_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.signals[i], self.signals[j], self.labels[idx]
