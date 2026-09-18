import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from magtrack.utils.fast_ddtw import fast_ddtw
from magtrack.utils.fast_dtw import fast_dtw_distance


def euclidean_distance(a: pd.DataFrame, b: pd.DataFrame, **kwargs) -> float:
    """Euclidean distance."""
    return float(np.linalg.norm(a - b))


def cosine_distance(a, b, **kwargs) -> float:
    """1 - cosine similarity."""
    return float(cosine(a, b))


def dtw_distance(a, b, radius=1, **kwargs) -> float:
    """Dynamic Time Warping distance (robust to local time shifts).

    Uses the in-tree reimplementation, which returns the same value as
    ``fastdtw.fastdtw(a, b, radius)[0]`` but avoids that package's pure-Python
    dynamic-programming table.
    """
    return fast_dtw_distance(a, b, radius=radius)


def ddtw_distance(signal_1, signal_2, K=10, **kwargs) -> float:
    """Derivative Dynamic Time Warping distance (robust to local time shifts)."""
    distance, path = fast_ddtw(signal_1, signal_2, K=K, debug=False)
    return float(distance)
