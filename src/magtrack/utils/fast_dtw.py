"""Fast, bit-for-bit compatible reimplementation of ``fastdtw.fastdtw`` by claude.

The upstream package ships an optional C extension, but it is not built in our
image, so ``fastdtw`` falls back to its pure-Python version.  That version keeps
the dynamic-programming table in a ``defaultdict`` keyed by ``(i, j)`` tuples and
picks the predecessor with ``min(..., key=lambda a: a[0])``, which costs three
dict lookups, three tuple allocations and three lambda calls per cell.  At
~50 us per input sample it dominates the colocation evaluation completely.

This module computes the *same* number by exploiting a property of the upstream
algorithm: ``__expand_window`` emits, for every row ``i``, a single contiguous
run of columns (it scans ``j`` upwards from the previous row's first hit and
stops at the first gap).  The window can therefore be represented as one
``[start, end]`` column range per row, which turns the DP into a flat scan over
Python floats and the window expansion into a sliding min/max over the coarse
path.

Everything that can affect the result is preserved:

* ``_reduce_by_half`` performs the same float64 arithmetic;
* the DP compares the three *summed* candidates in the upstream order
  (up, then left, then diagonal), so ties resolve to the same predecessor and
  the coarse path -- and hence the next window -- is identical;
* cells outside the window read as ``inf``, matching the ``defaultdict``.

Only the distance is returned; the caller discards the path.
"""

from __future__ import annotations

import numpy as np

_INF = float("inf")

# Predecessor codes, in the priority order used by upstream's ``min``.
_UP, _LEFT, _DIAG = 0, 1, 2


def _reduce_by_half(x: np.ndarray) -> np.ndarray:
    """Same values as upstream's ``[(x[i] + x[i+1]) / 2 for i in ...]``."""
    n = len(x) - (len(x) & 1)
    return (x[0:n:2] + x[1:n:2]) / 2


def _sliding_min(a: np.ndarray, radius: int) -> np.ndarray:
    """Minimum over ``a[i - radius : i + radius + 1]``, clipped at the edges."""
    out = a.copy()
    for k in range(1, radius + 1):
        np.minimum(out[:-k], a[k:], out=out[:-k])
        np.minimum(out[k:], a[:-k], out=out[k:])
    return out


def _sliding_max(a: np.ndarray, radius: int) -> np.ndarray:
    """Maximum over ``a[i - radius : i + radius + 1]``, clipped at the edges."""
    out = a.copy()
    for k in range(1, radius + 1):
        np.maximum(out[:-k], a[k:], out=out[:-k])
        np.maximum(out[k:], a[:-k], out=out[k:])
    return out


def _expand_window_rows(
        path_min: np.ndarray,
        path_max: np.ndarray,
        len_x: int,
        len_y: int,
        radius: int,
) -> tuple[list[int], list[int]]:
    """Per-row column ranges of upstream's ``__expand_window``.

    ``path_min``/``path_max`` are the smallest and largest column the coarse
    path visits in each coarse row.  Upstream grows every path cell by
    ``radius`` in both directions, doubles the result and then walks the rows in
    order, keeping only the first contiguous run at or after the previous row's
    first hit.  Because the coarse path is monotone, the grown cells of a coarse
    row form one interval, so the doubled interval is contiguous too.
    """
    n_coarse = len(path_min)
    lo_coarse = _sliding_min(path_min, radius)
    hi_coarse = _sliding_max(path_max, radius)

    # ``len_x`` odd means ``_reduce_by_half`` dropped the last sample, so the
    # final fine row maps to a coarse row one past the end of the path.
    n_rows_needed = ((len_x - 1) >> 1) + 1
    if n_rows_needed > n_coarse:
        edge = max(0, n_coarse - radius)
        lo_coarse = np.append(lo_coarse, path_min[edge:n_coarse].min())
        hi_coarse = np.append(hi_coarse, path_max[edge:n_coarse].max())

    lo_fine = (lo_coarse - radius) * 2
    hi_fine = (hi_coarse + radius) * 2 + 1
    lo_list = lo_fine.tolist()
    hi_list = hi_fine.tolist()

    last_col = len_y - 1
    starts: list[int] = [0] * len_x
    ends: list[int] = [0] * len_x
    start_j = 0
    for i in range(len_x):
        coarse = i >> 1
        lo = lo_list[coarse]
        hi = hi_list[coarse]
        if lo < start_j:
            lo = start_j
        if hi > last_col:
            hi = last_col
        if lo > hi:
            # Upstream would find no cell in this row and crash on the next one;
            # a monotone path never gets here, but keep the window usable.
            lo = hi
        starts[i] = lo
        ends[i] = hi
        start_j = lo
    return starts, ends


def _dtw_windowed(
        x: list[float],
        y: list[float],
        starts: list[int],
        ends: list[int],
        want_path: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Windowed DTW over contiguous per-row column ranges.

    Returns ``(distance, path_min, path_max)``; the path bounds are ``None``
    unless ``want_path``.
    """
    len_x = len(x)
    len_y = len(y)

    dirs = bytearray() if want_path else None
    offsets = [0] * len_x if want_path else None

    # Virtual row -1 holding only the origin, upstream's ``D[0, 0] = 0``.
    prev: list[float] = [0.0]
    prev_s = -1
    prev_e = -1

    cur: list[float] = []
    for i in range(len_x):
        s = starts[i]
        e = ends[i]
        width = e - s + 1

        # Previous-row values for columns [s-1, e], padded with inf so the
        # inner loop needs no bounds checks.
        pv = [_INF] * (width + 1)
        lo = s - 1
        if lo < prev_s:
            lo = prev_s
        hi = e
        if hi > prev_e:
            hi = prev_e
        if lo <= hi:
            pv[lo - s + 1:hi - s + 2] = prev[lo - prev_s:hi - prev_s + 1]

        xi = x[i]
        ys = y[s:e + 1]
        cur = [0.0] * width
        left = _INF

        if want_path:
            offsets[i] = len(dirs)
            row_dirs = bytearray(width)
            for k in range(width):
                d = xi - ys[k]
                if d < 0.0:
                    d = -d
                a_up = pv[k + 1] + d
                a_left = left + d
                a_diag = pv[k] + d
                if a_up <= a_left:
                    if a_up <= a_diag:
                        left = a_up
                        row_dirs[k] = _UP
                    else:
                        left = a_diag
                        row_dirs[k] = _DIAG
                elif a_left <= a_diag:
                    left = a_left
                    row_dirs[k] = _LEFT
                else:
                    left = a_diag
                    row_dirs[k] = _DIAG
                cur[k] = left
            dirs += row_dirs
        else:
            for k in range(width):
                d = xi - ys[k]
                if d < 0.0:
                    d = -d
                a_up = pv[k + 1] + d
                a_left = left + d
                a_diag = pv[k] + d
                if a_up <= a_left:
                    left = a_up if a_up <= a_diag else a_diag
                elif a_left <= a_diag:
                    left = a_left
                else:
                    left = a_diag
                cur[k] = left

        prev = cur
        prev_s = s
        prev_e = e

    # Upstream reads D[len_x, len_y] straight out of the defaultdict, so a
    # bottom-right cell outside the window yields inf.
    distance = cur[-1] if (len_x and ends[len_x - 1] == len_y - 1) else _INF

    if not want_path:
        return distance, None, None

    path_min = np.empty(len_x, dtype=np.int64)
    path_max = np.empty(len_x, dtype=np.int64)
    seen = bytearray(len_x)
    i = len_x - 1
    j = len_y - 1
    while i >= 0 and j >= 0:
        if not seen[i]:
            seen[i] = 1
            path_max[i] = j
        path_min[i] = j
        code = dirs[offsets[i] + j - starts[i]]
        if code == _UP:
            i -= 1
        elif code == _LEFT:
            j -= 1
        else:
            i -= 1
            j -= 1
    return distance, path_min, path_max


def _fast_dtw(
        x: np.ndarray,
        y: np.ndarray,
        radius: int,
        want_path: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    min_time_size = radius + 2
    len_x = len(x)
    len_y = len(y)

    if len_x < min_time_size or len_y < min_time_size:
        # Upstream's base case is an unwindowed DTW over the full grid.
        return _dtw_windowed(
            x.tolist(), y.tolist(), [0] * len_x, [len_y - 1] * len_x, want_path)

    _, path_min, path_max = _fast_dtw(
        _reduce_by_half(x), _reduce_by_half(y), radius, True)
    starts, ends = _expand_window_rows(path_min, path_max, len_x, len_y, radius)
    return _dtw_windowed(x.tolist(), y.tolist(), starts, ends, want_path)


def fast_dtw_distance(x, y, radius: int = 1) -> float:
    """Approximate DTW distance, identical to ``fastdtw.fastdtw(x, y, radius)[0]``."""
    x = np.asanyarray(x, dtype="float")
    y = np.asanyarray(y, dtype="float")
    distance, _, _ = _fast_dtw(x, y, radius, False)
    return float(distance)
