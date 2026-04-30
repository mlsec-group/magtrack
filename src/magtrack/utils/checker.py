from typing import Sequence, List

import numpy as np
import pandas as pd


class LowSamplingRateException(Exception):
    def __init__(self, message: str = ""):
        self.msg = message
        super().__init__(message)


class HighDistanceException(Exception):
    def __init__(self, message: str = ""):
        self.msg = message
        super().__init__(message)


class EmptyDataframeException(Exception):
    def __init__(self, message: str = ""):
        self.msg = message
        super().__init__(message)


def check_sampling_rate(df: pd.DataFrame, min_frequency: float, max_difference: float = 3,
                        column: str = 'timestamp') -> bool:
    """
    Checks if the sampling rate of the data in the DataFrame meets the minimum frequency requirement.
    Further there is a check that the distance between two timestamps does not exceed triple the expected interval.

    Parameters:
    - df: pandas DataFrame, the DataFrame containing the data
    - min_frequency: float, the minimum required frequency in Hz
    - max_difference: float, multiplier for the allowed maximum gap relative to the expected interval (default: 3)
    - column: str, timestamp column name (default: 'timestamp')

    Returns:
    - bool, True if the sampling rate check is passed, False otherwise
    """
    if df.empty:
        raise EmptyDataframeException("DataFrame is empty.")
    expected_interval = 1 / min_frequency
    time_diff = pd.Series(df[column].diff().dt.total_seconds())
    mean_interval = time_diff.mean()

    # Guard against NaN or zero mean interval which would lead to division-by-zero
    if pd.isna(mean_interval) or mean_interval <= 0:
        raise LowSamplingRateException(
            f"Unable to determine sampling frequency from timestamps (mean interval={mean_interval}).")

    actual_frequency = 1.0 / mean_interval

    if actual_frequency < min_frequency:
        raise LowSamplingRateException(
            f"Sampling rate is too low. Actual frequency: {actual_frequency} Hz, Minimum required: {min_frequency} Hz")

    if time_diff.max() > max_difference * expected_interval:
        raise HighDistanceException(
            f"Maximum time difference between samples exceeds triple the expected interval: {time_diff.max()} seconds")

    return True


def split_df_chunks_by_mask(
        chunks_df: pd.DataFrame,
        mask: Sequence,
        data_column: str = "data",
        segment_column: str = "subsegment_id",
) -> pd.DataFrame:
    """
    Split a DataFrame of chunk-DataFrames into consecutive True-mask subsegments.

    Input:
    - chunks_df: DataFrame with one row per chunk; `data_column` stores a pd.DataFrame chunk
    - mask: sequence of truthy/falsy values

    Output:
    - DataFrame containing only rows where mask is True (trimmed to min length of input/mask),
      plus `segment_column` that labels consecutive True runs as 0, 1, 2, ...
    """
    if chunks_df is None or chunks_df.empty:
        out_cols = list(chunks_df.columns) if isinstance(chunks_df, pd.DataFrame) else []
        return pd.DataFrame(columns=out_cols + [segment_column])

    if data_column not in chunks_df.columns:
        raise KeyError(f"Column '{data_column}' not found in chunks_df.")

    # Treat NA as False explicitly
    mask_series = pd.Series(mask).fillna(False).astype(bool)
    n = min(len(chunks_df), len(mask_series))
    if n == 0:
        return pd.DataFrame(columns=list(chunks_df.columns) + [segment_column])

    work = chunks_df.iloc[:n].copy()
    mask_arr = mask_series.to_numpy(dtype=bool)[:n]

    true_idx = np.flatnonzero(mask_arr)
    if true_idx.size == 0:
        return pd.DataFrame(columns=list(chunks_df.columns) + [segment_column])

    breaks = np.where(np.diff(true_idx) != 1)[0] + 1
    groups = np.split(true_idx, breaks)

    segment_ids = np.full(n, -1, dtype=int)
    for seg_id, grp in enumerate(groups):
        if grp.size > 0:
            segment_ids[grp] = seg_id

    work[segment_column] = segment_ids
    return work[work[segment_column] >= 0].reset_index(drop=True)


def check_sampling_rate_chunks(df_list: Sequence[pd.DataFrame], min_frequency: float, max_difference: float = 3,
                               column: str = 'timestamp', min_consecutive: int = 0) -> List[bool]:
    """
    Run sampling-rate checks on a sequence of DataFrame chunks and enforce a minimum consecutive-good-chunks rule.

    Behavior:
    - For each DataFrame in `df_list`, `check_sampling_rate` is called. If it returns True the chunk is considered
      initially valid; if it raises any exception the chunk is considered invalid (False).
    - After the per-chunk checks, contiguous runs of True values shorter than `min_consecutive` are converted to False.
    - Special case: if `min_consecutive == 0`, then all values must be True to pass — i.e. if any chunk failed initially,
      the function returns a list of all False values. If all chunks passed, the function returns all True values.

    Parameters:
    - df_list: Sequence[pd.DataFrame] - list/sequence of DataFrame chunks to validate
    - min_frequency: float - minimum required sampling frequency (Hz)
    - max_difference: float - multiplier for allowed maximum gap relative to expected interval (default: 3)
    - column: str - timestamp column name in the DataFrames (default: 'timestamp')
    - min_consecutive: int - minimum length of consecutive True runs to keep; if 0 all values must be True

    Returns:
    - List[bool] - boolean list with the same length as `df_list` indicating which chunks are accepted after
      enforcing the `min_consecutive` rule.

    Raises:
    - TypeError if df_list is not a sequence
    """
    if not isinstance(df_list, (list, tuple)):
        raise TypeError("df_list must be a list or tuple of pandas.DataFrame")

    n = len(df_list)
    if n == 0:
        return []

    if min_consecutive is None:
        min_consecutive = 0
    try:
        min_consecutive = int(min_consecutive)
    except Exception:
        raise TypeError("min_consecutive must be an integer")
    if min_consecutive < 0:
        min_consecutive = 0

    initial: List[bool] = []
    for df in df_list:
        try:
            check_sampling_rate(df, min_frequency, max_difference, column)
            initial.append(True)
        except (EmptyDataframeException, LowSamplingRateException, HighDistanceException):
            initial.append(False)

    if min_consecutive == 0:
        all_ok = all(initial)
        return [all_ok] * n

    final = initial.copy()
    i = 0
    while i < n:
        if not initial[i]:
            i += 1
            continue
        j = i
        while j < n and initial[j]:
            j += 1
        run_len = j - i
        if run_len < min_consecutive:
            for k in range(i, j):
                final[k] = False
        i = j

    return final
