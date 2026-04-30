import datetime
from typing import Union, Optional, List

import pandas as pd


def get_chunked_dataframe(
        df: pd.DataFrame,
        duration: Union[pd.Timedelta, datetime.timedelta, float, int],
        start_time: Optional[Union[float, int, pd.Timestamp]] = None,
        end_time: Optional[Union[float, int, pd.Timestamp]] = None,
        time_column='timestamp',
) -> List[pd.DataFrame]:
    """
    Split a time series DataFrame into contiguous chunks of a fixed duration.
    """
    if df.empty:
        return []

    chunks = []
    duration_td = pd.to_timedelta(duration)

    if start_time is None:
        current_start = df[time_column].iloc[0]
    else:
        current_start = start_time

    if end_time is None:
        end_time = df[time_column].max()

    # Build fixed windows [start, start + duration] and move by duration.
    # Inclusive end creates minimal overlap (boundary sample) between neighbors.
    while (current_start + duration_td) <= end_time:
        current_end = current_start + duration_td
        chunk = df[(df[time_column] >= current_start) & (df[time_column] <= current_end)].copy()
        if not chunk.empty:
            chunks.append(chunk)
        current_start = current_end

    return chunks
