import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, butter


def _get_sampling_rate(timestamps) -> float:
    """
    Estimate sampling frequency (Hz) from an array of timestamps.

    Accepts:
      - numeric (float/int) timestamps in seconds
      - numpy/pandas datetime-like timestamps (datetime64[ns] or pandas Timestamps)

    Returns:
      sampling frequency in Hz
    Raises:
      ValueError if there are fewer than 2 timestamps or the computed dt is 0.
    """
    ts = np.asarray(timestamps)
    if ts.size < 2:
        raise ValueError("Not enough data points to calculate sampling frequency.")

    # Numeric timestamps
    if np.issubdtype(ts.dtype, np.floating) or np.issubdtype(ts.dtype, np.integer):
        diffs = np.diff(ts).astype(float)
        dt = np.median(diffs)
    else:
        try:
            diffs_ns = np.diff(ts.astype('datetime64[ns]')).astype('timedelta64[ns]').astype(float)
            dt = np.median(diffs_ns) * 1e-9
        except Exception:
            diffs_ns = np.diff(pd.to_datetime(ts).values).astype('timedelta64[ns]').astype(float)
            dt = np.median(diffs_ns) * 1e-9

    if dt == 0 or np.isnan(dt) or np.isclose(dt, 0.0):
        raise ValueError("Calculated time delta is zero or invalid; cannot determine sampling frequency.")

    return 1.0 / dt


def _sliding_window(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Apply a centered rolling mean on the 'magnitude' column and drop NaNs produced by the window.
    If window_size <= 1 the input is returned unchanged (no copy) to avoid unnecessary memory use.

    Args:
        data: DataFrame with at least ['timestamp','magnitude'] columns.
        window_size: window size for pandas rolling.
    Returns:
        DataFrame with ['timestamp','magnitude'] where magnitude is smoothed and NaNs removed.
    """
    if window_size is None or window_size <= 1:
        return data  # return original object, no changes

    # Operate in-place to avoid allocating a copy of the whole DataFrame.
    data['magnitude'] = data['magnitude'].rolling(window=window_size, center=True).mean()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


_EPOCH = pd.Timestamp("1970-01-01")


def _downsample(
        data: pd.DataFrame,
        sampling_rate: float,
        origin: pd.Timedelta = None,
        duration: pd.Timedelta = None,
) -> pd.DataFrame:
    """
    Downsample a DataFrame to a uniform sampling rate.

    Args:
        data: DataFrame with columns ['timestamp' (timedelta), 'magnitude'].
        sampling_rate: target sampling rate in Hz.
        origin: optional timedelta anchor; two calls with the same origin produce identical output timestamps.
        duration: optional chunk duration used to enforce exact output length.
    Returns:
        DataFrame with ['timestamp' (timedelta), 'magnitude'] at the requested sampling_rate.
    """
    if data is None or len(data) == 0:
        raise ValueError("Cannot downsample empty data.")

    tgt_sample_ns = int(round(1_000_000_000 / sampling_rate))
    tgt_sample = pd.to_timedelta(tgt_sample_ns, unit='ns')
    timestamp_ns = pd.to_timedelta(data['timestamp']).astype('timedelta64[ns]')
    work = data.copy()
    work['timestamp'] = _EPOCH + timestamp_ns
    origin_ts = (_EPOCH + pd.to_timedelta(origin).as_unit('ns')) if origin is not None else None

    kwargs = {'origin': origin_ts} if origin_ts is not None else {}
    resampled = work.resample(tgt_sample, on='timestamp', **kwargs).mean()

    if duration is not None:
        duration_td = pd.to_timedelta(duration)
        target_len = int(round(duration_td / tgt_sample))
        if target_len > 0:
            if origin_ts is not None:
                target_start = origin_ts
            elif len(resampled.index) > 0:
                target_start = resampled.index[0]
            else:
                target_start = _EPOCH
            target_index = pd.date_range(start=target_start, periods=target_len, freq=tgt_sample)
            resampled = resampled.reindex(target_index)
            if resampled['magnitude'].isna().any():
                raise ValueError(
                    f"Too little data for synchronized downsampling: need {target_len} samples over {duration_td}."
                )

    resampled = resampled.reset_index().rename(columns={'index': 'timestamp'})
    resampled['timestamp'] = resampled['timestamp'] - _EPOCH
    return resampled[['timestamp', 'magnitude']]


def _normalize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the 'magnitude' column to zero mean and unit variance in-place.
    If the standard deviation is zero (constant signal) it subtracts the mean and leaves values as zero.

    Args:
        data: DataFrame with a 'magnitude' column. Modified in-place and returned.
    Returns:
        The same DataFrame object with normalized 'magnitude'.
    """
    if data is None or len(data) == 0:
        return data
    mag = data['magnitude']
    mean = mag.mean()
    std = mag.std()
    # Guard against zero std
    if std == 0 or np.isnan(std) or np.isclose(std, 0.0):
        data['magnitude'] = mag - mean
    else:
        data['magnitude'] = (mag - mean) / std
    return data


def filter_abs(
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        **kwargs,
) -> pd.DataFrame:
    """
    Calculate the absolute magnitude from x, y, z columns and return a DataFrame with timestamp and magnitude.
    Args:
        data: pandas.DataFrame with columns ['timestamp', 'x', 'y', 'z'].
        timestamp_column: name of the timestamp column in *data* (default ``"timestamp"``).
    Returns:
        pandas.DataFrame: columns ['timestamp', 'magnitude'].
    """
    mag = np.sqrt(data['x'].values ** 2 + data['y'].values ** 2 + data['z'].values ** 2)
    df = pd.DataFrame({
        'timestamp': data[timestamp_column],
        'magnitude': mag
    })
    return df


def filter_notch_highpass(
        data: pd.DataFrame,
        notch_freq: float = 16.7,
        Q: int = 1,
        highpass_cutoff: float = 0.1,
        timestamp_column: str = "timestamp",
        **kwargs,
) -> pd.DataFrame:
    """
    Remove DC/static field, notch out 16.7 Hz interference, and normalize.

    Args:
        data: pandas.DataFrame with columns ['timestamp', 'x', 'y', 'z'].
        notch_freq: Frequency to notch out (default 16.7 Hz).
        Q: Quality factor for notch filter.
        highpass_cutoff: Cutoff frequency for high-pass filter (Hz).
        timestamp_column: name of the timestamp column in *data* (default ``"timestamp"``).

    Returns:
        pd.DataFrame: ['timestamp', 'magnitude'].
    """
    timestamps = data[timestamp_column].values
    fs = _get_sampling_rate(timestamps)

    # High-pass filter to remove DC / slow drift
    hp_b, hp_a = butter(3, highpass_cutoff / (fs / 2), btype="highpass")

    def process_axis(x: np.ndarray) -> np.ndarray:
        x_hp = filtfilt(hp_b, hp_a, x)
        b_notch, a_notch = iirnotch(notch_freq, Q, fs)
        return filtfilt(b_notch, a_notch, x_hp)

    x_filt = process_axis(data["x"].values)
    y_filt = process_axis(data["y"].values)
    z_filt = process_axis(data["z"].values)

    mag = np.sqrt(x_filt ** 2 + y_filt ** 2 + z_filt ** 2)

    return pd.DataFrame({"timestamp": data[timestamp_column], "magnitude": mag})
