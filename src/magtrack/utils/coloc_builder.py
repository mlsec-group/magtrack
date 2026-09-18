import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from magtrack.utils.filters import _sliding_window
from magtrack.utils.loader import get_data_from_zipfile_by_name

_SAMPLE_COLUMNS = pd.Index(['timestamp', 'magnitude'])


class Trace:
    """A filtered magnetometer trace plus the arrays derived from it."""

    __slots__ = ('df', 'ts_ns', 'magnitude', 'diff_seconds', 'monotonic')

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.ts_ns = df['timestamp'].to_numpy().astype('timedelta64[ns]').astype('int64')
        self.magnitude = df['magnitude'].to_numpy()
        self.diff_seconds = df['timestamp'].diff().dt.total_seconds().to_numpy()
        self.monotonic = bool(np.all(np.diff(self.ts_ns) >= 0)) if len(self.ts_ns) > 1 else True

    def chunk_diff_seconds(self, rows) -> np.ndarray:
        """``chunk['timestamp'].diff().dt.total_seconds()`` without the leading NaN."""
        if isinstance(rows, slice):
            return self.diff_seconds[rows.start + 1:rows.stop]
        return self.df['timestamp'].iloc[rows].diff().dt.total_seconds().to_numpy()[1:]


def first_row(rows) -> int:
    """Position of the first row of a chunk, in trace order."""
    return rows.start if isinstance(rows, slice) else int(rows[0])


def row_count(rows) -> int:
    return rows.stop - rows.start if isinstance(rows, slice) else int(rows.size)


def _extract_trace(traintrack_basepath, zip_file, sensor_type, extraction_function) -> pd.DataFrame:
    """Decode and filter one recording (everything except the rolling window)."""
    zip_file_path = os.path.join(traintrack_basepath, zip_file + '.zip')
    magnetometer_recording = get_data_from_zipfile_by_name(zip_file_path, sensor_type + '.csv')

    magnetometer_trace = extraction_function(magnetometer_recording,
                                             timestamp_column='seconds_from_journey_start')
    magnetometer_trace['timestamp'] = magnetometer_trace['timestamp'].apply(lambda x: timedelta(seconds=x))
    return magnetometer_trace


class TraceLoader:
    """Loads filtered traces, optionally backed by an on-disk cache.

    The cached representation stores the exact ``int64`` microseconds of the
    timestamps and the raw ``float64`` magnitudes, so a cached trace is
    identical to a freshly decoded one.
    """

    def __init__(self, traintrack_basepath, sensor_type, extraction_function,
                 extraction_function_name, cache_dir=None):
        self.traintrack_basepath = traintrack_basepath
        self.sensor_type = sensor_type
        self.extraction_function = extraction_function
        self.extraction_function_name = extraction_function_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, zip_file) -> Path:
        return self.cache_dir / f"{zip_file}.{self.sensor_type}.{self.extraction_function_name}.npz"

    def load(self, zip_file) -> pd.DataFrame:
        if self.cache_dir is None:
            return _extract_trace(self.traintrack_basepath, zip_file, self.sensor_type,
                                  self.extraction_function)

        cache_file = self._cache_file(zip_file)
        if cache_file.exists():
            try:
                with np.load(cache_file) as cached:
                    timestamp_us = cached['timestamp_us']
                    magnitude = cached['magnitude']
                return pd.DataFrame({
                    'timestamp': timestamp_us.astype('timedelta64[us]'),
                    'magnitude': magnitude,
                })
            except Exception as exc:  # corrupt/partial cache entry: rebuild it
                tqdm.write(f"Ignoring unusable trace cache {cache_file}: {exc}")

        trace = _extract_trace(self.traintrack_basepath, zip_file, self.sensor_type,
                               self.extraction_function)
        tmp_file = cache_file.with_suffix(f".{os.getpid()}.tmp.npz")
        try:
            np.savez(
                tmp_file,
                timestamp_us=trace['timestamp'].to_numpy().astype('timedelta64[us]').astype('int64'),
                magnitude=trace['magnitude'].to_numpy(),
            )
            os.replace(tmp_file, cache_file)
        except OSError as exc:
            tqdm.write(f"Could not write trace cache {cache_file}: {exc}")
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
        return trace


def apply_rolling_window(trace: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Return the smoothed trace for ``rolling_window``.

    ``_sliding_window`` smooths in place, so a copy is required whenever it
    actually modifies the frame.
    """
    if rolling_window is None or rolling_window <= 1:
        return trace
    return _sliding_window(trace.copy(), window_size=rolling_window)


def as_ns(value) -> int:
    """Exact nanoseconds of a ``timedelta``-like value."""
    return int(pd.to_timedelta(value).as_unit('ns').value)


def chunk_windows(trace: Trace, duration_ns: int, start_ns: int, end_ns: int):
    """Chunk boundaries of :func:`get_chunked_dataframe` as ``(end_ns, rows)``.

    ``rows`` selects the rows of the chunk in trace order: a ``slice`` found by
    binary search for the usual sorted trace, and an array of row positions for
    a trace whose timestamps are not sorted.  A recording can end in a malformed
    row whose timestamp is far outside the journey, and ``_sliding_window`` only
    hides it when its rolling mean trims that row away, so the unsorted case has
    to be handled rather than rejected.  Empty windows are dropped and rows keep
    their original order, exactly like the boolean selection of the reference
    implementation.  All boundaries are exact in nanoseconds, so the integer
    comparisons match its ``Timedelta`` comparisons.
    """
    ts_ns = trace.ts_ns
    if len(ts_ns) == 0:
        return []

    monotonic = trace.monotonic
    windows = []
    current_start = start_ns
    while (current_start + duration_ns) <= end_ns:
        current_end = current_start + duration_ns
        if monotonic:
            lo = int(np.searchsorted(ts_ns, current_start, side='left'))
            hi = int(np.searchsorted(ts_ns, current_end, side='right'))
            rows = slice(lo, hi) if hi > lo else None
        else:
            selected = np.flatnonzero((ts_ns >= current_start) & (ts_ns <= current_end))
            rows = selected if selected.size else None
        if rows is not None:
            windows.append((current_end, rows))
        current_start = current_end
    return windows


def chunk_sampling_ok(trace: Trace, rows, min_frequency: float,
                      max_difference: float = 3) -> bool:
    """Vectorised equivalent of :func:`magtrack.utils.checker.check_sampling_rate`.

    ``check_sampling_rate`` computes ``df['timestamp'].diff().dt.total_seconds()``
    and takes its mean/max.  ``pandas`` replaces the leading ``NaN`` by ``0``
    before summing and divides by the number of non-NaN values, which is what is
    reproduced here so that the result is bit-identical.
    """
    n = row_count(rows)
    if n == 0:
        return False  # EmptyDataframeException
    expected_interval = 1 / min_frequency

    if n > 1:
        time_diff = np.empty(n, dtype=np.float64)
        time_diff[0] = 0.0
        time_diff[1:] = trace.chunk_diff_seconds(rows)
        mean_interval = time_diff.sum(dtype=np.float64) / float(n - 1)
    else:
        mean_interval = np.nan

    if np.isnan(mean_interval) or mean_interval <= 0:
        return False  # LowSamplingRateException
    if 1.0 / mean_interval < min_frequency:
        return False  # LowSamplingRateException
    # Only reachable for n > 1: a single-row chunk has a NaN mean and returned above.
    if time_diff[1:].max() > max_difference * expected_interval:
        return False  # HighDistanceException
    return True


def apply_min_consecutive(initial, min_consecutive: int = 0):
    """Boolean post-processing of :func:`check_sampling_rate_chunks`."""
    n = len(initial)
    if n == 0:
        return []
    if min_consecutive == 0:
        return [all(initial)] * n

    final = list(initial)
    i = 0
    while i < n:
        if not initial[i]:
            i += 1
            continue
        j = i
        while j < n and initial[j]:
            j += 1
        if j - i < min_consecutive:
            for k in range(i, j):
                final[k] = False
        i = j
    return final


def subsegment_runs(mask):
    """Consecutive ``True`` runs of ``mask`` as ``(subsegment_id, [chunk_id, ...])``.

    Mirrors :func:`magtrack.utils.checker.split_df_chunks_by_mask` followed by
    ``groupby('subsegment_id', sort=True).sort_values('chunk_id')``.
    """
    true_idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if true_idx.size == 0:
        return []
    breaks = np.where(np.diff(true_idx) != 1)[0] + 1
    return list(enumerate(np.split(true_idx, breaks)))


def grouped_means(values: np.ndarray, labels: np.ndarray, n_labels: int):
    """Per-label mean, using the same code path (and summation) as ``resample().mean()``.

    Labels without any sample keep ``NaN``, which is how ``resample`` reports
    empty bins.
    """
    means = np.full(n_labels, np.nan, dtype=np.float64)
    if labels.size == 0:
        return means
    grouped = pd.Series(values).groupby(labels).mean()
    means[grouped.index.to_numpy()] = grouped.to_numpy()
    return means


def sample_frame(timestamps_ns: np.ndarray, magnitude: np.ndarray) -> pd.DataFrame:
    """Build one down-sampled sample frame identical to ``_downsample``'s output."""
    frame = pd.DataFrame({
        'timestamp': timestamps_ns.astype('timedelta64[ns]'),
        'magnitude': magnitude,
    })
    return frame[_SAMPLE_COLUMNS]


def target_sample_ns(sampling_rate: float) -> int:
    return int(round(1_000_000_000 / sampling_rate))


def canonicalize_dtype_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Make every block's dtype object identical to the one of its backing array.

    ``pickle`` memoizes by object identity, so a block whose ``_dtype`` is a
    different (but equal) ``numpy.dtype`` instance than its backing array's
    serializes the descriptor twice instead of once.  Which of the two happens
    depends on allocator state — for parametrized dtypes such as
    ``timedelta64[ns]`` numpy does not intern the descriptor — so writing more
    than one dataset per process would otherwise produce pickles that differ in
    those few bytes from the ones written by a single-dataset process.
    Re-pointing the array at the block's descriptor is a no-op semantically —
    both describe ``timedelta64[ns]`` — and makes the bytes independent of
    allocation history.
    """
    try:
        blocks = df._mgr.blocks
    except AttributeError:
        return df
    for block in blocks:
        values = block.values
        ndarray = getattr(values, '_ndarray', None)
        dtype = getattr(values, '_dtype', None)
        if ndarray is None or dtype is None:
            continue
        if dtype is not ndarray.dtype and dtype == ndarray.dtype:
            try:
                ndarray.dtype = dtype
            except (AttributeError, ValueError):
                pass
    return df


def build_rows(
        dataset_path,
        durations,
        sampling_rates,
        rolling_windows,
        trainride_start_seconds,
        train_types,
        extraction_function,
        extraction_function_name,
        sensor_type='Magnetometer',
        start_delay=2,
        end_buffer=2,
        min_sample_frequency=90.0,
        normalize='trace',
        trace_cache_dir=None,
        skip_short_windows=True,
        progress=True,
):
    """Build the rows of every requested parameter combination in one pass.

    Returns ``(rows_by_combination, types_by_combination)``.  Both dictionaries
    are keyed by ``(rolling_window, duration, trainride_start_seconds,
    sampling_rate)``; ``types_by_combination`` holds the train type of each row so
    that the per-train-type datasets can be split off afterwards.
    """
    traintrack_basepath = os.path.dirname(dataset_path)
    colocated_electric_trips, traintrack_segments = load_traintrack_trips(dataset_path, train_types)

    loader = TraceLoader(traintrack_basepath, sensor_type, extraction_function,
                         extraction_function_name, trace_cache_dir)

    # The grid excludes combinations in which the chunk is not longer than the
    # smoothing window; leaving them out here avoids building rows nobody wants.
    combinations = [
        (rolling_window, duration, start_seconds, sampling_rate)
        for rolling_window in rolling_windows
        for duration in durations
        for start_seconds in trainride_start_seconds
        for sampling_rate in sampling_rates
        if not (skip_short_windows and duration * sampling_rate <= rolling_window)
    ]
    rows_by_combination = {combination: [] for combination in combinations}
    types_by_combination = {combination: [] for combination in combinations}

    trips_iter = colocated_electric_trips.iterrows()
    if progress:
        trips_iter = tqdm(trips_iter, desc="Processing trips", total=len(colocated_electric_trips))

    for _, trip in trips_iter:
        train_type = trip['train_type']
        raw_traces = {}
        for trip_zip_file in trip['zip_files']:
            if trip_zip_file in raw_traces:
                continue
            try:
                raw_traces[trip_zip_file] = loader.load(trip_zip_file)
            except Exception as e:
                tqdm.write(f"Failed to load {trip_zip_file}: {e}. Skipping.")

        trip_segments = traintrack_segments[traintrack_segments["trip_id"].eq(trip["trip_id"])]
        trip_segment_ids = trip_segments['segment'].unique()

        for rolling_window in rolling_windows:
            traces = {zip_file: Trace(apply_rolling_window(raw_trace, rolling_window))
                      for zip_file, raw_trace in raw_traces.items()}

            for duration in durations:
                duration_ns = as_ns(timedelta(seconds=duration))
                for trip_segment_id in trip_segment_ids:
                    segment_rows = trip_segments[trip_segments['segment'].eq(trip_segment_id)]
                    if segment_rows.empty:
                        continue
                    _process_segment(
                        rows_by_combination=rows_by_combination,
                        types_by_combination=types_by_combination,
                        trip=trip,
                        train_type=train_type,
                        segment_rows=segment_rows,
                        traces=traces,
                        rolling_window=rolling_window,
                        duration=duration,
                        duration_ns=duration_ns,
                        trainride_start_seconds=trainride_start_seconds,
                        sampling_rates=sampling_rates,
                        start_delay=start_delay,
                        end_buffer=end_buffer,
                        min_sample_frequency=min_sample_frequency,
                        normalize=normalize,
                    )
            del traces
        del raw_traces

    return rows_by_combination, types_by_combination


def load_traintrack_trips(dataset_path, train_types):
    """Colocated electric trips of the requested train types, in dataset order."""
    with open(dataset_path, 'r') as file:
        traintrack_dataset = yaml.safe_load(file)
    traintrack_trips = pd.DataFrame(traintrack_dataset['trips'])
    traintrack_segments = pd.DataFrame(traintrack_dataset['segments'])

    electric_trips = traintrack_trips[traintrack_trips['electric'] == 1]
    colocated_electric_trips = electric_trips[electric_trips['zip_files'].apply(lambda zip_files: len(zip_files) >= 2)]
    colocated_electric_trips = colocated_electric_trips[colocated_electric_trips['train_type'].isin(train_types)]
    return colocated_electric_trips, traintrack_segments


def warm_trace_cache(dataset_path, train_types, extraction_function, extraction_function_name,
                     sensor_type, trace_cache_dir):
    """Decode and cache every recording needed for ``train_types``.

    Running this once before a parallel grid avoids every worker decoding the
    same recordings at the same time.
    """
    colocated_electric_trips, _ = load_traintrack_trips(dataset_path, train_types)
    loader = TraceLoader(os.path.dirname(dataset_path), sensor_type, extraction_function,
                         extraction_function_name, trace_cache_dir)
    zip_files = list(dict.fromkeys(
        zip_file for zip_files in colocated_electric_trips['zip_files'] for zip_file in zip_files))
    cached = 0
    for zip_file in tqdm(zip_files, desc="Caching traces"):
        try:
            loader.load(zip_file)
            cached += 1
        except Exception as e:
            tqdm.write(f"Failed to load {zip_file}: {e}. Skipping.")
    return cached, len(zip_files)


def _process_segment(rows_by_combination, types_by_combination, trip, train_type, segment_rows, traces, rolling_window,
                     duration, duration_ns, trainride_start_seconds, sampling_rates, start_delay,
                     end_buffer, min_sample_frequency, normalize):
    """Emit the rows of one segment for every ``(start_seconds, sampling_rate)``.

    Chunk boundaries and the per-chunk sampling-rate check only depend on
    ``(rolling_window, duration)``; the chunk list of a positive
    ``trainride_start_seconds`` is a prefix of the one built for the full
    segment, so both are computed once here and reused.
    """
    n_expected_files = segment_rows['zip_file'].nunique()

    # Per zip file: the full chunk list plus everything derived from it.
    master = {}
    seg_id = None
    for _, segment_row in segment_rows.iterrows():
        segment_zip_file = segment_row['zip_file']
        seg_id = f"{str(segment_row.trip_id)}_{segment_row.segment}"
        trace = traces.get(segment_zip_file)
        if trace is None:
            continue

        recording_start_ns = as_ns(timedelta(seconds=segment_row.start_timestamp))
        recording_end_ns = as_ns(timedelta(seconds=segment_row.end_timestamp))
        if len(trace.ts_ns) == 0:
            continue
        if recording_start_ns < trace.ts_ns[0]:
            tqdm.write(f"Recording {seg_id} starts before the magnetometer trace. Skipping this segment.")
            continue

        start_ns = recording_start_ns + as_ns(timedelta(seconds=start_delay))
        full_end_ns = recording_end_ns - as_ns(timedelta(seconds=end_buffer))
        windows = chunk_windows(trace, duration_ns, start_ns, full_end_ns)
        initial_valid = [
            chunk_sampling_ok(trace, chunk_rows, min_sample_frequency)
            for _, chunk_rows in windows
        ]
        master[segment_zip_file] = (trace, recording_start_ns, recording_end_ns, start_ns,
                                    windows, initial_valid)

    if seg_id is None or not master:
        return

    for start_seconds in trainride_start_seconds:
        _process_segment_start(
            rows_by_combination=rows_by_combination,
            types_by_combination=types_by_combination,
            seg_id=seg_id,
            train_type=train_type,
            master=master,
            n_expected_files=n_expected_files,
            rolling_window=rolling_window,
            duration=duration,
            duration_ns=duration_ns,
            start_seconds=start_seconds,
            sampling_rates=sampling_rates,
            start_delay=start_delay,
            end_buffer=end_buffer,
            normalize=normalize,
        )


def _process_segment_start(rows_by_combination, types_by_combination, seg_id, train_type, master, n_expected_files,
                           rolling_window, duration, duration_ns, start_seconds, sampling_rates,
                           start_delay, end_buffer, normalize):
    start_delay_ns = as_ns(timedelta(seconds=start_delay))
    end_buffer_ns = as_ns(timedelta(seconds=end_buffer))
    start_seconds_ns = as_ns(timedelta(seconds=start_seconds))

    # Per zip file: the chunk list for this trainride_start_seconds plus the
    # magnitudes after the requested normalization.
    per_file = {}
    for segment_zip_file, (trace, recording_start_ns, recording_end_ns, start_ns, windows,
                           initial_valid) in master.items():
        if start_seconds > 0:
            if (recording_start_ns + start_delay_ns + start_seconds_ns + end_buffer_ns) > recording_end_ns:
                tqdm.write(
                    f"Recording {seg_id} is too short to include the first {start_seconds} seconds "
                    f"after start_delay. Skipping this segment.")
                continue
            if (recording_start_ns + start_delay_ns + start_seconds_ns + end_buffer_ns) > trace.ts_ns[-1]:
                tqdm.write(
                    f"Recording {seg_id} does not have enough magnetometer data for the first "
                    f"{start_seconds} seconds after start_delay. Skipping this segment.")
                continue
            end_ns = start_ns + start_seconds_ns
            n_chunks = 0
            for window_end_ns, _ in windows:
                if window_end_ns <= end_ns:
                    n_chunks += 1
                else:
                    break
        else:
            if (recording_start_ns + start_delay_ns + duration_ns + end_buffer_ns) > recording_end_ns:
                continue
            n_chunks = len(windows)

        file_windows = windows[:n_chunks]
        file_valid = apply_min_consecutive(initial_valid[:n_chunks])

        if start_seconds > 0 and not all(file_valid):
            tqdm.write(
                f"Segment {seg_id} has invalid sampling rate in some chunks. Skipping this segment "
                f"due to trainride_start_seconds > 0.")
            continue

        magnitude = _normalized_magnitude(trace, file_windows, normalize)
        per_file[segment_zip_file] = (trace, file_windows, file_valid, magnitude)

    if not per_file or len(per_file) != n_expected_files:
        return

    # `combined_valid_by_chunk` keeps the chunk ids present in every file.
    n_common = min(len(windows) for _, windows, _, _ in per_file.values())
    combined_valid = [True] * n_common
    for _, _, file_valid, _ in per_file.values():
        for chunk_id in range(n_common):
            combined_valid[chunk_id] = combined_valid[chunk_id] and file_valid[chunk_id]
    if not combined_valid:
        return
    if start_seconds > 0 and not all(combined_valid):
        tqdm.write(
            f"Segment {seg_id} has invalid sampling rate in some chunks. Skipping this segment "
            f"due to trainride_start_seconds > 0.")
        return

    # Use the latest chunk start across all colocated files as the shared downsampling origin,
    # so all files are aligned to the same time grid.
    origin_by_chunk = {}
    for trace, file_windows, _, _ in per_file.values():
        for chunk_id, (_, chunk_rows) in enumerate(file_windows):
            chunk_start = trace.ts_ns[first_row(chunk_rows)]
            if chunk_id not in origin_by_chunk or chunk_start > origin_by_chunk[chunk_id]:
                origin_by_chunk[chunk_id] = chunk_start

    for sampling_rate in sampling_rates:
        combination = (rolling_window, duration, start_seconds, sampling_rate)
        rows = rows_by_combination.get(combination)
        if rows is None:
            continue
        _emit_samples(rows, types_by_combination[combination], seg_id, train_type, per_file,
                      combined_valid, origin_by_chunk, duration, sampling_rate, normalize)


def _normalized_magnitude(trace: Trace, file_windows, normalize: str) -> np.ndarray:
    """Magnitudes of the trace after the ``trace`` normalization of ``normalize_chunks``.

    ``normalize_chunks`` derives one mean/std from the concatenation of all chunk
    magnitudes (boundary samples shared by two chunks are counted twice) and then
    rescales every chunk.  Both steps are elementwise, so the same result is
    obtained by rescaling the underlying trace once.
    """
    if normalize != 'trace' or not file_windows:
        return trace.magnitude

    parts = [trace.magnitude[chunk_rows] for _, chunk_rows in file_windows]
    all_magnitude = pd.Series(np.concatenate(parts))
    global_mean = all_magnitude.mean()
    global_std = all_magnitude.std()
    if global_std == 0 or np.isnan(global_std) or np.isclose(global_std, 0.0):
        return trace.magnitude - global_mean
    return (trace.magnitude - global_mean) / global_std


def _emit_samples(rows, row_types, seg_id, train_type, per_file, combined_valid, origin_by_chunk,
                  duration, sampling_rate, normalize):
    tgt_sample_ns = target_sample_ns(sampling_rate)
    target_len = int(round(as_ns(timedelta(seconds=duration)) / tgt_sample_ns))
    if target_len <= 0:
        return
    n_common = len(combined_valid)

    for segment_zip_file, (trace, file_windows, _, magnitude) in per_file.items():
        global_mask = [combined_valid[i] if i < n_common else False for i in range(len(file_windows))]
        dataset_data = []
        for segment_idx, chunk_ids in subsegment_runs(global_mask):
            selected = []
            for chunk_id in chunk_ids:
                _, chunk_rows = file_windows[chunk_id]
                chunk_magnitude = magnitude[chunk_rows]
                if np.isnan(chunk_magnitude).any():
                    tqdm.write(
                        f"Sub-chunk {chunk_id} of segment {seg_id}_{segment_idx} contains NaN values "
                        f"in 'magnitude'. Skipping this chunk.")
                    continue
                if normalize == "chunk":
                    chunk_magnitude = _normalize_values(chunk_magnitude)
                origin_ns = int(origin_by_chunk.get(int(chunk_id), trace.ts_ns[first_row(chunk_rows)]))
                selected.append((int(chunk_id), chunk_rows, origin_ns, chunk_magnitude))

            if not selected:
                continue

            labels = []
            values = []
            for slot, (_, chunk_rows, origin_ns, chunk_magnitude) in enumerate(selected):
                bins = (trace.ts_ns[chunk_rows] - origin_ns) // tgt_sample_ns
                keep = (bins >= 0) & (bins < target_len)
                labels.append(bins[keep] + slot * target_len)
                values.append(chunk_magnitude[keep])
            labels = np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)
            values = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
            means = grouped_means(values, labels, len(selected) * target_len)

            for slot, (chunk_id, _, origin_ns, _) in enumerate(selected):
                chunk_means = means[slot * target_len:(slot + 1) * target_len]
                if np.isnan(chunk_means).any():
                    tqdm.write(
                        f"Skipping sub-chunk {chunk_id} of segment {seg_id}_{segment_idx}: "
                        f"Too little data for synchronized downsampling: need {target_len} samples "
                        f"over {pd.to_timedelta(timedelta(seconds=duration))}.")
                    continue
                timestamps_ns = origin_ns + np.arange(target_len, dtype=np.int64) * tgt_sample_ns
                final_data = sample_frame(timestamps_ns, chunk_means.copy())
                dataset_data.append({
                    'id': f"{seg_id}_{segment_idx}_{chunk_id}",
                    'segment_id': f"{seg_id}_{segment_idx}",
                    'data': final_data,
                    'source_file': segment_zip_file,
                    'start_timestamp': final_data['timestamp'].iloc[0],
                    'end_timestamp': final_data['timestamp'].iloc[-1],
                })
        if dataset_data:
            rows.extend(dataset_data)
            row_types.extend([train_type] * len(dataset_data))


def _normalize_values(values: np.ndarray) -> np.ndarray:
    """Elementwise equivalent of :func:`magtrack.utils.filters._normalize`."""
    magnitude = pd.Series(values)
    mean = magnitude.mean()
    std = magnitude.std()
    if std == 0 or np.isnan(std) or np.isclose(std, 0.0):
        return (magnitude - mean).to_numpy()
    return ((magnitude - mean) / std).to_numpy()
