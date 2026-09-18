import hashlib
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from tqdm import tqdm

from magtrack.utils.checker import check_sampling_rate_chunks, split_df_chunks_by_mask
from magtrack.utils.filters import _sliding_window, _downsample, _normalize
from magtrack.utils.loader import get_data_from_zipfile_by_name
from magtrack.utils.splitter import get_chunked_dataframe

app = typer.Typer(help="Create dataset for colocation")


def normalize_chunks(chunks: list) -> list:
    """
    Normalize chunks using a single global mean/std across all chunks' 'magnitude' values,
    rather than per-chunk statistics. Chunks are modified in-place.
    """
    if not chunks:
        return chunks

    mags = []
    for c in chunks:
        if c is None:
            continue
        try:
            if len(c) == 0:
                continue
        except TypeError:
            continue
        if 'magnitude' not in c.columns:
            continue
        mags.append(c['magnitude'])

    if not mags:
        return chunks

    all_mag = pd.concat(mags, axis=0)
    global_mean = all_mag.mean()
    global_std = all_mag.std()

    if global_std == 0 or np.isnan(global_std) or np.isclose(global_std, 0.0):
        for c in chunks:
            if c is None or 'magnitude' not in getattr(c, 'columns', []):
                continue
            if len(c) == 0:
                continue
            c['magnitude'] = c['magnitude'] - global_mean
    else:
        for c in chunks:
            if c is None or 'magnitude' not in getattr(c, 'columns', []):
                continue
            if len(c) == 0:
                continue
            c['magnitude'] = (c['magnitude'] - global_mean) / global_std

    return chunks


def load_magnetometer_data(traintrack_basepath, zip_file, sensor_type, EXTRACTION_FUNCTION, rolling_window):
    zip_file_path = os.path.join(traintrack_basepath, zip_file + '.zip')
    magnetometer_recording = get_data_from_zipfile_by_name(zip_file_path, sensor_type + '.csv')

    magnetometer_trace = EXTRACTION_FUNCTION(magnetometer_recording,
                                             timestamp_column='seconds_from_journey_start')
    magnetometer_trace['timestamp'] = magnetometer_trace['timestamp'].apply(lambda x: timedelta(seconds=x))

    if rolling_window > 1:
        magnetometer_trace = _sliding_window(magnetometer_trace, window_size=rolling_window)

    return magnetometer_trace


@app.command("create")
def create(
        dataset_path: Path = typer.Option(..., help="Path to traintrack dataset (yaml)"),
        duration: int = typer.Option(10, help="Duration in seconds for each extracted window"),
        extraction_function_name: str = typer.Option("filter_notch_highpass",
                                                     help="Name of extraction function in utils.filters"),
        trainride_start_seconds: int = typer.Option(0, help="Seconds to use from the start of trainride (0 = all)"),
        output_path: Path = typer.Option(..., help="Path or file, where ml_datasets will be stored"),
        sensor_type: str = typer.Option("Magnetometer", help="Sensor type to extract from zipfiles"),
        filter_train_type: list[str] = typer.Option(['long_distance', 'regional'],
                                                    help="List of train types to include (long-distance, regional...). If not set, all train types are included."),
        dataset_name: str = typer.Option("magtrack", help="Name of the dataset (used in output filename)"),
        start_delay: int = typer.Option(2, help="Seconds to delay from recording start"),
        end_buffer: int = typer.Option(2, help="Seconds to trim from the end of each recording"),
        min_sample_frequency: float = typer.Option(90.0, help="Minimum sample frequency in Hz for samples"),
        rolling_window: int = typer.Option(5, help="Rolling window size"),
        normalize: str = typer.Option("trace",
                                      help="Normalization mode: 'none' (default), 'chunk' (normalize each chunk), or 'trace' (normalize whole trace before chunking)"),
        sampling_rate: float = typer.Option(20.0, help="Expected sampling rate in Hz"),
        seed=typer.Option('magtrack', help="Random seed"),
        overwrite: bool = typer.Option(False, help="Overwrite existing output file"),
        compression: str = typer.Option("zstd", help="Compression codec for pickle IO."),
):
    random.seed(seed)

    try:
        import magtrack.utils.filters as _filters
        if extraction_function_name and hasattr(_filters, extraction_function_name):
            EXTRACTION_FUNCTION = getattr(_filters, extraction_function_name)
        else:
            raise AttributeError("Extraction function not found")
    except Exception:
        typer.echo(f"Failed to import extraction function {extraction_function_name}.", err=True)
        raise typer.Exit(1)

    allowed_norms = {"none", "chunk", "trace"}
    if normalize not in allowed_norms:
        raise typer.BadParameter(f"normalize must be one of {sorted(allowed_norms)}")

    if output_path.exists() and output_path.is_file() and not overwrite:
        print(f"Output file {output_path} already exists. Use --overwrite to overwrite.")
        raise typer.Exit(1)

    allowed_types = {"long_distance", "regional"}
    for filter in filter_train_type:
        if filter not in allowed_types:
            raise typer.BadParameter(f"filter_train_type values must be one of {sorted(allowed_types)}")

    with open(dataset_path, 'r') as file:
        traintrack_dataset = yaml.safe_load(file)
    traintrack_basepath = os.path.dirname(dataset_path)
    traintrack_trips = pd.DataFrame(traintrack_dataset['trips'])
    traintrack_segments = pd.DataFrame(traintrack_dataset['segments'])

    coloc_rows = []

    electric_trips = traintrack_trips[traintrack_trips['electric'] == 1]
    colocated_electric_trips = electric_trips[electric_trips['zip_files'].apply(lambda zip_files: len(zip_files) >= 2)]
    colocated_electric_trips = colocated_electric_trips[colocated_electric_trips['train_type'].isin(filter_train_type)]

    trace_cache = {}

    for i, trip in tqdm(colocated_electric_trips.iterrows(), desc="Processing trips",
                        total=len(colocated_electric_trips)):
        trip_zip_files = trip['zip_files']
        trip_traces = {}

        for trip_zip_file in trip_zip_files:
            if trip_zip_file in trace_cache:
                trip_traces[trip_zip_file] = trace_cache[trip_zip_file]
            else:
                try:
                    trace_cache[trip_zip_file] = load_magnetometer_data(traintrack_basepath, trip_zip_file, sensor_type,
                                                                        EXTRACTION_FUNCTION, rolling_window)
                    trip_traces[trip_zip_file] = trace_cache[trip_zip_file]
                except Exception as e:
                    tqdm.write(f"Failed to load {trip_zip_file}: {e}. Skipping.")
                    continue

        trip_segments = traintrack_segments[
            traintrack_segments["trip_id"].eq(trip["trip_id"])
        ]
        trip_segment_ids = trip_segments['segment'].unique()
        for trip_segment_id in trip_segment_ids:
            segment_rows = trip_segments[trip_segments['segment'].eq(trip_segment_id)].copy()
            if segment_rows.empty:
                continue

            segment_chunk_dfs = {}
            combined_valid_by_chunk = None

            for _, segment_row in segment_rows.iterrows():
                segment_zip_file = segment_row['zip_file']
                seg_id = f"{str(segment_row.trip_id)}_{segment_row.segment}"
                if segment_zip_file not in trip_traces:
                    continue

                magnetometer_trace = trip_traces[segment_zip_file]
                recording_start_time = timedelta(seconds=segment_row.start_timestamp)
                recording_end_time = timedelta(seconds=segment_row.end_timestamp)
                try:
                    if trainride_start_seconds > 0:
                        if (recording_start_time + timedelta(
                                seconds=start_delay + trainride_start_seconds + end_buffer)) > recording_end_time:
                            tqdm.write(
                                f"Recording {seg_id} is too short to include the first {trainride_start_seconds} seconds after start_delay. Skipping this segment.")
                            continue
                        if (recording_start_time + timedelta(
                                seconds=start_delay + trainride_start_seconds + end_buffer)) > \
                                magnetometer_trace['timestamp'].iloc[-1]:
                            tqdm.write(
                                f"Recording {seg_id} does not have enough magnetometer data for the first {trainride_start_seconds} seconds after start_delay. Skipping this segment.")
                            continue
                    else:
                        if (recording_start_time + timedelta(
                                seconds=start_delay + duration + end_buffer)) > recording_end_time:
                            continue
                    if recording_start_time < magnetometer_trace['timestamp'].iloc[0]:
                        tqdm.write(f"Recording {seg_id} starts before the magnetometer trace. Skipping this segment.")
                        continue
                    start_time = recording_start_time + timedelta(seconds=start_delay)
                    if trainride_start_seconds:
                        end_time = recording_start_time + timedelta(seconds=start_delay + trainride_start_seconds)
                    else:
                        end_time = recording_end_time - timedelta(seconds=end_buffer)
                except IndexError:
                    continue

                magnetometer_trace_chunks = get_chunked_dataframe(
                    magnetometer_trace,
                    duration=timedelta(seconds=duration),
                    start_time=start_time,
                    end_time=end_time,
                )

                if normalize == "trace":
                    magnetometer_trace_chunks = normalize_chunks(magnetometer_trace_chunks)

                sampling_rate_check = check_sampling_rate_chunks(
                    magnetometer_trace_chunks,
                    min_frequency=min_sample_frequency,
                )

                if trainride_start_seconds > 0 and not all(sampling_rate_check):
                    tqdm.write(
                        f"Segment {seg_id} has invalid sampling rate in some chunks. Skipping this segment due to trainride_start_seconds > 0.")
                    continue

                magnetometer_trace_chunks_df = pd.DataFrame({'data': magnetometer_trace_chunks})
                magnetometer_trace_chunks_df['chunk_id'] = magnetometer_trace_chunks_df.index
                segment_chunk_dfs[segment_zip_file] = magnetometer_trace_chunks_df

                sampling_mask_df = pd.DataFrame({
                    'chunk_id': magnetometer_trace_chunks_df['chunk_id'].to_numpy(),
                    'is_valid': np.asarray(sampling_rate_check, dtype=bool),
                })

                if combined_valid_by_chunk is None:
                    combined_valid_by_chunk = sampling_mask_df.copy()
                else:
                    combined_valid_by_chunk = combined_valid_by_chunk.merge(
                        sampling_mask_df,
                        on='chunk_id',
                        how='inner',
                        suffixes=('', '_new'),
                    )
                    combined_valid_by_chunk['is_valid'] = (
                            combined_valid_by_chunk['is_valid'] & combined_valid_by_chunk['is_valid_new']
                    )
                    combined_valid_by_chunk = combined_valid_by_chunk[['chunk_id', 'is_valid']]

            if (
                    not segment_chunk_dfs
                    or combined_valid_by_chunk is None
                    or combined_valid_by_chunk.empty
                    or len(segment_chunk_dfs) != segment_rows['zip_file'].nunique()
            ):
                continue
            if trainride_start_seconds > 0 and not combined_valid_by_chunk['is_valid'].all():
                tqdm.write(
                    f"Segment {seg_id} has invalid sampling rate in some chunks. Skipping this segment due to trainride_start_seconds > 0.")
                continue

            downsampling_origin_by_chunk = {}
            for chunk_df_for_file in segment_chunk_dfs.values():
                if chunk_df_for_file.empty:
                    continue
                for _, chunk_row in chunk_df_for_file.iterrows():
                    chunk_id = chunk_row['chunk_id']
                    chunk_data = chunk_row['data']
                    if chunk_data is None or len(chunk_data) == 0:
                        continue
                    chunk_start = chunk_data['timestamp'].iloc[0]
                    if chunk_id not in downsampling_origin_by_chunk or chunk_start > downsampling_origin_by_chunk[
                        chunk_id]:
                        downsampling_origin_by_chunk[chunk_id] = chunk_start

            for segment_zip_file, chunk_df in segment_chunk_dfs.items():
                global_mask = chunk_df['chunk_id'].map(combined_valid_by_chunk['is_valid'].to_dict()).fillna(
                    False).astype(bool).tolist()
                subsegments = split_df_chunks_by_mask(chunk_df, global_mask)
                dataset_data = []
                for _, subsegment_df in subsegments.groupby('subsegment_id', sort=True):
                    for _, subsegment_row in subsegment_df.sort_values('chunk_id').iterrows():
                        segment_idx = subsegment_row['subsegment_id']
                        chunk_id = subsegment_row['chunk_id']
                        sub_chunk = subsegment_row['data'].copy()
                        if sub_chunk['magnitude'].isnull().any():
                            tqdm.write(
                                f"Sub-chunk {chunk_id} of segment {seg_id}_{segment_idx} contains NaN values in 'magnitude'. Skipping this chunk.")
                            continue
                        if normalize == "chunk":
                            sub_chunk = _normalize(sub_chunk)
                        downsampling_origin = downsampling_origin_by_chunk.get(chunk_id, sub_chunk['timestamp'].iloc[0])
                        try:
                            final_data = _downsample(
                                sub_chunk,
                                sampling_rate=sampling_rate,
                                origin=downsampling_origin,
                                duration=timedelta(seconds=duration),
                            )
                        except ValueError as e:
                            tqdm.write(
                                f"Skipping sub-chunk {chunk_id} of segment {seg_id}_{segment_idx}: {e}"
                            )
                            continue

                        dataset_data.append({
                            'id': f"{seg_id}_{segment_idx}_{chunk_id}",
                            'segment_id': f"{seg_id}_{segment_idx}",
                            'data': final_data,
                            'source_file': segment_zip_file,
                            'start_timestamp': final_data['timestamp'].iloc[0],
                            'end_timestamp': final_data['timestamp'].iloc[-1],
                        })
                if dataset_data:
                    coloc_rows.extend(dataset_data)

    if coloc_rows:
        coloc_df = pd.DataFrame(coloc_rows)
        old_length = len(coloc_df)
    else:
        typer.echo("No valid colocated samples found. Exiting without creating dataset.")
        raise typer.Exit(0)

    target_chunk_samples = int(duration * sampling_rate)
    valid_length_mask = coloc_df['data'].apply(lambda df: len(df) == target_chunk_samples)
    if not valid_length_mask.all():
        invalid_count = (~valid_length_mask).sum()
        tqdm.write(f"Dropping {invalid_count} sample(s) due to invalid chunk length (expected {target_chunk_samples} samples per chunk).")
        coloc_df = coloc_df[valid_length_mask].reset_index(drop=True)

    if trainride_start_seconds > 0:
        target_recording_length = trainride_start_seconds / duration
        segment_length_mask = coloc_df.groupby(['source_file', 'segment_id'])['data'].transform(
            lambda group: len(group) == target_recording_length)
        if not segment_length_mask.all():
            invalid_count = (~segment_length_mask).sum()
            tqdm.write(f"Dropping {invalid_count} sample(s) due to invalid segment length (expected {target_recording_length} samples per segment for trainride_start_seconds > 0).")
            coloc_df = coloc_df[segment_length_mask].reset_index(drop=True)
    else:
        id_counts = coloc_df['id'].value_counts()
        ids_to_keep = id_counts[id_counts >= 2].index
        coloc_df = coloc_df[coloc_df['id'].isin(ids_to_keep)].reset_index(drop=True)
        dropped = old_length - len(coloc_df)
        if dropped > 0:
            tqdm.write(f"Dropped {dropped} item(s) because their id appeared < 2 times.")

    if len(coloc_df) == 0:
        typer.echo("No valid colocated samples. Exiting without creating dataset.")
        raise typer.Exit(0)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    out_filename = f"{dataset_name}_coloc_first{trainride_start_seconds}_{duration}s_window{rolling_window}_{int(sampling_rate)}Hz.pkl"
    coloc_df.to_pickle(os.path.join(output_path, out_filename), compression=compression)
    meta_df = pd.DataFrame({
        'dataset_name': [dataset_name],
        'dataset_md5': [hashlib.md5(open(os.path.join(output_path, out_filename), 'rb').read()).hexdigest()],
        'extraction_function_name': [extraction_function_name],
        'duration': [duration],
        'trainride_start_seconds': [trainride_start_seconds],
        'sensor_type': [sensor_type],
        'filter_train_type': [filter_train_type],
        'start_delay': [start_delay],
        'end_buffer': [end_buffer],
        'min_sample_frequency': [min_sample_frequency],
        'rolling_window': [rolling_window],
        'normalize': [normalize],
        'sampling_rate': [sampling_rate],
        'seed': [seed],
        'num_samples': [len(coloc_df)],
    })
    meta_df.to_csv(os.path.join(output_path, f"{os.path.splitext(out_filename)[0]}.meta.csv"), index=False)

    typer.echo(f"Saved coloc dataset with {len(coloc_df)} samples to {os.path.join(output_path, out_filename)}")


if __name__ == "__main__":
    app()
