import datetime
import hashlib
import os
from pathlib import Path

import pandas as pd
import typer
import yaml
from loguru import logger
from tqdm import tqdm

from magtrack.utils.checker import LowSamplingRateException, EmptyDataframeException, \
    HighDistanceException, check_sampling_rate
from magtrack.utils.filters import filter_abs
from magtrack.utils.labels import load_labels_table
from magtrack.utils.loader import get_data_from_zipfile_by_name
from magtrack.utils.splitter import get_chunked_dataframe
from magtrack.utils.utils import get_data, get_trainride_data, resolve_seed

app = typer.Typer(help="Create dataset splits from labels_dict (typer CLI)")


def _is_valid_chunk(chunk, min_sample_frequency: float) -> bool:
    try:
        check_sampling_rate(chunk, min_frequency=min_sample_frequency)
        return True
    except (LowSamplingRateException, HighDistanceException, EmptyDataframeException):
        return False


def _chunk_duration_seconds(chunk: pd.DataFrame, time_column='timestamp') -> float:
    col = chunk[time_column]
    if pd.api.types.is_float_dtype(col) or pd.api.types.is_integer_dtype(col):
        return float(col.iloc[-1]) - float(col.iloc[0])
    try:
        ser = pd.to_datetime(col)
        return (ser.iloc[-1] - ser.iloc[0]).total_seconds()
    except Exception:
        return 0.0


def format_data(
        data: pd.DataFrame,
        zip_path: str,
        duration: int,
        trainride_flag: bool,
        min_sample_frequency: float,
        only_complete_recordings: bool,
        trainride_start_seconds: int,
        trip_id: str = None,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    time_column = 'seconds_elapsed'
    try:
        trace = filter_abs(data, timestamp_column=time_column)
        trace['timestamp'] = pd.to_datetime(trace['timestamp'], unit='s')
        if trainride_flag and trainride_start_seconds > 0:
            trace = trace[
                trace['timestamp'] < (trace['timestamp'].min() + pd.Timedelta(seconds=trainride_start_seconds + 1))]
    except KeyError:
        logger.error(f"Missing required columns in data from {zip_path}")
        return pd.DataFrame()
    if trace.empty:
        logger.error(f"Error extracting trace from data")
        return pd.DataFrame()
    if only_complete_recordings:
        try:
            check_sampling_rate(trace, min_frequency=min_sample_frequency)
        except (LowSamplingRateException, HighDistanceException, EmptyDataframeException) as e:
            logger.error(e.msg)
            return pd.DataFrame()
    try:
        chunks = get_chunked_dataframe(trace, duration=datetime.timedelta(seconds=duration))
    except BaseException as e:
        logger.error(f"Error chunking trace from {zip_path}: {e}")
        return pd.DataFrame()

    if not chunks:
        logger.error("No chunks extracted from trace")
        return pd.DataFrame()

    chunks_series = pd.Series(chunks)
    valid_mask = chunks_series.apply(_is_valid_chunk, min_sample_frequency=min_sample_frequency)

    duration_threshold = 0.95 * duration
    duration_mask = chunks_series.apply(
        lambda ch: _chunk_duration_seconds(ch) >= duration_threshold)

    combined_mask = valid_mask & duration_mask

    removed = (~combined_mask).sum()
    if removed:
        logger.info(f"Removed {removed} chunk(s) due to sampling/quality/duration issues")

    chunks = chunks_series[combined_mask].tolist()
    if not chunks:
        logger.error("All chunks removed due to sampling constraints")
        return pd.DataFrame()

    formatted_chunks = pd.DataFrame({'data': chunks})
    if trip_id:
        formatted_chunks['id'] = f"{os.path.basename(zip_path).replace('.zip', '')}_{trip_id}"
        formatted_chunks['trip_id'] = trip_id
    else:
        formatted_chunks['id'] = f"{os.path.basename(zip_path).replace('.zip', '')}"
    formatted_chunks['trainride'] = trainride_flag

    return formatted_chunks


def get_trip(trip_id, all_trips) -> dict:
    return next((trip for trip in all_trips if trip['trip_id'] == trip_id), None)


def process_zip(
        zip_path: str,
        duration: int,
        trainride_flag: bool,
        filter_train_type: list,
        min_sample_frequency: float,
        only_complete_recordings: bool,
        trainride_start_seconds: int,
        all_trips=None
) -> pd.DataFrame:
    if all_trips is None:
        all_trips = []
    if trainride_flag:
        try:
            labels = load_labels_table(zip_path)
            trips = labels['trip'].unique()
            final_data = pd.DataFrame()
            for trip in trips:
                label_rows = labels[labels['trip'] == trip]
                trip_data = get_trip(trip, all_trips)
                if trip_data['electric'] != 1:
                    continue
                if filter_train_type:
                    if trip_data['train_type'] not in filter_train_type:
                        continue
                data = get_trainride_data(zip_path, label_rows)
                final_data = pd.concat([final_data,
                                        format_data(data, zip_path, duration, trainride_flag, min_sample_frequency,
                                                    only_complete_recordings, trainride_start_seconds, trip)])
        except FileNotFoundError:
            logger.error(f"No labels found for {zip_path}, skipping trainride processing")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing trainride data: {e}")
            return pd.DataFrame()
        return final_data
    else:
        try:
            data = get_data(zip_path)
            return format_data(data, zip_path, duration, trainride_flag, min_sample_frequency, only_complete_recordings,
                               trainride_start_seconds)
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return pd.DataFrame()


@app.callback(invoke_without_command=True)
def create(
        ctx: typer.Context,
        traintrack_dataset_path: Path = typer.Option(..., help="Path to traintrack dataset (yaml)"),
        no_trainride_dataset_path: Path = typer.Option(..., help="Path to traintrack no-train dataset (yaml)"),
        duration: int = typer.Option(10, help="Duration in seconds for each extracted chunk"),
        output_path: Path = typer.Option(..., help="Path or file, where ml_datasets will be stored"),
        min_sample_frequency: float = typer.Option(90.0, help="Minimum sample frequency in Hz for samples"),
        only_complete_recordings: bool = typer.Option(True, help="Only use complete recordings without any issues"),
        trainride_start_seconds: int = typer.Option(0, help="Seconds to use from the start of trainride (0 = all)"),
        filter_train_type: list[str] = typer.Option(None,
                                                    help="List of train types to include (long-distance, regional...). If not set, all train types are included."),
        seed=typer.Option('magtrack', help="Random seed"),
        overwrite: bool = typer.Option(False, help="Overwrite existing output file"),
        balanced: bool = typer.Option(False,
                                      help="Whether to balance the number of samples with and without trainrides"),
        compression: str = typer.Option("zstd", help="Compression codec for pickle IO."),
):
    seed_int = resolve_seed(seed)

    try:
        if ctx.invoked_subcommand is not None:
            return
    except Exception:
        pass

    if filter_train_type:
        for i, item in enumerate(filter_train_type):
            filter_train_type[i] = item.strip('\'"')

    if output_path.exists() and output_path.is_file() and not overwrite:
        print(f"Output file {output_path} already exists. Use --overwrite to overwrite.")
        raise typer.Exit(1)

    trainride_dataset_name = os.path.splitext(os.path.basename(traintrack_dataset_path))[0]
    with open(traintrack_dataset_path, 'r') as file:
        trainride_dataset = yaml.safe_load(file)
        trainride_zipfiles = pd.DataFrame(trainride_dataset['zip_files'], columns=['zip_files'])
        trainride_basepath = os.path.dirname(traintrack_dataset_path)

    no_trainride_dataset_name = os.path.splitext(os.path.basename(no_trainride_dataset_path))[0]
    with open(no_trainride_dataset_path, 'r') as file:
        no_trainride_dataset = yaml.safe_load(file)
        no_trainride_zipfiles = pd.DataFrame(no_trainride_dataset['zip_files'], columns=['zip_files'])
        no_trainride_basepath = os.path.dirname(no_trainride_dataset_path)

    trainride_zipfiles = trainride_zipfiles.sample(frac=1, random_state=seed_int).reset_index(drop=True)
    no_trainride_zipfiles = no_trainride_zipfiles.sample(frac=1, random_state=seed_int).reset_index(drop=True)

    tmd_df = pd.DataFrame(columns=['id', 'data', 'trainride'])
    no_train_activity = {}

    for zip_file in tqdm(no_trainride_zipfiles['zip_files'], desc="Processing no-trainride zip files"):
        zip_file_path = os.path.join(no_trainride_basepath, zip_file + '.zip')
        try:
            activity_df = get_data_from_zipfile_by_name(zip_file_path, 'Activity.csv')
            if not activity_df.empty:
                secs = pd.to_numeric(activity_df['seconds_elapsed'], errors='coerce')
                # compute duration per row as next_sec - current_sec; last row duration set to 0
                diffs = secs.shift(-1) - secs
                diffs = diffs.fillna(0)

                activities = activity_df['activity'].fillna('unknown')
                per_file = diffs.groupby(activities).sum()
                for act, dur in per_file.items():
                    if pd.isna(dur):
                        continue
                    no_train_activity[act] = no_train_activity.get(act, 0.0) + float(dur)
        except FileNotFoundError:
            logger.warning(f"No Activity.csv found in {zip_file_path}. Skipping activity summary for this file.")
        except Exception as e:
            logger.warning(f"Error while extracting Activity.csv from {zip_file_path}: {e}")
        new_data = process_zip(zip_file_path, duration, False, filter_train_type, min_sample_frequency,
                               only_complete_recordings, trainride_start_seconds)
        tmd_df = pd.concat([tmd_df, new_data], ignore_index=True)

    for zip_file in tqdm(trainride_zipfiles['zip_files'], desc="Processing trainride zip files"):
        zip_file_path = os.path.join(trainride_basepath, zip_file + '.zip')
        new_data = process_zip(zip_file_path, duration, True, filter_train_type, min_sample_frequency,
                               only_complete_recordings, trainride_start_seconds, all_trips=trainride_dataset['trips'])
        tmd_df = pd.concat([tmd_df, new_data], ignore_index=True)
        if balanced:
            if len(tmd_df[tmd_df['trainride'] == True]) >= len(tmd_df[tmd_df['trainride'] == False]):
                break

    if output_path.is_dir():
        out_dirname = output_path
    else:
        out_dirname = os.path.dirname(output_path)
    Path(out_dirname).mkdir(parents=True, exist_ok=True)
    if output_path.is_dir():
        out_filename = f"tmd_{duration}s.pkl"
    else:
        out_filename = os.path.basename(output_path)
    tmd_df.to_pickle(os.path.join(out_dirname, out_filename), compression=compression)
    meta_df = pd.DataFrame({
        'trainride_dataset_name': [trainride_dataset_name],
        'no_trainride_dataset_name': [no_trainride_dataset_name],
        'dataset_md5': [hashlib.md5(open(os.path.join(out_dirname, out_filename), 'rb').read()).hexdigest()],
        'duration': [duration],
        'min_sample_frequency': [min_sample_frequency],
        'only_complete_recordings': [only_complete_recordings],
        'trainride_start_seconds': [trainride_start_seconds],
        'balanced': [balanced],
        'seed': [seed],
        'num_samples': [len(tmd_df)],
        'num_trainride_samples': [len(tmd_df[tmd_df['trainride'] == True])],
        'num_no_trainride_samples': [len(tmd_df[tmd_df['trainride'] == False])],
        'no_train_activity': [no_train_activity],
    })
    meta_df.to_csv(os.path.join(out_dirname, f"{os.path.splitext(out_filename)[0]}.meta.csv"), index=False)

    typer.echo(
        f"Saved TMD dataset with {len(tmd_df)} samples ({len(tmd_df[tmd_df['trainride'] == True])}/{len(tmd_df[tmd_df['trainride'] == False])} to {os.path.join(out_dirname, out_filename)}")


if __name__ == "__main__":
    app()
