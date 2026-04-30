import pandas as pd
from loguru import logger

from magtrack.utils.loader import get_data_from_zipfile_by_name


def resolve_seed(seed_value) -> int:
    if isinstance(seed_value, int):
        return seed_value
    return int.from_bytes(str(seed_value).encode(), "big") % 9_999_999


def get_trainride_data(zip_path: str, label_rows: pd.DataFrame, sensor: str = 'Magnetometer') -> pd.DataFrame:
    start_seconds = label_rows.iloc[0]['departure']
    end_seconds = label_rows.iloc[len(label_rows) - 1]['arrival']
    if pd.isna(start_seconds) or pd.isna(end_seconds):
        logger.error(f"skipping {zip_path}: departure or arrival is NaN")
        return pd.DataFrame()
    recording = get_data(zip_path, sensor)
    if recording.empty:
        return recording
    if 'seconds_from_journey_start' in recording.columns:
        recording = recording[
            (recording['seconds_from_journey_start'] >= float(start_seconds))
            & (recording['seconds_from_journey_start'] <= float(end_seconds))
            ].reset_index(drop=True)
    else:
        logger.error(
            f"sensor data in {zip_path} has no 'seconds_from_journey_start' column; "
            f"cannot filter with numeric departure/arrival labels"
        )
        return pd.DataFrame()
    return recording


def get_data(zip_path: str, sensor: str = 'Magnetometer') -> pd.DataFrame:
    try:
        return get_data_from_zipfile_by_name(zip_path, sensor + '.csv')
    except Exception as e:
        logger.error(f"error parsing data: {e} in file {zip_path}")
        return pd.DataFrame()
