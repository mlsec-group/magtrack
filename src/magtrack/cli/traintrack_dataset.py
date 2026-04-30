from __future__ import annotations

import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import yaml
from loguru import logger
from tqdm import tqdm

app = typer.Typer(help="Traintrack dataset utilities.")


def _sanitize_sensor_key(name: str) -> str:
    """Normalize sensor names for YAML keys."""
    out = []
    for ch in name.lower():
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_")


@app.callback()
def main(
        logfile: Optional[str] = typer.Option(None, "--logfile", "-l", help="Path to a log file to save logs to"),
):
    """Global options for all commands."""
    if logfile:
        logger.add(logfile, level="DEBUG")


@app.command("get-dataset-vars")
def get_dataset_vars(
        dataset_file: Path = typer.Argument(..., help="Path to dataset.yml."),
        output_file: Optional[Path] = typer.Option(None,
                                                   help="Output dataset_stats.yml path. Defaults to dataset.yml parent/dataset_stats.yml."),
):
    """Build dataset_stats.yml by processing zip files referenced in dataset.yml."""
    with open(dataset_file, encoding='utf-8') as f:
        try:
            dataset = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error loading dataset file: {e}")
            raise typer.Exit(code=1)

    zip_files = dataset.get('zip_files', [])
    journeys = dataset.get('journeys', [])
    trips = dataset.get('trips', [])

    zip_dir = dataset_file.parent
    if output_file is None:
        output_file = dataset_file.parent / "dataset_stats.yml"

    train_types = ["long_distance", "regional", "urban", "metro", "tram"]
    electric_types = [0, 1, 2, 3]

    recording_durations = {tt: timedelta(0) for tt in train_types}
    recording_segments = {tt: 0 for tt in train_types}
    electric_durations = {e: timedelta(0) for e in electric_types}
    electric_counts = {e: 0 for e in electric_types}
    electric_durations_per_type = {tt: {e: timedelta(0) for e in electric_types} for tt in train_types}
    electric_counts_per_type = {tt: {e: 0 for e in electric_types} for tt in train_types}

    stations = set(dataset.get('stations', []))
    unique_segments = set()
    total_segments = 0
    sensor_samples: dict[str, int] = {}

    for zip_name in tqdm(zip_files, desc="Processing zip labels"):
        zip_path = zip_dir / f"{zip_name}.zip"
        if not zip_path.exists():
            logger.warning(f"Zip file not found: {zip_path}")
            continue

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                labels_name = None
                for name in zf.namelist():
                    if name.endswith('Labels.csv'):
                        labels_name = name
                        break
                if labels_name is None:
                    logger.warning(f"No Labels.csv found in {zip_path.name}")
                    continue

                with zf.open(labels_name) as labels_file:
                    labels = pd.read_csv(labels_file)

                # Count all sensor samples (rows) from sensor CSV files.
                for name in zf.namelist():
                    if not name.endswith('.csv'):
                        continue
                    if name.endswith('Labels.csv') or name.endswith('Metadata.csv') or name.endswith('Annotation.csv'):
                        continue
                    sensor_name = Path(name).stem
                    sensor_key = _sanitize_sensor_key(sensor_name)
                    if not sensor_key:
                        continue
                    try:
                        with zf.open(name) as sensor_file:
                            sensor_df = pd.read_csv(sensor_file)
                            n_rows = int(len(sensor_df))
                    except Exception as e:
                        logger.warning(f"Failed reading sensor file {name} in {zip_path.name}: {e}")
                        continue
                    sensor_samples[sensor_key] = sensor_samples.get(sensor_key, 0) + n_rows
        except (zipfile.BadZipFile, OSError) as e:
            logger.warning(f"Failed reading {zip_path}: {e}")
            continue

        needed_cols = {'journey', 'trip', 'station', 'seconds_from_journey_start', 'type', 'electric', 'train_type'}
        if not needed_cols.issubset(labels.columns):
            logger.warning(f"{zip_path.name} Labels.csv missing columns {needed_cols - set(labels.columns)}")
            continue

        labels = labels.sort_values(['journey', 'trip', 'seconds_from_journey_start'])
        for (_, _, train_type, electric), grp in labels.groupby(['journey', 'trip', 'train_type', 'electric'],
                                                                sort=False):
            if train_type not in recording_durations or int(electric) not in electric_durations:
                continue
            pending_departure = None
            for _, row in grp.iterrows():
                event_type = str(row['type'])
                station = str(row['station'])
                t = float(row['seconds_from_journey_start'])
                stations.add(station)

                if event_type == 'departure':
                    pending_departure = (station, t)
                elif event_type == 'arrival' and pending_departure is not None:
                    start_station, start_t = pending_departure
                    duration_s = max(0.0, t - start_t)

                    recording_durations[train_type] += timedelta(seconds=duration_s)
                    recording_segments[train_type] += 1
                    electric_durations[int(electric)] += timedelta(seconds=duration_s)
                    electric_counts[int(electric)] += 1
                    electric_durations_per_type[train_type][int(electric)] += timedelta(seconds=duration_s)
                    electric_counts_per_type[train_type][int(electric)] += 1

                    unique_segments.add(f"{start_station}_{station}")
                    total_segments += 1
                    pending_departure = None

    recording_duration_total = timedelta(0)
    for tt in train_types:
        recording_duration_total += recording_durations[tt]

    stats = {
        "num_journeys": len(journeys),
        "num_trips": len(trips),
        "num_segments": int(total_segments),
        "num_unique_segments": len(unique_segments),
        "num_stations": len(stations),
        "num_zip_files": len(zip_files),
        "recording_duration_total": recording_duration_total.total_seconds() / 3600,
        "sensor_samples_total": int(sum(sensor_samples.values())),
    }

    for tt in train_types:
        stats[f"recording_duration_{tt}"] = recording_durations[tt].total_seconds() / 3600
        stats[f"num_segments_{tt}"] = recording_segments[tt]
    for e in electric_types:
        stats[f"electric_duration_{e}"] = electric_durations[e].total_seconds() / 3600
        stats[f"electric_count_{e}"] = electric_counts[e]
    for tt in train_types:
        for e in electric_types:
            stats[f"electric_duration_{tt}_{e}"] = electric_durations_per_type[tt][e].total_seconds() / 3600
            stats[f"electric_count_{tt}_{e}"] = electric_counts_per_type[tt][e]
    for sensor_key, n in sorted(sensor_samples.items()):
        stats[f"sensor_samples_{sensor_key}"] = int(n)

    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(stats, f, allow_unicode=True, sort_keys=False)
    logger.info(f"Dataset stats written to {output_file}")


if __name__ == "__main__":
    app()
