from __future__ import annotations

import gc
import re
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import yaml
from loguru import logger
from tqdm import tqdm

from magtrack.paper.generate_latex_vars import _DIGIT_WORD
from magtrack.utils.loader import read_pickle

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


def _activity_to_camel(name: str) -> str:
    """Turn an activity label into a CamelCase fragment usable in a macro name."""
    parts = re.split(r"[^0-9a-zA-Z]+", name)
    return "".join(p.capitalize() for p in parts if p)


def _format_number(value: float, round_string: str) -> str:
    """Format *value* with *round_string*."""
    try:
        return format(value, round_string)
    except (ValueError, TypeError):
        return str(int(round(value)))


def _read_activity_csv(zip_path: Path) -> Optional[pd.DataFrame]:
    """Read Activity.csv out of a recording zip, or None when it has none."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        name = "Activity.csv"
        if name not in zf.namelist():
            candidates = [n for n in zf.namelist() if n.endswith("/Activity.csv")]
            if not candidates:
                return None
            name = candidates[0]
        with zf.open(name) as f:
            return pd.read_csv(f)


def _activity_seconds_of(activity_df: pd.DataFrame) -> pd.Series:
    """Seconds spent per activity in one recording.

    Each row of Activity.csv marks the moment an activity started, so a row
    lasts until the next one; the final row is open-ended and counts as zero.
    This mirrors what create-tmd-dataset stores in ``no_train_activity``.
    """
    secs = pd.to_numeric(activity_df["seconds_elapsed"], errors="coerce")
    diffs = (secs.shift(-1) - secs).fillna(0)
    activities = activity_df["activity"].fillna("unknown")
    return diffs.groupby(activities).sum()


@app.command("get-activity-vars")
def get_activity_vars(
        dataset_file: Path = typer.Argument(..., help="Path to no_trainride_data.yml."),
        output_path: Optional[Path] = typer.Option(None, help="Output .tex file. Prints to stdout when omitted."),
        round_string: str = typer.Option(".0f", help="Format spec for the second values, e.g. '.0f' or '.1f'."),
):
    """Generate LaTeX \\newcommand definitions for the recorded activity times.

    Reads Activity.csv from every recording zip listed in the no-trainride
    dataset and sums the time spent per activity, emitting one command per
    activity plus a total.
    """
    with open(dataset_file, encoding="utf-8") as f:
        try:
            dataset = yaml.safe_load(f)
        except yaml.YAMLError as e:
            typer.echo(f"Error loading dataset file: {e}", err=True)
            raise typer.Exit(code=1)

    zip_files = dataset.get("zip_files", [])
    if not zip_files:
        typer.echo(f"No 'zip_files' listed in {dataset_file}", err=True)
        raise typer.Exit(code=1)

    zip_dir = dataset_file.parent
    activity_totals: dict[str, float] = {}
    n_read = 0
    n_missing_zip = 0
    n_without_activity = 0

    for zip_name in tqdm(zip_files, desc="Reading Activity.csv"):
        zip_path = zip_dir / f"{zip_name}.zip"
        if not zip_path.exists():
            logger.warning(f"Zip file not found: {zip_path}")
            n_missing_zip += 1
            continue
        try:
            activity_df = _read_activity_csv(zip_path)
        except (zipfile.BadZipFile, OSError) as e:
            logger.warning(f"Failed reading {zip_path.name}: {e}")
            continue
        if activity_df is None:
            logger.warning(f"No Activity.csv in {zip_path.name}")
            n_without_activity += 1
            continue
        if activity_df.empty:
            n_without_activity += 1
            continue
        missing = {"seconds_elapsed", "activity"} - set(activity_df.columns)
        if missing:
            logger.warning(f"{zip_path.name} Activity.csv missing columns {missing}")
            continue

        for activity, seconds in _activity_seconds_of(activity_df).items():
            if pd.isna(seconds):
                continue
            activity_totals[str(activity)] = activity_totals.get(str(activity), 0.0) + float(seconds)
        n_read += 1

    if not activity_totals:
        typer.echo("No activity data found in any recording.", err=True)
        raise typer.Exit(code=1)

    logger.info(f"Read {n_read}/{len(zip_files)} recordings "
                f"({n_missing_zip} missing, {n_without_activity} without Activity.csv)")

    tex_commands: list[str] = []
    for activity, seconds in activity_totals.items():
        name = f"\\TmdNoTrainActivity{_activity_to_camel(activity)}Time"
        tex_commands.append(
            f"\\newcommand{{{name}}}{{\\numprint{{{_format_number(seconds, round_string)}}}\\,s\\xspace}}")

    total_secs = sum(activity_totals.values())
    tex_commands.append(
        f"\\newcommand{{\\TmdNoTrainActivityTotalTime}}"
        f"{{\\numprint{{{_format_number(total_secs, round_string)}}}\\,s\\xspace}}")

    text = "\n".join(tex_commands) + "\n"
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        typer.echo(f".tex file saved to {output_path}")
    else:
        typer.echo(text, nl=False)


_TRAIN_TYPES: list[tuple[str, str]] = [
    ("long_distance", "LongDistance"),
    ("regional", "Regional"),
    ("urban", "Urban"),
    ("metro", "Metro"),
    ("tram", "Tram"),
]
_COLOC_TRAIN_TYPES: list[tuple[str, str]] = [
    ("all", "All"),
    ("long_distance", "LongDistance"),
    ("regional", "Regional"),
]
_ELECTRIC_LEVELS: list[tuple[int, str]] = [
    (0, "ElectricNo"),
    (1, "ElectricYes"),
    (2, "ElectricNoOnElectrified"),
    (3, "ElectricPartly"),
]


class _TexVars:
    """Collects \\newcommand definitions, skipping values the stats file lacks."""

    def __init__(self, stats: dict, round_string: str):
        self.stats = stats
        self.round_string = round_string
        self.lines: list[str] = []
        self.missing: list[str] = []

    def _value(self, key: str):
        if key not in self.stats or self.stats[key] is None:
            self.missing.append(key)
            return None
        return self.stats[key]

    def count(self, name: str, key: str) -> None:
        """Integer quantity, e.g. a number of segments."""
        value = self._value(key)
        if value is None:
            return
        self.lines.append(f"\\newcommand{{\\{name}}}{{\\numprint{{{int(value)}}}\\xspace}}")

    def duration(self, name: str, key: str) -> None:
        """Duration in hours, as stored by get-dataset-vars."""
        value = self._value(key)
        if value is None:
            return
        self.lines.append(
            f"\\newcommand{{\\{name}}}{{\\numprint{{{_format_number(float(value), self.round_string)}}}\\xspace}}")

    def percent(self, name: str, key: str, total_key: str) -> None:
        """Share of *key* in *total_key*, in percent."""
        value = self._value(key)
        total = self._value(total_key)
        if value is None or total is None:
            return
        share = 100.0 * float(value) / float(total) if float(total) else 0.0
        self.lines.append(
            f"\\newcommand{{\\{name}}}{{\\PctPrint{{{_format_number(share, self.round_string)}}}}}")

    def seconds(self, name: str, key: str, scale: float = 1.0) -> None:
        """Value converted to seconds, printed with a unit."""
        value = self._value(key)
        if value is None:
            return
        self.lines.append(
            f"\\newcommand{{\\{name}}}{{\\numprint{{{_format_number(float(value) * scale, '.0f')}}}\\,s\\xspace}}")


@app.command("get-stats-vars")
def get_stats_vars(
        stats_file: Path = typer.Argument(..., help="Path to the dataset_stats.yml built by get-dataset-vars."),
        output_path: Optional[Path] = typer.Option(None, help="Output .tex file. Prints to stdout when omitted."),
        round_string: str = typer.Option(".1f", help="Format spec for durations and percentages, e.g. '.1f'."),
):
    """Generate LaTeX \\newcommand definitions describing the traintrack dataset.

    Turns every figure in dataset_stats.yml into a command: the corpus totals,
    the per-train-type durations and segment counts, the electrification
    breakdown both absolute and as a share, and the total recording time in
    seconds.
    """
    with open(stats_file, encoding="utf-8") as f:
        try:
            stats = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            typer.echo(f"Error loading stats file: {e}", err=True)
            raise typer.Exit(code=1)

    if not isinstance(stats, dict) or not stats:
        typer.echo(f"No stats found in {stats_file}", err=True)
        raise typer.Exit(code=1)

    tex = _TexVars(stats, round_string)

    # Corpus totals.
    tex.count("TrainTrackZipFiles", "num_zip_files")
    tex.count("TrainTrackJourneys", "num_journeys")
    tex.count("TrainTrackTrips", "num_trips")
    tex.count("TrainTrackSegments", "num_segments")
    tex.count("TrainTrackUniqueSegments", "num_unique_segments")
    tex.count("TrainTrackStations", "num_stations")
    tex.duration("TrainTrackRecordingDuration", "recording_duration_total")

    # Per train type.
    for key, name in _TRAIN_TYPES:
        tex.duration(f"TrainTrack{name}Duration", f"recording_duration_{key}")
        tex.count(f"TrainTrack{name}Segments", f"num_segments_{key}")

    # Electrification over the whole corpus, then per train type.
    for level, level_name in _ELECTRIC_LEVELS:
        tex.duration(f"TrainTrack{level_name}Duration", f"electric_duration_{level}")
        tex.count(f"TrainTrack{level_name}Count", f"electric_count_{level}")
    for key, name in _TRAIN_TYPES:
        for level, level_name in _ELECTRIC_LEVELS:
            tex.duration(f"TrainTrack{name}{level_name}Duration", f"electric_duration_{key}_{level}")
            tex.count(f"TrainTrack{name}{level_name}Count", f"electric_count_{key}_{level}")

    # Electrification shares: per train type against that type's totals, then
    # over the whole corpus.
    for key, name in _TRAIN_TYPES:
        for level, level_name in _ELECTRIC_LEVELS:
            tex.percent(f"TrainTrack{name}{level_name}CountPct",
                        f"electric_count_{key}_{level}", f"num_segments_{key}")
        for level, level_name in _ELECTRIC_LEVELS:
            tex.percent(f"TrainTrack{name}{level_name}DurationPct",
                        f"electric_duration_{key}_{level}", f"recording_duration_{key}")
    for level, level_name in _ELECTRIC_LEVELS:
        tex.percent(f"TrainTrack{level_name}CountPct", f"electric_count_{level}", "num_segments")
    for level, level_name in _ELECTRIC_LEVELS:
        tex.percent(f"TrainTrack{level_name}DurationPct", f"electric_duration_{level}", "recording_duration_total")

    # The trainride recording time in seconds, for the TMD activity table.
    tex.seconds("TmdTrainActivityTime", "recording_duration_total", scale=3600.0)

    if not tex.lines:
        typer.echo(f"No known keys found in {stats_file}", err=True)
        raise typer.Exit(code=1)
    if tex.missing:
        logger.warning(f"{len(tex.missing)} stat(s) not in {stats_file.name}, commands skipped: "
                       f"{', '.join(sorted(set(tex.missing))[:8])}"
                       f"{' …' if len(set(tex.missing)) > 8 else ''}")

    text = "\n".join(tex.lines) + "\n"
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        typer.echo(f".tex file saved to {output_path}")
    else:
        typer.echo(text, nl=False)


@app.command("get-coloc-dataset-vars")
def get_coloc_dataset_vars(
        datasets_dir: Path = typer.Argument("datasets/coloc_datasets",
                                            help="Directory holding the normalized_trace_<type>_trains folders."),
        output_path: Optional[Path] = typer.Option(None, help="Output .tex file. Prints to stdout when omitted."),
        prefix: str = typer.Option("ColocDataset", help="Prefix for the generated command names."),
):
    """Generate LaTeX \\newcommand definitions for the colocation dataset sizes.

    Counts the distinct segments in every dataset of every train type and
    reports the range across them.  The number is not a single value: how much
    of a ride a dataset requires decides how many recordings survive, so a
    900 s dataset holds far fewer segments than a 60 s one.

    Every .pkl has to be read, so this takes a while on a full dataset tree.
    """
    if not datasets_dir.is_dir():
        typer.echo(f"Dataset directory not found: {datasets_dir}", err=True)
        raise typer.Exit(code=1)

    tex_commands: list[str] = []
    for type_key, type_name in _COLOC_TRAIN_TYPES:
        type_dir = datasets_dir / f"normalized_trace_{type_key}_trains"
        if not type_dir.is_dir():
            logger.warning(f"No directory for train type '{type_key}': {type_dir}")
            continue

        pkl_files = sorted(type_dir.glob("*.pkl"))
        if not pkl_files:
            logger.warning(f"No .pkl datasets in {type_dir}")
            continue

        counts: list[int] = []
        counts_by_first: dict[int, list[int]] = {}
        for pkl_path in tqdm(pkl_files, desc=f"  {type_key}", leave=False):
            try:
                data = read_pickle(pkl_path)
            except Exception as e:
                logger.warning(f"Failed reading {pkl_path.name}: {e}")
                continue
            if "segment_id" not in data.columns:
                logger.warning(f"{pkl_path.name} has no 'segment_id' column")
                del data
                continue
            n_segments = int(data["segment_id"].nunique())
            del data
            gc.collect()

            counts.append(n_segments)
            match = re.search(r"_coloc_first(\d+)_", pkl_path.name)
            if match:
                counts_by_first.setdefault(int(match.group(1)), []).append(n_segments)
            else:
                logger.warning(f"Cannot read the trainride length from {pkl_path.name}")

        if not counts:
            logger.warning(f"No readable datasets for train type '{type_key}'")
            continue

        logger.info(f"{type_key}: {len(counts)} dataset(s), "
                    f"{min(counts)}-{max(counts)} segments")
        tex_commands.append(
            f"\\newcommand{{\\{prefix}{type_name}SegmentsMin}}{{\\numprint{{{min(counts)}}}\\xspace}}")
        tex_commands.append(
            f"\\newcommand{{\\{prefix}{type_name}SegmentsMax}}{{\\numprint{{{max(counts)}}}\\xspace}}")

        for first in sorted(counts_by_first):
            word = _DIGIT_WORD.get(first, f"S{first}")
            smallest = min(counts_by_first[first])
            logger.info(f"{type_key} first={first}s: {len(counts_by_first[first])} dataset(s), "
                        f"min {smallest} segments")
            tex_commands.append(
                f"\\newcommand{{\\{prefix}{type_name}SegmentsFirst{word}}}"
                f"{{\\numprint{{{smallest}}}\\xspace}}")

    if not tex_commands:
        typer.echo(f"No datasets found under {datasets_dir}", err=True)
        raise typer.Exit(code=1)

    text = "\n".join(tex_commands) + "\n"
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        typer.echo(f".tex file saved to {output_path}")
    else:
        typer.echo(text, nl=False)


if __name__ == "__main__":
    app()
