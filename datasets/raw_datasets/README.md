# Raw Datasets

This directory contains four folders with sensor recordings.
Each recording is stored as a ZIP file containing CSV files for the individual sensor streams (e.g. `Accelerometer.csv`, `Magnetometer.csv`, `Gyroscope.csv`, etc.).

Recordings in `trainride_recordings` additionally contain a `Labels.csv` file.
Each row in that file marks a station event along the journey, with columns: `journey`, `trip`, `station`, `seconds_from_journey_start`, `type` (arrival or departure), `electric`, and `train_type`.
Recordings in `no_trainride_recordings` do not contain a `Labels.csv`.

The `electric` column encodes the electrification (16.7 Hz) status of the train and track:

| Value | Meaning |
|-------|---------|
| 0 | Non-electric train on non-electric track |
| 1 | Electric train on electric track |
| 2 | Non-electric train on electric track (e.g. diesel on electrified track) |
| 3 | Change between electric/non-electric during the trip (e.g. due to locomotive change) |

The `train_type` column contains one of three categories: `long_distance`, `regional`, `urban` or `tram`.

## Folders

### `trainride_recordings_sample` / `no_trainride_recordings_sample`

These sample folders contain a subset of ZIP files. The recordings inside are **truncated** — the CSV files only cover a short initial segment of each recording.
This makes it possible to inspect the dataset structure (file layout, column names, sensor types).
Due to upload size limitation, the sample folders are not suitable for training or evaluation.

### `trainride_recordings` / `no_trainride_recordings`

These folders contain the full-length counterparts.
Because the complete ZIP files are not included in this repository, each folder provides a `dataset.yml` file that lists all recording identifiers belonging to the respective dataset.
This allows deeper investigation of the dataset composition and can serve as a reference for reproducing the full dataset.
