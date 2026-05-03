## Dataset overview

This document briefly describes the datasets included in this repository, what each folder contains, and quick commands to inspect the sample files. Note: the datasets bundled with this submission are intentionally small subsets to keep archive size manageable — see the notes below.

### Raw Datasets

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


### Preprocessed Datasets

#### TMD Datasets

TMD (transport-mode detection) datasets used for the TMD examples and evaluation. Example files included:
```
  - `tmd_s60_d10_small.pkl`
```

#### Colocation Datasets

prebuilt colocation datasets (pickle files). Example files included:
```
  - `all_coloc_first0_30s_window10_10Hz.pkl`
  - `all_coloc_first300_60s_window10_40Hz.pkl`
  - `all_coloc_first60_5s_window50_40Hz.pkl`
```

