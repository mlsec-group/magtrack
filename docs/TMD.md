## Create TMD datasets

Use `create-tmd-dataset` to generate trainride/no-trainride chunks.

Required arguments:
- `--traintrack-dataset-path` (YAML with trainride zip files and trips metadata)
- `--no-trainride-dataset-path` (YAML with no-trainride zip files)
- `--output-path` (output file path, e.g. `.pkl`)

Note about the included sample data
---------------------------------

The dataset files included in the `datasets/` sample directories are intentionally very limited due to size restrictions for this submission. In particular, the file `datasets/tmd_datasets/tmd_s60_d10_small.pkl` is a small subset provided to allow quick testing and to reproduce the example evaluation shown below. When the full dataset is released, it will be possible to reproduce the complete experiments and results described in the paper.

Use the provided small TMD dataset for the example evaluation below; the full datasets (when distributed) can be used to reproduce the full set of results.

To reproduce our results you need the first `0` and `60` seconds with durations `3`, `5`, and `10` seconds:

```bash
create-tmd-dataset --traintrack-dataset-path datasets/raw_datasets/trainride_recordings_sample/dataset.yml --no-trainride-dataset-path datasets/raw_datasets/no_train_recordings_sample/dataset.yml --trainride-start-seconds 0 --duration 3 --output-path data/tmd/tmd_s0_d3.pkl
create-tmd-dataset --traintrack-dataset-path datasets/raw_datasets/trainride_recordings_sample/dataset.yml --no-trainride-dataset-path datasets/raw_datasets/no_train_recordings_sample/dataset.yml --trainride-start-seconds 0 --duration 5 --output-path data/tmd/tmd_s0_d5.pkl
create-tmd-dataset --traintrack-dataset-path datasets/raw_datasets/trainride_recordings_sample/dataset.yml --no-trainride-dataset-path datasets/raw_datasets/no_train_recordings_sample/dataset.yml --trainride-start-seconds 0 --duration 10 --output-path data/tmd/tmd_s0_d10.pkl

create-tmd-dataset --traintrack-dataset-path datasets/raw_datasets/trainride_recordings_sample/dataset.yml --no-trainride-dataset-path datasets/raw_datasets/no_train_recordings_sample/dataset.yml --trainride-start-seconds 60 --duration 3 --output-path data/tmd/tmd_s60_d3.pkl
create-tmd-dataset --traintrack-dataset-path datasets/raw_datasets/trainride_recordings_sample/dataset.yml --no-trainride-dataset-path datasets/raw_datasets/no_train_recordings_sample/dataset.yml --trainride-start-seconds 60 --duration 5 --output-path data/tmd/tmd_s60_d5.pkl
create-tmd-dataset --traintrack-dataset-path datasets/raw_datasets/trainride_recordings_sample/dataset.yml --no-trainride-dataset-path datasets/raw_datasets/no_train_recordings_sample/dataset.yml --trainride-start-seconds 60 --duration 10 --output-path data/tmd/tmd_s60_d10.pkl
```

## Evaluate TMD detectors

Use `evaluate-tmd` to run our evaluation, which is an Optuna-based hyperparameter search on a TMD dataset. For the purposes of the submission and quick reproduction, run the example evaluation on the provided small dataset `datasets/tmd_datasets/tmd_s60_d10_small.pkl`.

The search commands are exposed as subcommands of the `evaluate-tmd` Typer app (`optuna-search-fft`, `optuna-search-psd`, `optuna-search-goertzel`) and accept options such as `--n-trials`, `--runs`, and `--sliding-window-length`.

Example (use the small example dataset included in `datasets/tmd_datasets`):

```bash
dataset="datasets/tmd_datasets/tmd_s60_d10_small.pkl"
for sw in 1 3 6 9 12; do
  evaluate-tmd optuna-search-fft "$dataset" --output-path results/tmd --runs 5 --sliding-window-length $sw
done
```
