# Magtrack Artifacts

## Setup

### Apptainer Setup
Assuming Apptainer is already installed, build the container image from the repository root.

Run the build command:

```bash
apptainer build python.sif apptainer.def
```

This creates `python.sif` in the repository root.

You can run the commands inside the container with:

```bash
apptainer exec python.sif <command>
```

### Python Setup

For we recomment using a Python virtual environment to manage dependencies.
We recommend to use Python 3.14.

From the repository root in editable mode:

```bash
pip install -e .
```

This installs the CLI commands:
- `evaluate-nor-tmd`
- `create-tmd-dataset`
- `hypersearch-tmd`
- `train-ml-model`
- `hyperparameter-search`
- `evaluate-ml-model`
- `evaluate-majority-ml`

## Reproducing results

### NOR-TMD reimplementation

This part reimplements the NOR-TMD evaluation pipeline as a 4-stage CLI workflow:
1. parse raw CSV from zip into DuckDB
2. preprocess sensor streams into rolling-window statistics
3. pivot sensor ids to model-ready feature columns
4. train/evaluate an XGBoost classifier

#### Dataset

Download the NOR-TMD dataset from Kaggle:
https://www.kaggle.com/datasets/scholarone/nor-tmd

You need the dataset zip file path for the CLI `--zip-path` argument.

#### Reproduce results (zipfile argument + 10 runs)

Run the full pipeline end-to-end with 10 training runs:

```bash
evaluate-nor-tmd --zip-path path/to/nor_tmd.zip --runs 10
```

Notes:
- `--zip-path` is required.
- `--runs 10` performs 10 train/eval runs and prints a mean+-std summary table.

#### Stage-by-stage usage (optional)

If you want to run each stage separately:

```bash
evaluate-nor-tmd parse --zip-path path/to/nor_tmd.zip
evaluate-nor-tmd preprocess
evaluate-nor-tmd pivot
evaluate-nor-tmd train --runs 10
```

#### Output artifacts (default paths)

All outputs are written under `data/nor_tmd/` by default:
- `nor_tmd_complete.db` (DuckDB from parse)
- `segmented_df.parquet` (preprocessed windows)
- `data_android_centered.parquet` (pivoted Android dataset used for train)
- `data_ios_centered.parquet` (pivoted iOS dataset)

### Create TMD datasets

Use `create-tmd-dataset` to generate trainride/no-trainride chunks.

Required arguments:
- `--traintrack-dataset-path` (YAML with trainride zip files and trips metadata)
- `--no-trainride-dataset-path` (YAML with no-trainride zip files)
- `--output-path` (output file path, e.g. `.pkl`)

To reproduce our results you need the first `0`, `60`, and `300` seconds with durations `3`, `5`, and `10` seconds:

```bash
create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 0 --duration 3 --output-path data/tmd/tmd_s0_d3.pkl
create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 0 --duration 5 --output-path data/tmd/tmd_s0_d5.pkl
create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 0 --duration 10 --output-path data/tmd/tmd_s0_d10.pkl

create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 60 --duration 3 --output-path data/tmd/tmd_s60_d3.pkl
create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 60 --duration 5 --output-path data/tmd/tmd_s60_d5.pkl
create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 60 --duration 10 --output-path data/tmd/tmd_s60_d10.pkl

create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 300 --duration 3 --output-path data/tmd/tmd_s300_d3.pkl
create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 300 --duration 5 --output-path data/tmd/tmd_s300_d5.pkl
create-tmd-dataset --traintrack-dataset-path data/traintrack_sample/dataset.yml --no-trainride-dataset-path data/traintrack_no_train_sample/dataset.yml --trainride-start-seconds 300 --duration 10 --output-path data/tmd/tmd_s300_d10.pkl
```

### Evaluate TMD detectors

Use `evaluate-tmd` to run our evaluation, that is an optuna-based hyperparameter search on the created datasets.

For each dataset, both FFT- and PSD-based detectors are searched over sliding window lengths `1`, `3`, `6`, `9`, and `12` with `5000` trials and `5` evaluation runs each.

```bash
for dataset in \
    data/tmd/tmd_s0_d3.pkl \
    data/tmd/tmd_s0_d5.pkl \
    data/tmd/tmd_s0_d10.pkl \
    data/tmd/tmd_s60_d3.pkl \
    data/tmd/tmd_s60_d5.pkl \
    data/tmd/tmd_s60_d10.pkl \
    data/tmd/tmd_s300_d3.pkl \
    data/tmd/tmd_s300_d5.pkl \
    data/tmd/tmd_s300_d10.pkl; do
  for sw in 1 3 6 9 12; do
    hypersearch-tmd optuna-search-fft "$dataset" --runs 5 --sliding-window-length $sw
    hypersearch-tmd optuna-search-psd "$dataset" --runs 5 --sliding-window-length $sw
  done
done
```

Results for each dataset are written as `<dataset>.optuna_search_fft.csv` / `<dataset>.optuna_search_psd.csv` next to the input file.

### Create Colocation datasets

You can run colocation dataset generation with:
- `duration`: `5 10 20 30 60`
- `sampling-rate`: `10 20 40 95`
- `rolling-window`: `1 5 10 50 100 150`
- `trainride-start-seconds`: `0 60 300 600 900`

and skip combinations where:
- `duration * sampling_rate <= rolling_window`

Manual run for all valid combinations:

```bash
TRAINTRACK_PATH="/path/to/traintrack_dataset/dataset.yml"
OUTPUT_DIR="/path/to/normalized_trace_all"

mkdir -p "${OUTPUT_DIR}"

for duration in 5 10 20 30 60; do
  for sampling_rate in 10 20 40 60; do
    product=$((duration * sampling_rate))
    for rolling_window in 1 5 10 50 100 150; do
      if (( product <= rolling_window )); then
        continue
      fi
      for start_seconds in 0 60 300 600 900; do
        create-colocation-dataset \
          --dataset-path "${TRAINTRACK_PATH}" \
          --duration "${duration}" \
          --trainride-start-seconds "${start_seconds}" \
          --sampling-rate "${sampling_rate}" \
          --rolling-window "${rolling_window}" \
          --normalize trace \
          --output-path "${OUTPUT_DIR}"
      done
    done
  done
done
```


### ML Pipeline

We also provide an end-to-end Machine Learning pipeline to train and evaluate CNN models for colocation detection.

#### Step 1: Hyperparameter Search (coloc_ml)

To reproduce the coloc_ml results, begin by running the hyperparameter search using Optuna. This will search for the best model and training configuration for colocation detection.

Run the following command (inside the Apptainer container or your Python environment):

```bash
hyperparameter-search --base-data-dir /path/to/normalized_trace_all_trains/
```

This will evaluate various model configurations and save the best hyperparameters to:

- `coloc_model/model_hparams.yaml`

Make sure the dataset directory matches your setup.


**2. Train ML Model**
To train the CNN model using the optimized hyperparameters found in `coloc_model/model_hparams.yaml`, run:

```bash
train-ml-model /path/to/dataset.pkl
```

- The script will automatically use the hyperparameters from `coloc_model/model_hparams.yaml` (override with `--hparams-path` if needed).
- Training logs and metrics will be saved for visualization in TensorBoard under the `runs/` directory.
- The final trained model weights will be saved to `coloc_model/colocation_net.pth`.

You can launch TensorBoard to monitor training progress with:

```bash
tensorboard --logdir runs/
```

Make sure your dataset path matches the format expected by the pipeline (see dataset generation instructions above).


**3. Evaluate ML Model**
To evaluate the trained model on test datasets, use the new inference script:

```bash
python src/magtrack/cli/ml_inference.py --dataset-path /path/to/dataset.pkl --model-path coloc_model/colocation_net.pth --hparams-path coloc_model/model_hparams.yaml --results-dir results/
```

This will compute evaluation metrics and save results to the specified results directory.

**Parallel Evaluation Example:**
To efficiently evaluate all dataset subsets in parallel, use the provided script:

```bash
scripts/run_eval.sh
```

This script uses GNU parallel to run inference on all dataset splits and logs outputs for each job. You can customize the datasets and parallelization settings inside scripts/run_eval.sh.


**4. Evaluate Majority-Vote ML Model**
To evaluate the trained model with segment-pair majority voting, use the following command:

```bash
python src/magtrack/cli/evaluate_majority_ml.py --dataset-path /path/to/dataset.pkl --model-path coloc_model/colocation_net.pth --hparams-path coloc_model/model_hparams.yaml --results-dir results/coloc_ml/
```

This will:
- Save per-pair prediction CSVs for each dataset under `results/coloc_ml/majority_vote/` (one CSV per dataset, e.g. `all_coloc_first0_10s_window100_20Hz.csv`).
- Save summary metrics (F1, MCC, etc.) for each dataset as CSVs in the same directory.

**Parallel Evaluation Example:**
To run majority-vote evaluation on many datasets in parallel (e.g., on a cluster), use the provided batch script:

```bash
scripts/run_majority_hydra.sh
```

This script demonstrates how to dispatch jobs for all dataset splits using SLURM. You can adapt it for your environment and dataset list.

**5. Plotting Results**
After running the evaluation steps, you can generate publication-ready plots using the provided scripts in `src/magtrack/paper/`:

**A. Metric Heatmaps**
To generate heatmaps for metrics (accuracy, F1, MCC, etc.) from the summary CSVs:

```bash
python src/magtrack/paper/heatmaps.py results/coloc_ml/majority_vote/your_summary_metrics.csv --out-dir results/plots --output-format pdf
```

Replace `your_summary_metrics.csv` with the summary metrics file you want to plot. The script will output heatmaps to the specified directory in PDF (or PGF) format.

**B. Majority-Vote MCC Plots**
To process all majority-vote CSVs and plot MCC as a function of voting length and duration:

```bash
python src/magtrack/paper/plot_majority_ml.py
```

This script will:
- Aggregate metrics from all `results/coloc_ml/majority_vote/*_metrics_by_k.csv` files
- Generate MCC plots (PDF and PGF) in the results directory

You can further customize or adapt these scripts for your publication needs.