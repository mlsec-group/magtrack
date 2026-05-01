### Create Colocation datasets

**Warning:** The example colocation data included in `datasets/raw_datasets/` contains a very small amount of sensor recordings.
With this limited data it is not possible to construct valid colocation datasets that reflect real-world colocated recordings (colocation requires multiple synchronized recordings of the same event). The commands shown below are illustrative — use the full released dataset to generate valid colocation datasets and reproduce the published results.
We provide pre-generated colocation datasets in `datasets/colocation_datasets/` for testing.


You can run colocation dataset generation with:
- `duration`: `5 10 20 30 60`
- `sampling-rate`: `10 20 40 60`
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

### Evaluate Colocation Distance

Run classical distance-based colocation evaluation (DTW, DDTW, etc.) using the `evaluate-coloc-distance` command.

The command has the following simple invocation form:

```bash
evaluate-coloc-distance <method> <dataset.pkl> <output.csv>
```

Example:

```bash
evaluate-coloc-distance dtw "${DATASET}" "/home/space/mlsec/magtrack2/results_dtw/dtw_r10.csv"
```

This will run the DTW-based evaluation on the provided dataset and write per-evaluation results to the given CSV file.


### ML Pipeline

We also provide an end-to-end Machine Learning pipeline to train and evaluate CNN models for colocation detection.

#### Hyperparameter Search

**Warning:** The example colocation data included in `datasets/coloc_datasets/` are very few.
With this limited data it is not possible to run a reasonable hyperparameter search.

To reproduce the coloc_ml results, begin by running the hyperparameter search using Optuna. This will search for the best model and training configuration for colocation detection.

Run the following command:

```bash
hyperparameter-search --base-data-dir /path/to/normalized_trace_all_trains/
```

This will evaluate various model configurations and save the best hyperparameters to:

- `coloc_model/model_hparams.yaml`

Make sure the dataset directory matches your setup.


#### Train ML Model
To train the CNN model using the optimized hyperparameters found in `coloc_model/model_hparams.yaml`, run:

**Warning:** You might need to adapt the `model_hparams.yaml`.

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


#### Evaluate ML Model

To evaluate the trained model on test datasets, use the new inference script:

```bash
ml-inference --dataset-path datasets/coloc_datasets/all_coloc_first60_5s_window50_40Hz.pkl --model-path coloc_model/colocation_net.pth --hparams-path coloc_model/model_hparams.yaml --results-dir data/results/coloc_ml
```

This will compute evaluation metrics and save results to the specified results directory.

#### 4. Evaluate Majority-Vote ML Model

To evaluate the trained model with segment-pair majority voting, use the following command:

```bash
evaluate-majority-ml --dataset-path datasets/coloc_datasets/all_coloc_first60_5s_window50_40Hz.pkl --model-path coloc_model/colocation_net.pth --hparams-path coloc_model/model_hparams.yaml --results-dir data/coloc_ml
```

This will:
- Save per-pair prediction CSVs for each dataset under `data/results/coloc_ml/majority_vote/` (one CSV per dataset, e.g. `all_coloc_first0_10s_window100_20Hz.csv`).
- Save summary metrics (F1, MCC, etc.) for each dataset as CSVs in the same directory.
