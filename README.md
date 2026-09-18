# On the Same Track: Privacy Leaks in Electric Rail Transport via Magnetic Fields

This is the artifact repository for the publication:

> On the Same Track: Privacy Leaks in Electric Rail Transport via Magnetic Fields.
> Stefan Czybik, Pia Hanfeld and Konrad Rieck.
> 33rd ACM Conference on Computer and Communications Security (CCS), 2026.

| |                                           |
| --- |-------------------------------------------|
| Code (this repository) | <https://github.com/mlsec-group/magtrack> |
| Code, archived snapshot | <https://doi.org/10.5281/zenodo.22662413> |
| TrainTrack dataset | <https://doi.org/10.5281/zenodo.22206115> |
| License | CC BY-SA 4.0 (paper, code and data)       |

## Overview

### The idea

Electric trains especially in central and northern Europe draw their power from an overhead line that carries alternating current at **16.7 Hz**.
That current produces a magnetic field, and the field reaches into the passenger cabin.
A smartphone magnetometer can measure it.

This matters for privacy because the magnetometer is **not protected by a permission** on Android or iOS.
Any installed app can read it silently.
Two things follow:

1. The 16.7 Hz component is only present on (or very close to) an electric train. Its presence tells an app that the user is **riding a train**.
2. The strength of the field changes over time, driven by how the train and every other train on the same overhead line accelerates and brakes. 
   All passengers in the same train see the same time pattern.
   Comparing two recordings therefore tells an app whether two people **travelled together**.

### The two attacks

**Transport mode detection (TMD).**
We take a short chunk of the magnetometer signal, detrend it, apply a Hann window and an FFT, and measure the signal-to-noise ratio in a narrow band around 16.7 Hz.
If the SNR is above a threshold, the chunk is a train ride.
A majority vote over consecutive chunks stabilises the decision.
The free parameters (signal band width, noise bandwidth, SNR threshold) are found with an Optuna search.
There are no learned features and no multi-sensor fusion — one sensor and one spectral feature.

**Colocation inference.**
We compare the 16.7 Hz traces of two recordings made by two different phones and decide whether they come from the same ride.
Both traces are cut to the same length and aligned by device clock, so length itself carries no information.
We implement three variants:

| Variant | How it decides | Script directory |
| --- | --- | --- |
| Baseline (Nguyen et al.) | DDTW on the **absolute** field magnitude, threshold fitted on the train split | `scripts/coloc_baseline/` |
| Distance-based (ours) | DTW on the **16.7 Hz trace**, threshold fitted on the train split | `scripts/coloc_distance/` |
| Learning-based (ours) | 1D CNN over a pair of chunks, plus a majority vote over the chunks | `scripts/coloc_ml/` |

Every variant is evaluated over the same grid: how much of a ride is observed, the sampling rate of the trace (10–60 Hz), and the rolling window used for smoothing (1–150 samples).
The grid is the point — it shows how *little* data an attacker actually needs.

### The results you should get

These are the headline numbers from the paper.
The experiments below reproduce them.

| Claim | Attack | Result                            | Paper |
| --- | --- |-----------------------------------| --- |
| C1 | TMD, single chunk | MCC 0.83, F1 0.94                 | Table 4 |
| C1 | TMD, with majority vote | MCC 0.93, F1 0.98                 | Table 4 |
| C2 | Colocation, baseline (DDTW on absolute field) | MCC 0.09                          | Table 8 |
| C3 | Colocation, distance-based (DTW on 16.7 Hz trace) | MCC 0.50                          | Tables 6 and 8 |
| C4 | Colocation, learning-based (CNN) | MCC 0.79, F1 0.89                 | Figures 6–7, Table 8 |
| C4 | CNN is ~16× faster than DTW | < 1 ms vs. up to ~600 ms per pair | Figure 8 |

Both attacks work best on the **first 300 s** of a ride.
Five minutes of travel already carry enough structure.

### The dataset

The second half of the artifact is **TrainTrack**, [released on Zenodo](https://zenodo.org/records/22206115).
All inertial sensors were recorded, not only the magnetometer, so the dataset is useful well beyond this paper.
To the best of our knowledge it is the largest collection of train rides with inertial sensor data.

## Requirements

**Hardware**

- **500 GB free disk space.** About 137 GB download, roughly the same again after extraction, plus the generated datasets.
- **64 GB RAM.** The colocation evaluation holds a whole dataset in memory per worker.
- **32 CPU cores.** Everything except the CNN is CPU-bound and parallel. More cores means proportionally less wall-clock time.
- **CUDA GPU.** The learning-based pipeline uses this. Without one, training and evaluation are much slower.

**Software**

- A Linux system with a shell.
- [Apptainer](https://apptainer.org/) — everything runs inside the container,
  so no Python setup on the host is needed.
- `curl`, `unzip` and `sha256sum` for the download.

Tested on Debian 13 with Apptainer 1.5.2. We recommend that combination to
avoid version problems.

## Setup

Run everything **from the repository root**.

### 1. Download the data

```bash
scripts/download.sh
```

This fetches the 137 GB TrainTrack record from Zenodo, checks every file against `sha256sums.txt` and unpacks it into:

```
traintrack-dataset/          # train recordings (.zip) + traintrack.yml
no-trainride-dataset/        # non-train recordings + no_trainride_data.yml
```

The download is resumable — re-run the script and it only fetches what is missing. Extraction takes about 20 minutes, the download depends on your connection.

| Variable | Meaning |
| --- | --- |
| `ZENODO_SKIP_EXTRACT=1` | download only, unpack later |
| `ZENODO_VERIFY_ONLY=1` | only re-check what is already on disk |
| `ZENODO_DEST_DIR` | where the archives are stored (default `traintrack-zenodo/`) |
| `ZENODO_FILE_FILTER` | regex, for partial downloads |

### 2. Build the container

```bash
scripts/build.sh
```

This creates `python.sif` in the repository root.
Every script expects it there.
To run a single command inside it:

```bash
apptainer exec python.sif <command>
```

### 3. Check the setup

```bash
scripts/check.sh
```

This verifies the host tools, the container and its commands, free disk space, RAM, the GPU (if any), and that every recording listed in the dataset `.yml` files is actually present.
It exits non-zero if something required is missing, warnings are fine.

## Repository layout

```
apptainer.def                 container definition
requirements.txt              pinned Python dependencies
pyproject.toml                package metadata and the CLI entry points

src/magtrack/
  cli/                        the CLI commands (see the table below)
  utils/                      dataset building, filters, distances, model, evaluation
  paper/                      figures, tables and LaTeX variables

scripts/
  download.sh build.sh check.sh      set-up helpers
  prepare_datasets.sh                runs all four dataset-creation steps
  dataset_creation/                  build the datasets from the raw recordings
  tmd/                               transport mode detection            (E1)
  coloc_baseline/                    colocation baseline, DDTW           (E2)
  coloc_distance/                    colocation, distance-based, DTW     (E3)
  coloc_ml/                          colocation, learning-based, CNN     (E4)
  benchmark/                         inference time of both approaches

datasets/                     generated datasets (created by the scripts)
coloc_model/                  the trained CNN and its hyperparameters
results/                      output of all experiments, and our own results
  paper/                      the figures, tables and variables used in the paper
model_checkpoints/  runs/     training checkpoints and tensorboard logs
```

Each experiment directory holds an `evaluate.sh` (or a search/train script) and a `paper.sh`.
`evaluate.sh` produces CSV files, `paper.sh` turns them into the figures and tables.

Here are the commands available in the container:

| Command | Purpose |
| --- | --- |
| `create-tmd-dataset` | build one TMD dataset |
| `create-colocation-datasets` | build colocation datasets (whole grid in one pass) |
| `traintrack-dataset` | corpus statistics and dataset LaTeX variables |
| `evaluate-tmd` | Optuna search for our TMD detector |
| `fast-evaluation` | re-run only the best TMD parameters |
| `evaluate-nor-tmd` | NOR-TMD baseline pipeline (optional, 4 stages) |
| `evaluate-coloc-distance` | colocation via `dtw`, `ddtw`, `euclidean`, `cosine` |
| `hyperparameter-search`, `train-ml-model` | CNN search and training |
| `ml-inference`, `evaluate-majority-ml` | CNN inference and majority vote |
| `benchmark-inference` | time one comparison for both approaches |
| `plot-results`, `generate-tables`, `generate-latex-vars` | figures, tables, LaTeX variables |

## Preparing the datasets

All four experiments read generated datasets, so this step comes first:

```bash
scripts/prepare_datasets.sh
```

It runs four steps in order and prints the time each one took:

| Step | Builds | Into |
| --- | --- | --- |
| `dataset_creation/tmd_datasets.sh` | 15 TMD datasets (5 start offsets × 3 chunk durations) | `datasets/tmd_datasets/` |
| `dataset_creation/coloc_datasets.sh` | colocation datasets on the filtered 16.7 Hz trace | `datasets/coloc_datasets/` |
| `dataset_creation/coloc_baseline_datasets.sh` | colocation datasets on the absolute field | `datasets/coloc_baseline/` |
| `dataset_creation/paper.sh` | dataset statistics as LaTeX variables | `results/paper/vars/` |

The scripts run under GNU parallel inside the container and cap the number of workers by available memory.

| Variable | Meaning |
| --- | --- |
| `COLOC_PARALLEL_JOBS` | maximum parallel jobs (default 20, capped by RAM) |
| `COLOC_MEMORY_PER_JOB_GIB` | memory budgeted per job (default 8) |
| `COLOC_MEMORY_FREE_TO_START` | free memory needed to start a job (default 12G) |
| `PAPER_REBUILD_STATS=0` | reuse the shipped `traintrack_stats.yml` instead of recomputing it |

## Reproducing the results

The four experiments are independent and each one is one script.

**To compare**, move our results aside before re-running:

```bash
mv results results_org
```

### E1 — Transport mode detection

An Optuna search over the SNR threshold and the band widths, for each of the 15 datasets and each of the 5 majority-vote window lengths.

```bash
scripts/tmd/run_parameter_search.sh    # the full search
scripts/tmd/paper.sh                   # figure, table, LaTeX variables
```

The full search is 5 000 trials × 15 datasets × 5 sliding windows = 375 000 evaluations which takes **about 3 days**.
If you only want to confirm the reported numbers, use the fast path instead:

```bash
scripts/tmd/fast_evaluation.sh         # minutes
```

It reads the result CSVs the search produced, takes the best parameter set per dataset, and re-runs just those 15 evaluations through the same code path.
No sampler is involved when parameters are read back, so the numbers reproduce exactly.

**Output:**
one CSV per dataset and sliding window in `results/tmd/`.
`paper.sh` picks the best configuration and writes
`results/paper/figures/tmd_results_recording_time.png` (Figure 5),
`results/paper/tables/tmd_summary_table.tex` (Table 5) and
`results/paper/vars/tmd_results.tex`.

**Expect:**
MCC 0.83 / F1 0.94 for a single 10 s chunk, MCC 0.93 / F1 0.98 with  a 12-chunk majority vote over the first 300 s.

| Variable | Meaning |
| --- | --- |
| `TMD_RESULTS_DIR` | where the search results are (default `results/tmd`) |
| `TMD_DATASET_DIR` | where the datasets are (default `datasets/tmd_datasets`) |
| `TMD_FAST_METRIC` | metric used to pick the best row (default `mcc_test`) |
| `TMD_FAST_TOLERANCE` | allowed deviation when reproducing (default `1e-6`) |
| `TMD_SEARCH_MEMORY_PER_JOB_GIB` | memory budgeted per search process (default 4) |

<details>
<summary>Optional: the NOR-TMD baseline (Appendix A.1 of the paper)</summary>

Our reimplementation of Skretting et al., an XGBoost classifier over rolling-window statistics of several sensors.
It needs a separate download, the [NOR-TMD dataset from Kaggle](https://www.kaggle.com/datasets/scholarone/nor-tmd), saved as `nor-tmd.zip` in the repository root.

```bash
scripts/tmd/nor_tmd_baseline.sh
```

It runs four stages: parse (CSV → DuckDB), preprocess (rolling windows), pivot (sensors to columns, split by OS), train (XGBoost, 10 runs).
The result table is at the end of `results/nor_tmd/nor_tmd_train.log`.
</details>

### E2 — Colocation baseline (DDTW on the absolute field)

Our reimplementation of Nguyen et al.
It computes the DDTW distance between all trace pairs, fits the best distance threshold on the training split, and applies it to the held-out test split.

```bash
scripts/coloc_baseline/evaluate.sh
scripts/coloc_baseline/paper.sh
```

**Output:**
one row per dataset in `results/coloc_baseline/ddtw.csv`, plus
`long_distance_ddtw.csv` and `regional_ddtw.csv`.
`paper.sh` writes `results/paper/figures/coloc_abs_ddtw.png` and
`results/paper/vars/coloc_distances_abs_ddtw.tex`.

**Expect:**
a best MCC of about **0.09**.
In our setting all compared traces have the same length, which removes trace length as a discriminating feature and leaves only the shape of the raw field.
That is a much harder task than in the original study.

### E3 — Colocation, distance-based (DTW on the 16.7 Hz trace)

The same scheme as E2, but on the filtered and normalized 16.7 Hz trace and with DTW.

```bash
scripts/coloc_distance/evaluate.sh
scripts/coloc_distance/paper.sh
```

**Output:**
`results/coloc_distance/dtw_r1.csv`, `long_distance_dtw_r1.csv`,
`regional_dtw_r1.csv`. `paper.sh` writes
`results/paper/figures/coloc_dtw1.png` and
`results/paper/vars/coloc_distances_dtw_r1.tex`.

**Expect:**
a best MCC of **0.50 ± 0.04**, at 300 s of recording and 60 data points per second.

Both E2 and E3 share these variables:

| Variable | Meaning |
| --- | --- |
| `COLOC_TRAIN_TYPES` | which subsets to sweep (default `all long_distance regional`) |
| `COLOC_RESULTS_CSV` | output CSV (also decides the output directory) |
| `COLOC_DATA_BASE_DIR` | where the datasets are |
| `COLOC_RUNS` | evaluation repeats per dataset (default 10) |
| `COLOC_TEST_FRACTION` | held-out fraction, split by journey (default 0.3) |
| `COLOC_SEED` | random seed (default `magtrack`) |
| `COLOC_METRIC` | metric the threshold is optimised for (default `mcc`) |
| `COLOC_WORKERS` | worker processes (default: all cores) |
| `COLOC_SAMPLE` | evaluate only a sample of pairs, for a quick smoke test |

### E4 — Colocation, learning-based (CNN)

A 1D CNN takes two chunks as two input channels and classifies the pair as colocated or not.
Per-chunk decisions are aggregated with a majority vote.

```bash
scripts/coloc_ml/hyperparameter_search.sh   # optional, not reproducible
scripts/coloc_ml/train.sh                   # optional, not reproducible
scripts/coloc_ml/evaluate.sh                # ~5 h
scripts/coloc_ml/paper.sh                   # ~1 min
```

**The search and the training are not bit-exact reproducible**
(GPU non-determinism, and the search is a genetic NSGA-II sampler over 1 000 trials).
We ship the model from our own run, so you can skip the first two scripts and go straight to `evaluate.sh` to reproduce the reported numbers.

```
coloc_model/model_hparams.yaml     # the configuration the search selected
coloc_model/colocation_net.pth     # the trained weights
```

If you do run the search and the training, they overwrite those two files.

`evaluate.sh` runs in six steps: it generates the evaluation job scripts, runs them for `all`, `long_distance` and `regional`, then runs the majority vote and computes its metrics.

**Output:**
`results/coloc_ml/evaluation_results.results_ml.csv` and the same
under `long_distance/` and `regional/`, plus `results/coloc_ml/majority_vote/master_metrics_by_k.csv`.
`paper.sh` writes
`results/paper/figures/mcc_paper_single_evaluation_results.results_ml.pdf` (MCC heatmaps, Figure 6),
`results/paper/figures/mcc_chunk60s_win1_10Hz.pdf` (majority vote, Figure 7),
`results/paper/tables/mcc_train_types.tex` (Table 7),
`results/paper/tables/hparam_search_table.tex` (Table 9, only if `results/coloc_ml/hyperparam_search.db` exists) and
`results/paper/vars/coloc_ml.tex`.

**Expect:** a best MCC of **0.79** and F1 of **0.89**, at 300 s of recording,
60 s chunks and a rolling window of 1.

All three scripts detect a CUDA GPU and fall back to the CPU if there is none.
Training progress can be watched with `tensorboard --logdir runs/`, checkpoints
land in `model_checkpoints/`.

| Variable | Meaning |
| --- | --- |
| `COLOC_NUM_GPUS` | GPUs to use (default: auto-detected, 0 without CUDA) |
| `COLOC_PARALLEL_JOBS` | concurrent evaluation jobs (default 4 with a GPU, `nproc-1` on CPU) |
| `COLOC_N_TRIALS` | Optuna trials in the search (default 1000) |
| `COLOC_MAX_EPOCHS` | epochs per search trial (default 200) |
| `COLOC_TRAIN_EPOCHS` | epochs for the final training (default 1000) |
| `COLOC_BATCH_SIZE`, `COLOC_LEARNING_RATE`, `COLOC_WEIGHT_DECAY` | final training settings |
| `COLOC_DATA_DIR`, `COLOC_DATA_DIR_LONG_DISTANCE`, `COLOC_DATA_DIR_REGIONAL` | dataset directories |

### Inference time

How long a single comparison takes, for both approaches, on the CPU.

```bash
scripts/benchmark/evaluate.sh
scripts/benchmark/paper.sh
```

It times 50 randomly selected pairs per dataset, repeated five times.
Loading and preprocessing are excluded.

**Output:**
one row per timed pair in
`results/benchmark/inference_times.csv`, and
`results/paper/figures/runtime_comparison.pdf` (Figure 8) plus
`results/paper/vars/runtime.tex`.

**Expect:** DTW scales linearly with both the sampling rate and the recording
length. The CNN is independent of the
sampling rate and grows only with the number of chunks.
Be aware that the exact numbers depend on the CPU used, so they are not reproducible.

## Our results, and how to compare

The repository ships the output of our own runs, so you can compare against
them directly instead of only inspecting your own.

| Path | Contents |
| --- | --- |
| `results/tmd/` | TMD parameter search, one CSV per dataset and sliding window |
| `results/coloc_distance/` | DTW sweep, 114 rows for `all` and `long_distance`, 108 for `regional` |
| `results/coloc_baseline/` | DDTW baseline |
| `results/coloc_ml/` | CNN evaluation, per train type and majority vote |
| `results/benchmark/` | measured inference times, one row per timed pair |
| `results/paper/` | the figures, tables and LaTeX variables used in the paper |

After running the experiments and `diff` the two directories, or just use `git diff` you can see if your results differ from ours.
Ignore the `total_distance_time_s` and `total_cpu_hours` columns.
They record measured runtime and legitimately differ between machines.

## Ethics

Running this code on this data poses no security or privacy risk.
The data are nevertheless anonymized recordings of journeys made by real people.
Deanonymising them should not be easy, but it would mean processing movement profiles — personal data, without consent — and therefore needs ethical and legal approval.
Please contact us if you plan research in that direction.

Our full ethical considerations, including how participants consented and how the data was anonymized, are in [ETHICS.md](ETHICS.md).

## Citation

```bibtex
@inproceedings{czybik2026magtrack,
  title     = {On the Same Track: Privacy Leaks in Electric Rail Transport via Magnetic Fields},
  author    = {Czybik, Stefan and Hanfeld, Pia and Rieck, Konrad},
  booktitle = {Proceedings of the 33rd ACM Conference on Computer and Communications Security (CCS)},
  year      = {2026},
  doi       = {10.1145/3830454.3846554}
}
```

Code and dataset are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
