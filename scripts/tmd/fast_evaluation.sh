#!/usr/bin/env bash
set -euo pipefail

# Fast alternative to run_parameter_search.sh.
#
# The full search explores 5000 trials for each of the 15 datasets and 5 sliding
# windows (375,000 evaluations) and runs for days. This reads the result CSVs
# that search already produced, takes the best parameter set for each dataset
# across all sliding windows, and re-runs just that one -- 15 evaluations -- to
# check the recorded metrics still come out the same.

# Paths
RESULTS_DIR="${TMD_RESULTS_DIR:-results/tmd}"
DATASET_DIR="${TMD_DATASET_DIR:-datasets/tmd_datasets}"
OUTPUT_CSV="${TMD_FAST_OUTPUT_CSV:-${RESULTS_DIR}/fast_evaluation.csv}"

SELECT_METRIC="${TMD_FAST_METRIC:-mcc_test}"
TOLERANCE="${TMD_FAST_TOLERANCE:-1e-6}"
THRESHOLD_METRIC="${TMD_FAST_THRESHOLD_METRIC:-mcc}"

NPROC=$(nproc)
DATASET_JOBS="${TMD_FAST_DATASET_JOBS:-0}"
N_JOBS="${TMD_FAST_N_JOBS:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

cd "${REPO_ROOT}"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  echo "Build it first with:  scripts/build.sh"
  exit 1
fi

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "Search results not found: ${RESULTS_DIR}"
  echo "Run scripts/tmd/run_parameter_search.sh first, or set TMD_RESULTS_DIR."
  exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Datasets not found: ${DATASET_DIR}"
  echo "Build them first with scripts/dataset_creation/tmd_datasets.sh."
  exit 1
fi

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Dataset dir:  ${DATASET_DIR}"
echo "Best row by:  ${SELECT_METRIC}   tolerance: ${TOLERANCE}"
echo "Parallelism:  ${DATASET_JOBS:-auto} dataset(s) at a time, ${N_JOBS:-auto} precompute worker(s) each"

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  fast-evaluation \
  --results-dir "${RESULTS_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --output-csv "${OUTPUT_CSV}" \
  --metric "${SELECT_METRIC}" \
  --tolerance "${TOLERANCE}" \
  --threshold-metric "${THRESHOLD_METRIC}" \
  --dataset-jobs "${DATASET_JOBS}" \
  --n-jobs "${N_JOBS}" \
  "$@"
