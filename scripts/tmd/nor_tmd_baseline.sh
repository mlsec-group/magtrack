#!/usr/bin/env bash
set -euo pipefail

# Reproduces the NOR-TMD baseline results by running the four pipeline stages
# (parse -> preprocess -> pivot -> train) one after another inside apptainer.

# Paths
ZIP_PATH="${NOR_TMD_ZIP_PATH:-nor-tmd.zip}"
CSV_FILE="${NOR_TMD_CSV_FILE:-nor_tmd.csv}"
DATA_DIR="${NOR_TMD_DATA_DIR:-datasets/nor_tmd}"
OUTPUT_DIR="${NOR_TMD_OUTPUT_DIR:-results/nor_tmd}"

# Runtime options
RUNS="${NOR_TMD_RUNS:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "NOR-TMD archive not found: ${ZIP_PATH}"
  echo "Download it from https://www.kaggle.com/datasets/scholarone/nor-tmd"
  exit 1
fi

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

# The repo is always bound; the archive and the data dir may live elsewhere
# (e.g. on a share), so bind their parents too when they are outside the repo.
BIND_ARGS=(--bind "${REPO_ROOT}:${REPO_ROOT}")
add_bind() {
  local path
  path="$(cd "$(dirname "$1")" && pwd)"
  if [[ "${path}" != "${REPO_ROOT}" && "${path}" != "${REPO_ROOT}"/* ]]; then
    BIND_ARGS+=(--bind "${path}:${path}")
  fi
}
add_bind "${ZIP_PATH}"
add_bind "${DATA_DIR}/."
add_bind "${OUTPUT_DIR}/."

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Archive:      ${ZIP_PATH}"
echo "Data dir:     ${DATA_DIR}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Training runs: ${RUNS}"

# Run one pipeline stage, teeing its output into ${OUTPUT_DIR}/<stage>.log.
run_stage() {
  local stage="$1"
  shift
  local log_path="${OUTPUT_DIR}/nor_tmd_${stage}.log"

  echo
  echo "=== ${stage} === (log: ${log_path})"
  apptainer exec "${BIND_ARGS[@]}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
    evaluate-nor-tmd "${stage}" "$@" 2>&1 | tee "${log_path}"
}

# Stage 1/4: CSV inside the zip -> DuckDB. --overwrite keeps re-runs from
# appending the dataset to an already populated database.
run_stage parse --zip-path "${ZIP_PATH}" --csv-file "${CSV_FILE}" --db-dir "${DATA_DIR}" --overwrite

# Stage 2/4: rolling-window segmentation and aggregation -> segmented_df.parquet
run_stage preprocess --db-dir "${DATA_DIR}"

# Stage 3/4: pivot sensors to columns, split by OS -> data_{android,ios}_centered.parquet
run_stage pivot --data-dir "${DATA_DIR}"

# Stage 4/4: XGBoost train/eval, mean+-std summary over ${RUNS} runs
run_stage train --data-dir "${DATA_DIR}" --runs "${RUNS}"

echo
echo "Done. Artifacts in ${DATA_DIR}:"
echo "  nor_tmd_complete.db          (DuckDB from parse)"
echo "  segmented_df.parquet         (preprocessed windows)"
echo "  data_android_centered.parquet (pivoted Android dataset used for train)"
echo "  data_ios_centered.parquet    (pivoted iOS dataset)"
echo "Logs in ${OUTPUT_DIR}. The result table is at the end of ${OUTPUT_DIR}/nor_tmd_train.log"
