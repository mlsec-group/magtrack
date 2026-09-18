#!/usr/bin/env bash
set -euo pipefail

# Paths
TRAINTRACK_PATH="traintrack-dataset/traintrack.yml"
NO_TRAINRIDE_PATH="no-trainride-dataset/no_trainride_data.yml"
OUTPUT_BASE_DIR="datasets/tmd_datasets"

# Runtime options
PARALLEL_JOBS=15
COMMANDS_FILE="${OUTPUT_BASE_DIR}/create_tmd_commands.txt"
PARALLEL_JOBLOG="${OUTPUT_BASE_DIR}/create_tmd_parallel_joblog.tsv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

# Parameter grid
TRAINRIDE_START_SECONDS=(0 60 300 600 900)
DURATIONS=(3 5 10)

mkdir -p "${OUTPUT_BASE_DIR}"
: > "${COMMANDS_FILE}"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

for start_seconds in "${TRAINRIDE_START_SECONDS[@]}"; do
  for duration in "${DURATIONS[@]}"; do
    output_filename="tmd_s${start_seconds}_d${duration}.pkl"
    output_path="${OUTPUT_BASE_DIR}/${output_filename}"
    log_path="${OUTPUT_BASE_DIR}/${output_filename%.pkl}.log"

    echo "create-tmd-dataset --traintrack-dataset-path \"${TRAINTRACK_PATH}\" --no-trainride-dataset-path \"${NO_TRAINRIDE_PATH}\" --trainride-start-seconds \"${start_seconds}\" --duration \"${duration}\" --output-path \"${output_path}\" >> \"${log_path}\" 2>&1" >> "${COMMANDS_FILE}"
  done
done

num_commands=$(wc -l < "${COMMANDS_FILE}")
if (( num_commands == 0 )); then
  echo "No valid commands generated."
  exit 1
fi

echo "Generated ${num_commands} commands in ${COMMANDS_FILE}"
echo "Using apptainer image: ${PYTHON_SIF}"
echo "Running with GNU parallel (${PARALLEL_JOBS} jobs)..."

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  parallel --jobs "${PARALLEL_JOBS}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}"

echo "Done. GNU parallel job log: ${PARALLEL_JOBLOG}"
