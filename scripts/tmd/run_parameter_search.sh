#!/usr/bin/env bash
set -euo pipefail

# Paths
DATASET_DIR="datasets/tmd_datasets"
OUTPUT_DIR="results/tmd"

# Runtime options
NPROC=$(nproc)
MAX_CONCURRENT_DATASETS=15
(( MAX_CONCURRENT_DATASETS > NPROC - 1 )) && MAX_CONCURRENT_DATASETS=$(( NPROC - 1 ))
MEMORY_FREE_TO_START="${TMD_SEARCH_MEMORY_FREE_TO_START:-8G}"
MEMORY_PER_JOB_GIB="${TMD_SEARCH_MEMORY_PER_JOB_GIB:-6}"
mem_gib=$(( $(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo) / 1024 / 1024 ))
max_by_mem=$(( mem_gib / MEMORY_PER_JOB_GIB ))
(( max_by_mem < 1 )) && max_by_mem=1
(( MAX_CONCURRENT_DATASETS > max_by_mem )) && MAX_CONCURRENT_DATASETS=${max_by_mem}
N_JOBS=$(( NPROC/2 - 1 < NPROC/MAX_CONCURRENT_DATASETS ? NPROC/2 - 1 : NPROC/MAX_CONCURRENT_DATASETS ))
(( N_JOBS < 2 )) && N_JOBS=2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

# Parameter grid
DATASETS=(
  "${DATASET_DIR}/tmd_s0_d3.pkl"
  "${DATASET_DIR}/tmd_s0_d5.pkl"
  "${DATASET_DIR}/tmd_s0_d10.pkl"
  "${DATASET_DIR}/tmd_s60_d3.pkl"
  "${DATASET_DIR}/tmd_s60_d5.pkl"
  "${DATASET_DIR}/tmd_s60_d10.pkl"
  "${DATASET_DIR}/tmd_s300_d3.pkl"
  "${DATASET_DIR}/tmd_s300_d5.pkl"
  "${DATASET_DIR}/tmd_s300_d10.pkl"
  "${DATASET_DIR}/tmd_s600_d3.pkl"
  "${DATASET_DIR}/tmd_s600_d5.pkl"
  "${DATASET_DIR}/tmd_s600_d10.pkl"
  "${DATASET_DIR}/tmd_s900_d3.pkl"
  "${DATASET_DIR}/tmd_s900_d5.pkl"
  "${DATASET_DIR}/tmd_s900_d10.pkl"
)
SLIDING_WINDOWS=(1 3 6 9 12)

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

for dataset in "${DATASETS[@]}"; do
  commands_file="${dataset}.run_tmd_commands.txt"
  : > "${commands_file}"
  dataset_name=$(basename "${dataset}")
  for sw in "${SLIDING_WINDOWS[@]}"; do
    log_path="$OUTPUT_DIR/${dataset_name}.sw${sw}.fft.log"
    echo "evaluate-tmd optuna-search-fft \"${dataset}\" --runs 5 --n-jobs ${N_JOBS} --sliding-window-length ${sw} --output-path ${OUTPUT_DIR} >> \"${log_path}\" 2>&1" >> "${commands_file}"
  done
done

mkdir -p "${OUTPUT_DIR}"

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Running with GNU parallel: up to ${MAX_CONCURRENT_DATASETS} dataset(s) concurrently, 1 job per dataset at a time..."
echo "Memory: ${MEMORY_PER_JOB_GIB} GiB budgeted per process, ${MEMORY_FREE_TO_START} free required to start one."

pids=()
for dataset in "${DATASETS[@]}"; do
  commands_file="${dataset}.run_tmd_commands.txt"
  joblog="${dataset}.run_tmd_parallel_joblog.tsv"

  while (( $(jobs -rp | wc -l) >= MAX_CONCURRENT_DATASETS )); do
    wait -n || true
  done

  apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
    parallel --jobs 1 --memfree "${MEMORY_FREE_TO_START}" --joblog "${joblog}" < "${commands_file}" &
  pids+=($!)
done

wait "${pids[@]}" || true

echo "Done."
