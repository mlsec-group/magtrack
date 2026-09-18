#!/usr/bin/env bash
set -euo pipefail

# Paths
TRAINTRACK_PATH="traintrack-dataset/traintrack.yml"
OUTPUT_BASE_DIR="datasets/coloc_datasets"

# Runtime options
PARALLEL_JOBS="${COLOC_PARALLEL_JOBS:-20}"
MEMORY_FREE_TO_START="${COLOC_MEMORY_FREE_TO_START:-12G}"
MEMORY_PER_JOB_GIB="${COLOC_MEMORY_PER_JOB_GIB:-8}"
COMMANDS_FILE="${OUTPUT_BASE_DIR}/create_colocation_commands.txt"
PARALLEL_JOBLOG="${OUTPUT_BASE_DIR}/create_colocation_parallel_joblog.tsv"
TRACE_CACHE_DIR="${COLOC_TRACE_CACHE_DIR:-${OUTPUT_BASE_DIR}/trace_cache}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

# Parameter grid
DURATIONS=(5 10 20 30 60)
SAMPLING_RATES=(10 20 40 60)
ROLLING_WINDOWS=(1 5 10 50 100 150)
TRAINRIDE_START_SECONDS=(0 60 300 600 900)
SHORT_CHUNK_DURATION=5
SHORT_TRAINRIDE_START_SECONDS=(10 30)

DATASET_NAMES=("all" "long_distance" "regional")
DATASET_TRAIN_TYPES=("long_distance,regional" "long_distance" "regional")

mkdir -p "${OUTPUT_BASE_DIR}"
: > "${COMMANDS_FILE}"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

START_ARGS=()
for start_seconds in "${TRAINRIDE_START_SECONDS[@]}"; do
  START_ARGS+=("--trainride-start-seconds \"${start_seconds}\"")
done

SHORT_START_ARGS=("${START_ARGS[@]}")
for start_seconds in "${SHORT_TRAINRIDE_START_SECONDS[@]}"; do
  SHORT_START_ARGS+=("--trainride-start-seconds \"${start_seconds}\"")
done

SPEC_ARGS=()
for i in "${!DATASET_NAMES[@]}"; do
  dataset_name="${DATASET_NAMES[$i]}"
  output_dir="${OUTPUT_BASE_DIR}/normalized_trace_${dataset_name}_trains"
  mkdir -p "${output_dir}"
  SPEC_ARGS+=("--dataset-spec \"${dataset_name}:${DATASET_TRAIN_TYPES[$i]}:${output_dir}\"")
done

CACHE_ARG=""
if [[ -n "${TRACE_CACHE_DIR}" ]]; then
  mkdir -p "${TRACE_CACHE_DIR}"
  CACHE_ARG="--trace-cache-dir \"${TRACE_CACHE_DIR}\""
fi

for rolling_window in "${ROLLING_WINDOWS[@]}"; do
  for duration in "${DURATIONS[@]}"; do
    for sampling_rate in "${SAMPLING_RATES[@]}"; do
      # Exclude combinations where duration * sampling_rate <= rolling_window.
      if (( duration * sampling_rate <= rolling_window )); then
        continue
      fi

      if (( duration == SHORT_CHUNK_DURATION )); then
        start_args=("${SHORT_START_ARGS[@]}")
      else
        start_args=("${START_ARGS[@]}")
      fi

      log_path="${OUTPUT_BASE_DIR}/create_colocation_${duration}s_window${rolling_window}_${sampling_rate}Hz.log"
      echo "create-colocation-datasets --dataset-path \"${TRAINTRACK_PATH}\" --output-path \"${OUTPUT_BASE_DIR}\" --duration \"${duration}\" --sampling-rate \"${sampling_rate}\" --rolling-window \"${rolling_window}\" ${start_args[*]} ${SPEC_ARGS[*]} --normalize trace ${CACHE_ARG} >> \"${log_path}\" 2>&1" >> "${COMMANDS_FILE}"
    done
  done
done

num_commands=$(wc -l < "${COMMANDS_FILE}")
if (( num_commands == 0 )); then
  echo "No valid commands generated."
  exit 1
fi

echo "Generated ${num_commands} commands in ${COMMANDS_FILE}"
echo "Using apptainer image: ${PYTHON_SIF}"

if [[ -n "${TRACE_CACHE_DIR}" ]]; then
  echo "Filling trace cache in ${TRACE_CACHE_DIR}..."
  apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
    create-colocation-datasets --dataset-path "${TRAINTRACK_PATH}" --output-path "${OUTPUT_BASE_DIR}" \
    --duration 5 --sampling-rate 20 --rolling-window 1 --trainride-start-seconds 0 \
    --trace-cache-dir "${TRACE_CACHE_DIR}" --warm-trace-cache-only
fi

mem_gib=$(( $(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo) / 1024 / 1024 ))
max_jobs=$(( mem_gib / MEMORY_PER_JOB_GIB ))
(( max_jobs < 1 )) && max_jobs=1
(( PARALLEL_JOBS > max_jobs )) && PARALLEL_JOBS=${max_jobs}

echo "Running with GNU parallel (${PARALLEL_JOBS} jobs)..."

parallel_status=0
apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  parallel --jobs "${PARALLEL_JOBS}" --memfree "${MEMORY_FREE_TO_START}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}" || parallel_status=$?

failed=$(awk -F'\t' 'NR > 1 && $7 != 0' "${PARALLEL_JOBLOG}" | wc -l)
if (( failed > 0 )); then
  echo "ERROR: ${failed} of ${num_commands} jobs failed; their datasets are missing."
  echo "Failing jobs (see the matching .log files in ${OUTPUT_BASE_DIR}):"
  awk -F'\t' 'NR > 1 && $7 != 0 {print "  exit " $7 ": " $NF}' "${PARALLEL_JOBLOG}" | head -20
  echo "Fix the cause and re-run to rebuild them."
  exit 1
fi
if (( parallel_status != 0 )); then
  echo "ERROR: GNU parallel exited with status ${parallel_status}."
  exit 1
fi

echo "Done. GNU parallel job log: ${PARALLEL_JOBLOG}"
