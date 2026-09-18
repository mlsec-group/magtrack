#!/usr/bin/env bash
set -euo pipefail

# Sweeps the distance-based colocation evaluation (DTW) over the recording
# length x sampling rate x rolling window grid and appends one row per dataset
# to a results CSV.
#
# The sweep is repeated for every train-type dataset (all / long_distance /
# regional).  Each one lives in its own directory and uses its own file name
# prefix, and gets its own results CSV: the "all" datasets keep the plain CSV
# name, the others are prefixed with the train type.

# Paths
DATA_BASE_DIR="${COLOC_DATA_BASE_DIR:-datasets/coloc_datasets}"
RESULTS_CSV="${COLOC_RESULTS_CSV:-results/coloc_distance/dtw_r1.csv}"

TRAIN_TYPES=(${COLOC_TRAIN_TYPES:-all long_distance regional})
CHUNK_DURATION="${COLOC_CHUNK_DURATION:-5}"

# Runtime options
RADIUS="${COLOC_RADIUS:-1}"
TEST_FRAC="${COLOC_TEST_FRACTION:-0.3}"
SEED="${COLOC_SEED:-magtrack}"
METRIC="${COLOC_METRIC:-mcc}"
RUNS="${COLOC_RUNS:-10}"
CHUNKSIZE="${COLOC_CHUNKSIZE:-32}"
WORKERS="${COLOC_WORKERS:-}"
SAMPLE="${COLOC_SAMPLE:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

# Parameter grid
FIRST_SECONDS=(10 30 60 300 600 900)
SAMPLING_RATES=(10 20 40 60)
ROLLING_WINDOWS=(1 5 10 50 100 150)

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ ! -d "${DATA_BASE_DIR}" ]]; then
  echo "Dataset base directory not found: ${DATA_BASE_DIR}"
  echo "Generate it first with scripts/dataset_creation/coloc_datasets.sh"
  exit 1
fi

CSV_DIR="$(dirname "${RESULTS_CSV}")"
CSV_BASE="$(basename "${RESULTS_CSV}")"
LOG_DIR="${CSV_DIR}/logs"
mkdir -p "${CSV_DIR}" "${LOG_DIR}"

EXTRA_ARGS=()
[[ -n "${WORKERS}" ]] && EXTRA_ARGS+=(--workers "${WORKERS}")
[[ -n "${SAMPLE}" ]] && EXTRA_ARGS+=(--sample "${SAMPLE}")

n_per_type=0
n_undersized=0
for first in "${FIRST_SECONDS[@]}"; do
  for sampling_rate in "${SAMPLING_RATES[@]}"; do
    for rolling_window in "${ROLLING_WINDOWS[@]}"; do
      if (( CHUNK_DURATION * sampling_rate <= rolling_window )); then
        n_undersized=$(( n_undersized + 1 ))
      else
        n_per_type=$(( n_per_type + 1 ))
      fi
    done
  done
done

n_runs_total=0
for train_type in "${TRAIN_TYPES[@]}"; do
  scan_dir="${DATA_BASE_DIR}/normalized_trace_${train_type}_trains"
  if [[ ! -d "${scan_dir}" ]]; then
    continue
  fi
  for first in "${FIRST_SECONDS[@]}"; do
    for sampling_rate in "${SAMPLING_RATES[@]}"; do
      for rolling_window in "${ROLLING_WINDOWS[@]}"; do
        if (( CHUNK_DURATION * sampling_rate <= rolling_window )); then
          continue
        fi
        scan_pkl="${scan_dir}/${train_type}_coloc_first${first}_${CHUNK_DURATION}s_window${rolling_window}_${sampling_rate}Hz.pkl"
        if [[ -f "${scan_pkl}" ]]; then
          n_runs_total=$(( n_runs_total + 1 ))
        fi
      done
    done
  done
done

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Dataset base: ${DATA_BASE_DIR}"
echo "Train types:  ${TRAIN_TYPES[*]}"
echo "Datasets:     <TYPE>_coloc_first<FIRST>_${CHUNK_DURATION}s_window<WINDOW>_<HZ>Hz.pkl"
echo "First (s):    ${FIRST_SECONDS[*]}"
echo "Sampling:     ${SAMPLING_RATES[*]} Hz"
echo "Windows:      ${ROLLING_WINDOWS[*]}"
echo "DTW radius:   ${RADIUS}"
echo "Evaluation:   ${RUNS} runs, test_frac=${TEST_FRAC}, metric=${METRIC}, seed=${SEED}"
echo "Results dir:  ${CSV_DIR}"
echo "Logs:         ${LOG_DIR}"
echo "Combinations: ${n_per_type} per train type (${n_undersized} not applicable)"
echo "Evaluations:  ${n_runs_total} dataset(s) present to evaluate"

n_done=0
n_total=0
n_types=0
n_run=0
MISSING=()
FAILED=()
SUMMARY=()

for train_type in "${TRAIN_TYPES[@]}"; do
  data_dir="${DATA_BASE_DIR}/normalized_trace_${train_type}_trains"
  if [[ "${train_type}" == "all" ]]; then
    results_csv="${CSV_DIR}/${CSV_BASE}"
  else
    results_csv="${CSV_DIR}/${train_type}_${CSV_BASE}"
  fi

  echo
  echo "##################################################################"
  echo "# train type: ${train_type}"
  echo "#   datasets: ${data_dir}"
  echo "#   results:  ${results_csv}"
  echo "##################################################################"

  if [[ ! -d "${data_dir}" ]]; then
    echo "SKIP: dataset directory not found: ${data_dir}"
    SUMMARY+=("${train_type}: directory missing (${data_dir})")
    continue
  fi
  n_types=$(( n_types + 1 ))

  type_done=0
  for first in "${FIRST_SECONDS[@]}"; do
    for sampling_rate in "${SAMPLING_RATES[@]}"; do
      for rolling_window in "${ROLLING_WINDOWS[@]}"; do
        if (( CHUNK_DURATION * sampling_rate <= rolling_window )); then
          continue
        fi
        n_total=$(( n_total + 1 ))

        dataset_name="${train_type}_coloc_first${first}_${CHUNK_DURATION}s_window${rolling_window}_${sampling_rate}Hz"
        pkl_path="${data_dir}/${dataset_name}.pkl"

        if [[ ! -f "${pkl_path}" ]]; then
          echo
          echo "SKIP: dataset not found: ${pkl_path}"
          MISSING+=("${dataset_name}.pkl")
          continue
        fi

        log_path="${LOG_DIR}/dtw_r${RADIUS}_${dataset_name}.log"
        n_run=$(( n_run + 1 ))
        echo
        echo "=== [${n_run}/${n_runs_total}] dtw ${train_type} first=${first}s ${sampling_rate}Hz window=${rolling_window} === (log: ${log_path})"

        status=0
        apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
          evaluate-coloc-distance dtw "${pkl_path}" "${results_csv}" \
          --radius "${RADIUS}" \
          "${EXTRA_ARGS[@]}" \
          --chunksize "${CHUNKSIZE}" \
          --test-frac "${TEST_FRAC}" \
          --seed "${SEED}" \
          --metric "${METRIC}" \
          --runs "${RUNS}" 2>&1 | tee "${log_path}" || status=$?

        if (( status != 0 )); then
          echo "ERROR: evaluation failed for ${dataset_name} (exit ${status}); see ${log_path}"
          FAILED+=("${dataset_name}.pkl")
        else
          type_done=$(( type_done + 1 ))
          n_done=$(( n_done + 1 ))
        fi
      done
    done
  done

  echo
  echo "--- ${train_type}: ${type_done}/${n_per_type} evaluated into ${results_csv}"
  SUMMARY+=("${train_type}: ${type_done}/${n_per_type} -> ${results_csv}")
done

echo
echo "Done. ${n_done}/${n_total} combination(s) evaluated across ${n_types} train type(s)."
printf '  %s\n' "${SUMMARY[@]}"

if (( ${#MISSING[@]} > 0 )); then
  echo "Missing datasets (${#MISSING[@]}), create them with scripts/dataset_creation/coloc_datasets.sh:"
  printf '  %s\n' "${MISSING[@]}"
fi

if (( ${#FAILED[@]} > 0 )); then
  echo "Failed evaluations (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi

if (( n_done == 0 )); then
  echo "No datasets were evaluated."
  exit 1
fi
