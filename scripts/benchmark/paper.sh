#!/usr/bin/env bash
set -euo pipefail

# Turns the inference-time benchmark into the paper figure: per-pair computation
# time vs recording length, one colour per sampling rate, one line style per
# approach.
#
# Run scripts/benchmark/evaluate.sh first; this reads its CSV.

# Paths
RESULTS_CSV="${BENCH_RESULTS_CSV:-results/benchmark/inference_times.csv}"
FIGURE_DIR="${PAPER_FIGURE_DIR:-results/paper/figures}"
VARS_DIR="${PAPER_VARS_DIR:-results/paper/vars}"
VARS_PATH="${VARS_DIR}/runtime.tex"
DISTANCE_RESULTS_CSV="${BENCH_DISTANCE_RESULTS_CSV:-results/coloc_distance/dtw_r1.csv}"

PAPER_FORMAT="${PAPER_FORMAT:-pdf}"
FIGURE_PATH="${FIGURE_DIR}/runtime_comparison.${PAPER_FORMAT}"

# Plot options
INCLUDE_VOTE="${BENCH_INCLUDE_VOTE:-1}"
STDEV="${BENCH_STDEV:-0}"
LOG_Y="${BENCH_LOG_Y:-0}"
LOG_X="${BENCH_LOG_X:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "Benchmark CSV not found: ${RESULTS_CSV}"
  echo "Run scripts/benchmark/evaluate.sh first."
  exit 1
fi

mkdir -p "${FIGURE_DIR}" "${VARS_DIR}"

PLOT_ARGS=()
[[ "${INCLUDE_VOTE}" == "0" ]] && PLOT_ARGS+=(--no-include-vote)
[[ "${STDEV}" != "0" ]] && PLOT_ARGS+=(--stdev)
[[ "${LOG_Y}" != "0" ]] && PLOT_ARGS+=(--log-y)
[[ "${LOG_X}" != "0" ]] && PLOT_ARGS+=(--log-x)

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Benchmark CSV: ${RESULTS_CSV}"
echo "Figure:        ${FIGURE_PATH}"
echo "LaTeX vars:    ${VARS_PATH}"
echo "Format:        ${PAPER_FORMAT}"
echo "Majority vote counted in the ML time: $( [[ "${INCLUDE_VOTE}" == "0" ]] && echo no || echo yes )"

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  plot-results runtime-comparison "${RESULTS_CSV}" "${FIGURE_PATH}" \
  "${PLOT_ARGS[@]}"

VARS_ARGS=()
[[ -n "${DISTANCE_RESULTS_CSV}" && -f "${DISTANCE_RESULTS_CSV}" ]] && \
  VARS_ARGS+=(--results-csv "${DISTANCE_RESULTS_CSV}")
[[ "${INCLUDE_VOTE}" == "0" ]] && VARS_ARGS+=(--no-include-vote)

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  generate-latex-vars runtime "${RESULTS_CSV}" "${VARS_PATH}" \
  "${VARS_ARGS[@]}"

echo
echo "Done."
echo "  ${FIGURE_PATH}"
echo "  ${VARS_PATH}"
