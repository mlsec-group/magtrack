#!/usr/bin/env bash
set -euo pipefail

# Turns the transport mode detection parameter search into the paper artifacts:
# the score-vs-recording-time figure, the LaTeX \newcommand definitions and the
# summary table.
#
# Run scripts/tmd/run_parameter_search.sh first; this reads its results.

# Paths
RESULTS_DIR="${TMD_RESULTS_DIR:-results/tmd}"
FIGURE_DIR="${PAPER_FIGURE_DIR:-results/paper/figures}"
VARS_DIR="${PAPER_VARS_DIR:-results/paper/vars}"
TABLES_DIR="${PAPER_TABLES_DIR:-results/paper/tables}"

PAPER_FORMAT="${PAPER_FORMAT:-pdf}"
FIGURE_PATH="${FIGURE_DIR}/tmd_results_recording_time.${PAPER_FORMAT}"
VARS_PATH="${VARS_DIR}/tmd_results.tex"
TABLE_PATH="${TABLES_DIR}/tmd_summary_table.tex"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "Results directory not found: ${RESULTS_DIR}"
  echo "Run scripts/tmd/run_parameter_search.sh first, or point TMD_RESULTS_DIR at it."
  exit 1
fi

if ! compgen -G "${RESULTS_DIR}/tmd_s*_d*.optuna_search_*.csv" > /dev/null; then
  echo "No 'tmd_s*_d*.optuna_search_*.csv' files in ${RESULTS_DIR}; nothing to plot."
  exit 1
fi

mkdir -p "${FIGURE_DIR}" "${VARS_DIR}" "${TABLES_DIR}"

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Figure:       ${FIGURE_PATH}"
echo "LaTeX vars:   ${VARS_PATH}"
echo "Table:        ${TABLE_PATH}"
echo "Format:       ${PAPER_FORMAT}"

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  plot-results tmd-results "${RESULTS_DIR}" "${FIGURE_PATH}"

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  generate-latex-vars tmd-results "${RESULTS_DIR}" "${VARS_PATH}"

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  generate-tables tmd-results "${RESULTS_DIR}" "${TABLE_PATH}"

echo
echo "Done."
echo "  ${FIGURE_PATH}"
echo "  ${VARS_PATH}"
echo "  ${TABLE_PATH}"
