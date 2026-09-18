#!/usr/bin/env bash
set -euo pipefail

# Turns the absolute-field DDTW baseline sweep into the paper artifacts: the
# score-vs-duration figure and the LaTeX \newcommand definitions.
#
# Run scripts/coloc_baseline/evaluate.sh first; this reads its results CSV.

# Paths
RESULTS_CSV="${COLOC_RESULTS_CSV:-results/coloc_baseline/ddtw.csv}"
FIGURE_DIR="${PAPER_FIGURE_DIR:-results/paper/figures}"
VARS_DIR="${PAPER_VARS_DIR:-results/paper/vars}"

PAPER_FORMAT="${PAPER_FORMAT:-pdf}"
FIGURE_PATH="${FIGURE_DIR}/coloc_abs_ddtw.${PAPER_FORMAT}"
VARS_PATH="${VARS_DIR}/coloc_distances_abs_ddtw.tex"

DISTANCE_NAME="${PAPER_DISTANCE_NAME:-abs_ddtw}"

# Plot options.
SCORE="${PAPER_SCORE:-mcc}"
YLIM_MIN="${PAPER_YLIM_MIN:-0}"
YLIM_MAX="${PAPER_YLIM_MAX:-0.5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "Results CSV not found: ${RESULTS_CSV}"
  echo "Run scripts/coloc_baseline/evaluate.sh first, or point COLOC_RESULTS_CSV"
  echo "at an existing sweep (e.g. results/coloc_baseline/abs_ddtw.csv)."
  exit 1
fi

mkdir -p "${FIGURE_DIR}" "${VARS_DIR}"

VARS_INPUT="${RESULTS_CSV}"
STAGE_DIR=""
cleanup() { [[ -n "${STAGE_DIR}" ]] && rm -rf "${STAGE_DIR}"; }
trap cleanup EXIT
if [[ "$(basename "${RESULTS_CSV}" .csv)" != "${DISTANCE_NAME}" ]]; then
  STAGE_DIR="$(mktemp -d "$(dirname "${RESULTS_CSV}")/.paper_stage_XXXXXX")"
  VARS_INPUT="${STAGE_DIR}/${DISTANCE_NAME}.csv"
  cp "${RESULTS_CSV}" "${VARS_INPUT}"
fi

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Results CSV:  ${RESULTS_CSV}"
echo "Figure:       ${FIGURE_PATH}"
echo "LaTeX vars:   ${VARS_PATH}  (macro names derived from '${DISTANCE_NAME}')"
echo "Score:        ${SCORE} (y-axis ${YLIM_MIN}..${YLIM_MAX})"
echo "Format:       ${PAPER_FORMAT}"

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  plot-results coloc-distance "${RESULTS_CSV}" "${FIGURE_PATH}" \
  --score "${SCORE}" \
  --ylim "${YLIM_MIN}" "${YLIM_MAX}"

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  generate-latex-vars coloc-distances "${VARS_INPUT}" "${VARS_PATH}"

echo
echo "Done."
echo "  ${FIGURE_PATH}"
echo "  ${VARS_PATH}"
