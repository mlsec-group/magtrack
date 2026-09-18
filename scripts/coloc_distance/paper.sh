#!/usr/bin/env bash
set -euo pipefail

# Turns the DTW colocation sweep into the paper artifacts: the score-vs-duration
# figure and the LaTeX \newcommand definitions.
#
# Run scripts/coloc_distance/evaluate.sh first; this reads its results CSV.

# Paths
RESULTS_CSV="${COLOC_RESULTS_CSV:-results/coloc_distance/dtw_r1.csv}"
FIGURE_DIR="${PAPER_FIGURE_DIR:-results/paper/figures}"
VARS_DIR="${PAPER_VARS_DIR:-results/paper/vars}"

PAPER_FORMAT="${PAPER_FORMAT:-pdf}"
FIGURE_PATH="${FIGURE_DIR}/coloc_dtw1.${PAPER_FORMAT}"

DISTANCE_NAME="${PAPER_DISTANCE_NAME:-dtw_r1}"

# Plot options
SCORE="${PAPER_SCORE:-mcc}"
YLIM_MIN="${PAPER_YLIM_MIN:-0}"
YLIM_MAX="${PAPER_YLIM_MAX:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "Results CSV not found: ${RESULTS_CSV}"
  echo "Run scripts/coloc_distance/evaluate.sh first, or point COLOC_RESULTS_CSV"
  echo "at an existing sweep (e.g. results/coloc_dtw/dtw_r1.csv)."
  exit 1
fi

mkdir -p "${FIGURE_DIR}" "${VARS_DIR}"

CSV_DIR="$(dirname "${RESULTS_CSV}")"
CSV_BASE="$(basename "${RESULTS_CSV}")"

# evaluate.sh writes one CSV per train type: "all" keeps the plain CSV name, the
# others carry the train type as a prefix.
TRAIN_TYPES=(${PAPER_TRAIN_TYPES:-all long_distance regional})

STAGE_DIR=""
cleanup() { [[ -n "${STAGE_DIR}" ]] && rm -rf "${STAGE_DIR}"; }
trap cleanup EXIT

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Results CSV:  ${RESULTS_CSV}"
echo "Figure:       ${FIGURE_PATH}"
echo "Train types:  ${TRAIN_TYPES[*]}"
echo "Score:        ${SCORE} (y-axis ${YLIM_MIN}..${YLIM_MAX})"
echo "Format:       ${PAPER_FORMAT}"

# The figure stays the all-trains sweep.
apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  plot-results coloc-distance "${RESULTS_CSV}" "${FIGURE_PATH}" \
  --score "${SCORE}" \
  --ylim "${YLIM_MIN}" "${YLIM_MAX}"

WRITTEN=()
MISSING=()

for train_type in "${TRAIN_TYPES[@]}"; do
  if [[ "${train_type}" == "all" ]]; then
    csv="${RESULTS_CSV}"
    macro_name="${DISTANCE_NAME}"
    vars_path="${VARS_DIR}/coloc_distances_${DISTANCE_NAME}.tex"
  else
    csv="${CSV_DIR}/${train_type}_${CSV_BASE}"
    macro_name="${train_type}_${DISTANCE_NAME}"
    vars_path="${VARS_DIR}/coloc_${train_type}_${DISTANCE_NAME}.tex"
  fi

  if [[ ! -f "${csv}" ]]; then
    echo
    echo "WARNING: ${csv} not found, skipping ${train_type}."
    MISSING+=("${train_type}")
    continue
  fi

  # Macro names are derived from the CSV stem, so each train type needs its own
  # stem or the three files would define identically named macros. Stage a
  # renamed copy whenever the CSV is not already called what the macros need.
  vars_input="${csv}"
  if [[ "$(basename "${csv}" .csv)" != "${macro_name}" ]]; then
    [[ -n "${STAGE_DIR}" ]] || STAGE_DIR="$(mktemp -d "${CSV_DIR}/.paper_stage_XXXXXX")"
    vars_input="${STAGE_DIR}/${macro_name}.csv"
    cp "${csv}" "${vars_input}"
  fi

  echo
  echo "=== ${train_type}: ${csv}"
  echo "    -> ${vars_path}  (macros from '${macro_name}')"
  apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
    generate-latex-vars coloc-distances "${vars_input}" "${vars_path}"
  WRITTEN+=("${vars_path}")
done

echo
echo "Done."
echo "  ${FIGURE_PATH}"
for path in "${WRITTEN[@]}"; do
  echo "  ${path}"
done
if (( ${#MISSING[@]} > 0 )); then
  echo
  echo "Skipped (no CSV): ${MISSING[*]}"
  echo "Run scripts/coloc_distance/evaluate.sh to produce them."
fi
