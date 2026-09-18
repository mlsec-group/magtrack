#!/usr/bin/env bash
set -euo pipefail

# Turns the raw datasets into the paper's dataset artifacts: the traintrack
# corpus statistics and the activity times of the non-train recordings, both as
# LaTeX \newcommand definitions.
#
# This reads the recordings themselves, so it needs the full Zenodo download,
# not just the generated datasets.

# Paths
TRAINTRACK_PATH="${TRAINTRACK_PATH:-traintrack-dataset/traintrack.yml}"
TRAINTRACK_STATS="${TRAINTRACK_STATS:-traintrack-dataset/traintrack_stats.yml}"
NO_TRAINRIDE_PATH="${NO_TRAINRIDE_PATH:-no-trainride-dataset/no_trainride_data.yml}"
COLOC_DATASETS_DIR="${COLOC_DATASETS_DIR:-datasets/coloc_datasets}"
VARS_DIR="${PAPER_VARS_DIR:-results/paper/vars}"
STATS_VARS_PATH="${VARS_DIR}/traintrack_dataset.tex"
ACTIVITY_VARS_PATH="${VARS_DIR}/tmd_activity.tex"
COLOC_VARS_PATH="${VARS_DIR}/coloc_datasets.tex"

REBUILD_STATS="${PAPER_REBUILD_STATS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ "${REBUILD_STATS}" != "0" && ! -f "${TRAINTRACK_PATH}" ]]; then
  echo "Traintrack dataset not found: ${TRAINTRACK_PATH}"
  echo "Download the dataset first, or set PAPER_REBUILD_STATS=0 to reuse ${TRAINTRACK_STATS}."
  exit 1
fi

if [[ "${REBUILD_STATS}" == "0" && ! -f "${TRAINTRACK_STATS}" ]]; then
  echo "Statistics file not found: ${TRAINTRACK_STATS}"
  echo "Unset PAPER_REBUILD_STATS to build it from ${TRAINTRACK_PATH}."
  exit 1
fi

if [[ ! -f "${NO_TRAINRIDE_PATH}" ]]; then
  echo "No-trainride dataset not found: ${NO_TRAINRIDE_PATH}"
  exit 1
fi

if [[ ! -d "${COLOC_DATASETS_DIR}" ]]; then
  echo "Colocation datasets not found: ${COLOC_DATASETS_DIR}"
  echo "Build them first with scripts/dataset_creation/coloc_datasets.sh"
  exit 1
fi

mkdir -p "${VARS_DIR}"

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Traintrack:   ${TRAINTRACK_PATH}"
echo "No-trainride: ${NO_TRAINRIDE_PATH}"
echo "Coloc data:   ${COLOC_DATASETS_DIR}"
echo "Statistics:   ${TRAINTRACK_STATS}$( [[ "${REBUILD_STATS}" == "0" ]] && echo "  (reused)" || echo "  (rebuilt)" )"
echo "LaTeX vars:   ${STATS_VARS_PATH}"
echo "              ${ACTIVITY_VARS_PATH}"
echo "              ${COLOC_VARS_PATH}"

run_in_container() {
  apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" "$@"
}

# 1/4: corpus statistics from the traintrack recordings.
if [[ "${REBUILD_STATS}" != "0" ]]; then
  echo
  echo "=== dataset statistics ==="
  run_in_container traintrack-dataset get-dataset-vars "${TRAINTRACK_PATH}" \
    --output-file "${TRAINTRACK_STATS}"
fi

# 2/4: those statistics as LaTeX definitions.
echo
echo "=== dataset variables ==="
run_in_container traintrack-dataset get-stats-vars "${TRAINTRACK_STATS}" \
  --output-path "${STATS_VARS_PATH}"

# 3/4: activity times of the non-train recordings.
echo
echo "=== activity variables ==="
run_in_container traintrack-dataset get-activity-vars "${NO_TRAINRIDE_PATH}" \
  --output-path "${ACTIVITY_VARS_PATH}"

# 4/4: how many segments the colocation datasets hold.  Every .pkl is read, so
# this is the slowest step and needs room for the largest dataset in memory.
echo
echo "=== colocation dataset variables ==="
run_in_container traintrack-dataset get-coloc-dataset-vars "${COLOC_DATASETS_DIR}" \
  --output-path "${COLOC_VARS_PATH}"

echo
echo "Done."
echo "  ${TRAINTRACK_STATS}"
echo "  ${STATS_VARS_PATH}"
echo "  ${ACTIVITY_VARS_PATH}"
echo "  ${COLOC_VARS_PATH}"
