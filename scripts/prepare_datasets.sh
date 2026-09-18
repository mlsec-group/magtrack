#!/usr/bin/env bash
set -euo pipefail

# Builds every dataset the experiments need.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SIF="${PYTHON_SIF:-${REPO_ROOT}/python.sif}"

STEPS=(
  "dataset_creation/tmd_datasets.sh"
  "dataset_creation/coloc_datasets.sh"
  "dataset_creation/coloc_baseline_datasets.sh"
  "dataset_creation/paper.sh"
)

fmt_duration() {
  local s=$1
  if (( s >= 3600 )); then
    printf '%dh %02dm %02ds' $(( s / 3600 )) $(( s % 3600 / 60 )) $(( s % 60 ))
  elif (( s >= 60 )); then
    printf '%dm %02ds' $(( s / 60 )) $(( s % 60 ))
  else
    printf '%ds' "${s}"
  fi
}

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  echo "Build it first with:  scripts/build.sh"
  exit 1
fi

for step in "${STEPS[@]}"; do
  if [[ ! -f "${SCRIPT_DIR}/${step}" ]]; then
    echo "Step script not found: ${SCRIPT_DIR}/${step}"
    exit 1
  fi
done

cd "${REPO_ROOT}"

total=${#STEPS[@]}
n=0
step=""
started=$(date +%s)

trap 'echo; echo "FAILED at [${n}/${total}] scripts/${step}"; echo "Rerun scripts/prepare_datasets.sh once the cause is fixed."' ERR

echo "=============================================================================="
echo " Preparing datasets (${total} steps)"
echo " Repository: ${REPO_ROOT}"
echo "=============================================================================="

for step in "${STEPS[@]}"; do
  n=$(( n + 1 ))
  echo
  echo "------------------------------------------------------------------------------"
  echo " [${n}/${total}] scripts/${step}"
  echo "------------------------------------------------------------------------------"
  step_started=$(date +%s)

  bash "${SCRIPT_DIR}/${step}"

  echo
  echo " [${n}/${total}] scripts/${step} finished in $(fmt_duration $(( $(date +%s) - step_started )))"
done

trap - ERR

echo
echo "=============================================================================="
echo " All ${total} steps done in $(fmt_duration $(( $(date +%s) - started )))"
echo "=============================================================================="
echo "   datasets/tmd_datasets      transport mode detection datasets"
echo "   datasets/coloc_datasets    colocation datasets (filtered trace)"
echo "   datasets/coloc_baseline    colocation baseline (absolute field)"
echo "   results/paper/vars         dataset LaTeX variables"
