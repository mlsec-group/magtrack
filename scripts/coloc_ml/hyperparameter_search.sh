#!/usr/bin/env bash
set -euo pipefail

# Step 1 of the coloc ML pipeline: Optuna hyperparameter search over the model
# and training configuration.
#
# Usage:
#   bash scripts/coloc_ml/hyperparameter_search.sh
#
# Next:
#   scripts/coloc_ml/train.sh   trains the best configuration
#
# Environment variables (with defaults):
#   COLOC_RESULTS_DIR    - Directory for coloc ML results (default: ./results)
#   COLOC_DATA_DIR       - Directory containing the coloc .pkl datasets (all trains)
#   COLOC_N_TRIALS       - Number of Optuna trials (overrides COLOC_HPARAMS_SEARCH, default: 1000)
#   COLOC_HPARAMS_SEARCH - Number of Optuna trials (default: 1000)
#   COLOC_MAX_EPOCHS     - Max epochs per trial during search (default: 200)
#   COLOC_TIMEOUT        - Timeout in seconds for the search (0 = no timeout)
#   COLOC_N_JOBS         - Number of concurrent Optuna trials (default: 5)
#   COLOC_NUM_GPUS       - Number of GPUs for parallel trials (default: auto-detected, 0 without CUDA)
#   COLOC_TEST_FRACTION  - Fraction of data for testing (default: 0.3)
#   COLOC_SEED           - Random seed (default: "magtrack")
#   PYTHON_SIF           - Path to Apptainer image (default: ./python.sif)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="${PYTHON_SIF:-./python.sif}"

COLOC_RESULTS_DIR="${COLOC_RESULTS_DIR:-${REPO_ROOT}/results}"
COLOC_DATA_DIR="${COLOC_DATA_DIR:-./datasets/coloc_datasets/normalized_trace_all_trains/}"

# Output paths
HPARAMS_OUTPUT="${COLOC_RESULTS_DIR}/model_hparams.yaml"
DB_OUTPUT="${COLOC_RESULTS_DIR}/coloc_ml/hyperparam_search.db"

DATA_DIR="${COLOC_DATA_DIR}"

# ── Hyperparameter search settings ────────────────────────────────────────────
N_TRIALS="${COLOC_N_TRIALS:-${COLOC_HPARAMS_SEARCH:-1000}}"
MAX_EPOCHS="${COLOC_MAX_EPOCHS:-200}"
TIMEOUT="${COLOC_TIMEOUT:-0}"
N_JOBS="${COLOC_N_JOBS:-5}"
TEST_FRACTION="${COLOC_TEST_FRACTION:-0.3}"
SEED="${COLOC_SEED:-"magtrack"}"

# ── GPU detection ─────────────────────────────────────────────────────────────
NUM_GPUS="${COLOC_NUM_GPUS:-}"
if [[ -z "${NUM_GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
  else
    NUM_GPUS=0
  fi
fi

GPU_ARGS=()
(( NUM_GPUS > 0 )) && GPU_ARGS+=(--nv)

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "ERROR: Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: Dataset directory not found: ${DATA_DIR}"
  echo "Set COLOC_DATA_DIR to point to your dataset directory."
  exit 1
fi

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "$(dirname "${HPARAMS_OUTPUT}")"
mkdir -p "$(dirname "${DB_OUTPUT}")"

echo "=============================================================================="
echo " Coloc ML Pipeline - hyperparameter_search.sh"
echo "=============================================================================="
echo " Python image:     ${PYTHON_SIF}"
echo " Data directory:   ${DATA_DIR}"
echo " Hyperparams out:  ${HPARAMS_OUTPUT}"
echo " DB output:        ${DB_OUTPUT}"
echo " GPUs detected:    ${NUM_GPUS}$( ((NUM_GPUS==0)) && echo "  (CPU fallback)" )"
echo "=============================================================================="

# ── Hyperparameter Search ──────────────────────────────────────────
echo ""
echo "=============================================================================="
echo " Hyperparameter Search"
echo "=============================================================================="
echo " N trials:    ${N_TRIALS}"
echo " Max epochs:  ${MAX_EPOCHS}"
echo " Timeout:     ${TIMEOUT}s"
echo " N jobs:      ${N_JOBS}"
echo " Num GPUs:    ${NUM_GPUS}"
echo ""

apptainer exec "${GPU_ARGS[@]}" --bind "${DATA_DIR}:/data/" "${PYTHON_SIF}" \
  hyperparameter-search \
  --base-data-dir /data/ \
  --test-fraction "${TEST_FRACTION}" \
  --seed "${SEED}" \
  --n-trials "${N_TRIALS}" \
  --max-epochs "${MAX_EPOCHS}" \
  --timeout "${TIMEOUT}" \
  --n-jobs "${N_JOBS}" \
  --num-gpus "${NUM_GPUS}" \
  --hparams-output "${HPARAMS_OUTPUT}" \
  --db-output "${DB_OUTPUT}"

echo ""
echo "Hyperparameter search complete."
echo "  Best hparams saved to: ${HPARAMS_OUTPUT}"
echo "  Optuna DB saved to:    ${DB_OUTPUT}"

echo ""
echo "=============================================================================="
echo " Hyperparameter search - COMPLETE"
echo "=============================================================================="
echo " Outputs:"
echo "   Hyperparameters:    ${HPARAMS_OUTPUT}"
echo "   Optuna DB:          ${DB_OUTPUT}"
echo ""
echo " Next: bash scripts/coloc_ml/train.sh"
echo "=============================================================================="
