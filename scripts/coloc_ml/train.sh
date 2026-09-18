#!/usr/bin/env bash
set -euo pipefail

# Step 2 of the coloc ML pipeline: train the model with the best configuration
# found by the hyperparameter search.
#
# Usage:
#   bash scripts/coloc_ml/train.sh
#
# Requires:
#   scripts/coloc_ml/hyperparameter_search.sh  (produces model_hparams.yaml)
# Next:
#   scripts/coloc_ml/evaluate.sh               (evaluates the trained model)
#
# Environment variables (with defaults):
#   COLOC_RESULTS_DIR    - Directory for coloc ML results (default: ./results)
#   COLOC_DATA_DIR       - Directory containing the coloc .pkl datasets (all trains)
#   COLOC_TRAIN_EPOCHS   - Max epochs for final training (default: 1000)
#   COLOC_TEST_FRACTION  - Fraction of data for testing (default: 0.3)
#   COLOC_SEED           - Random seed (default: "magtrack")
#   COLOC_NUM_GPUS       - GPUs to use (default: auto-detected, 0 without CUDA)
#   COLOC_DATASET_PATH   - Path to ml dataset pkl (or 'from_config')
#   PYTHON_SIF           - Path to Apptainer image (default: ./python.sif)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="${PYTHON_SIF:-./python.sif}"

COLOC_RESULTS_DIR="${COLOC_RESULTS_DIR:-${REPO_ROOT}/results}"
COLOC_DATA_DIR="${COLOC_DATA_DIR:-./datasets/coloc_datasets/normalized_trace_all_trains/}"

HPARAMS_OUTPUT="${COLOC_RESULTS_DIR}/model_hparams.yaml"

CHECKPOINT_DIR="${COLOC_RESULTS_DIR}/coloc_ml/checkpoints"
MODEL_OUTPUT="${REPO_ROOT}/coloc_model/colocation_net.pth"
HPARAMS_PATH="${REPO_ROOT}/coloc_model/model_hparams.yaml"

DATA_DIR="${COLOC_DATA_DIR}"

# ── Training settings ─────────────────────────────────────────────────────────
TRAIN_EPOCHS="${COLOC_TRAIN_EPOCHS:-1000}"
TEST_FRACTION="${COLOC_TEST_FRACTION:-0.3}"
SEED="${COLOC_SEED:-"magtrack"}"
DATASET_PATH="${COLOC_DATASET_PATH:-from_config}"

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

if [[ ! -f "${HPARAMS_OUTPUT}" ]]; then
  echo "ERROR: Hyperparameters not found: ${HPARAMS_OUTPUT}"
  echo "Run scripts/coloc_ml/hyperparameter_search.sh first."
  exit 1
fi

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "$(dirname "${MODEL_OUTPUT}")"

echo "=============================================================================="
echo " Coloc ML Pipeline - train.sh"
echo "=============================================================================="
echo " Python image:     ${PYTHON_SIF}"
echo " Data directory:   ${DATA_DIR}"
echo " Hyperparams in:   ${HPARAMS_OUTPUT}"
echo " Model output:     ${MODEL_OUTPUT}"
echo " Checkpoints:      ${CHECKPOINT_DIR}"
echo " GPUs detected:    ${NUM_GPUS}$( ((NUM_GPUS==0)) && echo "  (CPU fallback)" )"
echo "=============================================================================="

# ── Train ML Model with Best Configuration ─────────────────────────
echo ""
echo "=============================================================================="
echo " Train ML Model (best configuration)"
echo "=============================================================================="
echo " Epochs:      ${TRAIN_EPOCHS}"
echo " Batch size, learning rate, weight decay and the dataset come from"
echo " ${HPARAMS_PATH} (the hyperparameter search result)."
echo ""

# Copy hparams to coloc_model for the training script to find
cp "${HPARAMS_OUTPUT}" "${HPARAMS_PATH}"

apptainer exec "${GPU_ARGS[@]}" --bind "${REPO_ROOT}:${REPO_ROOT}" --bind "${DATA_DIR}:/data/" "${PYTHON_SIF}" \
  train-ml-model \
  --dataset-path "${DATASET_PATH}" \
  --test-fraction "${TEST_FRACTION}" \
  --seed "${SEED}" \
  --epochs "${TRAIN_EPOCHS}" \
  --hparams-path "${HPARAMS_PATH}" \
  --save

echo ""
echo "Model training complete."
echo "  Best model saved to: ${MODEL_OUTPUT}"

echo ""
echo "=============================================================================="
echo " Training - COMPLETE"
echo "=============================================================================="
echo " Outputs:"
echo "   Best model:         ${MODEL_OUTPUT}"
echo "   Hyperparameters:    ${HPARAMS_PATH}"
echo "   Checkpoints:        ${CHECKPOINT_DIR}"
echo ""
echo " Next: bash scripts/coloc_ml/evaluate.sh"
echo "=============================================================================="
