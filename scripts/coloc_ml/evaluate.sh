#!/usr/bin/env bash
set -euo pipefail

# Steps 3-8 of the coloc ML pipeline: generate the evaluation jobs, run them for
# all/long_distance/regional, run the majority vote and compute its metrics.
#
# Usage:
#   bash scripts/coloc_ml/evaluate.sh
#
# Requires:
#   scripts/coloc_ml/hyperparameter_search.sh  (produces model_hparams.yaml)
#   scripts/coloc_ml/train.sh                  (produces colocation_net.pth)
#
# Environment variables (with defaults):
#   COLOC_RESULTS_DIR             - Directory for coloc ML evaluation results (default: ./results)
#   COLOC_DATA_DIR                - Directory containing coloc .pkl datasets (all trains)
#   COLOC_DATA_DIR_LONG_DISTANCE  - Directory containing long_distance coloc .pkl datasets
#   COLOC_DATA_DIR_REGIONAL       - Directory containing regional coloc .pkl datasets
#   COLOC_TEST_FRACTION           - Fraction of data for testing (default: 0.3)
#   COLOC_SEED                    - Random seed (default: "magtrack")
#   COLOC_NUM_GPUS                - GPUs to use (default: auto-detected, 0 without CUDA)
#   COLOC_PARALLEL_JOBS           - Concurrent evaluation jobs
#                                   (default: 4 with a GPU, nproc-1 on CPU)
#   PYTHON_SIF                    - Path to Apptainer image (default: ./python.sif)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="${PYTHON_SIF:-./python.sif}"

COLOC_RESULTS_DIR="${COLOC_RESULTS_DIR:-${REPO_ROOT}/results}"
COLOC_DATA_DIR="${COLOC_DATA_DIR:-./datasets/coloc_datasets/normalized_trace_all_trains/}"
COLOC_DATA_DIR_LONG_DISTANCE="${COLOC_DATA_DIR_LONG_DISTANCE:-./datasets/coloc_datasets/normalized_trace_long_distance_trains/}"
COLOC_DATA_DIR_REGIONAL="${COLOC_DATA_DIR_REGIONAL:-./datasets/coloc_datasets/normalized_trace_regional_trains/}"

# Inputs from the training step
MODEL_OUTPUT="${REPO_ROOT}/coloc_model/colocation_net.pth"
HPARAMS_PATH="${REPO_ROOT}/coloc_model/model_hparams.yaml"

# Dataset & evaluation paths
DATA_DIR="${COLOC_DATA_DIR}"
DATA_DIR_LONG_DISTANCE="${COLOC_DATA_DIR_LONG_DISTANCE}"
DATA_DIR_REGIONAL="${COLOC_DATA_DIR_REGIONAL}"
RESULTS_DIR_COLOC="${COLOC_RESULTS_DIR}/coloc_ml"
RESULTS_DIR_COLOC_LONG_DISTANCE="${COLOC_RESULTS_DIR}/coloc_ml/long_distance"
RESULTS_DIR_COLOC_REGIONAL="${COLOC_RESULTS_DIR}/coloc_ml/regional"
EVAL_SCRIPT="${REPO_ROOT}/scripts/coloc_ml/tmp/eval_ml.sh"
EVAL_SCRIPT_LONG_DISTANCE="${REPO_ROOT}/scripts/coloc_ml/tmp/eval_ml_long_distance.sh"
EVAL_SCRIPT_REGIONAL="${REPO_ROOT}/scripts/coloc_ml/tmp/eval_ml_regional.sh"
MAJORITY_SCRIPT="${REPO_ROOT}/scripts/coloc_ml/tmp/eval_majority_ml.sh"

TEST_FRACTION="${COLOC_TEST_FRACTION:-0.3}"
SEED="${COLOC_SEED:-"magtrack"}"

# ── GPU detection ─────────────────────────────────────────────────────────────
# GPU work needs the NVIDIA stack passed into the container via --nv, and the
# hyperparameter search hard-errors when num_gpus > 0 without CUDA.  Detect the
# devices and fall back to CPU when there are none.  Set COLOC_NUM_GPUS to
# override.
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

if (( NUM_GPUS > 0 )); then
  PARALLEL_JOBS="${COLOC_PARALLEL_JOBS:-16}"
else
  NPROC="$(nproc)"
  PARALLEL_JOBS="${COLOC_PARALLEL_JOBS:-$(( NPROC > 1 ? NPROC - 1 : 1 ))}"
fi

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

if [[ ! -f "${MODEL_OUTPUT}" ]]; then
  echo "ERROR: Trained model not found: ${MODEL_OUTPUT}"
  echo "Run scripts/coloc_ml/train.sh first."
  exit 1
fi

if [[ ! -f "${HPARAMS_PATH}" ]]; then
  echo "ERROR: Hyperparameters not found: ${HPARAMS_PATH}"
  echo "Run scripts/coloc_ml/train.sh first."
  exit 1
fi

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR_COLOC}"
mkdir -p "${RESULTS_DIR_COLOC_LONG_DISTANCE}"
mkdir -p "${RESULTS_DIR_COLOC_REGIONAL}"

echo "=============================================================================="
echo " Coloc ML Pipeline - evaluate.sh"
echo "=============================================================================="
echo " Python image:     ${PYTHON_SIF}"
echo " Data directory:   ${DATA_DIR}"
echo " Model:            ${MODEL_OUTPUT}"
echo " Hyperparameters:  ${HPARAMS_PATH}"
echo " Results dir:      ${RESULTS_DIR_COLOC}"
echo " GPUs detected:    ${NUM_GPUS}$( ((NUM_GPUS==0)) && echo "  (CPU fallback)" )"
echo " Parallel jobs:    ${PARALLEL_JOBS}"
echo "=============================================================================="

# ── Step 1/6: Generate Evaluation Jobs ───────────────────────────────────────
echo ""
echo "=============================================================================="
echo " Step 1/6: Generate Evaluation Jobs"
echo "=============================================================================="

# Generate eval_ml.sh (all trains)
echo "Generating eval_ml.sh..."
apptainer exec "${GPU_ARGS[@]}" --bind "${REPO_ROOT}:${REPO_ROOT}" --bind "${DATA_DIR}:/data/" "${PYTHON_SIF}" \
  generate-evaluate-ml-jobs \
  "/data/" \
  "${MODEL_OUTPUT}" \
  --duration 5 \
  --duration 10 \
  --duration 20 \
  --duration 30 \
  --duration 60 \
  --sampling-rate 10 \
  --sampling-rate 20 \
  --sampling-rate 40 \
  --sampling-rate 60 \
  --rolling-window 1 \
  --rolling-window 5 \
  --rolling-window 10 \
  --rolling-window 50 \
  --rolling-window 100 \
  --rolling-window 150 \
  --trainride-start-seconds 0 \
  --trainride-start-seconds 60 \
  --trainride-start-seconds 300 \
  --trainride-start-seconds 600 \
  --trainride-start-seconds 900 \
  --hparams-path "${HPARAMS_PATH}" \
  --test-fraction "${TEST_FRACTION}" \
  --threshold 0.0 \
  --batch-size 1024 \
  --seed "${SEED}" \
  --results-dir "${RESULTS_DIR_COLOC}" \
  --python-sif "${PYTHON_SIF}" \
  --bind-path "${DATA_DIR}:/data/" \
  --parallel-jobs "${PARALLEL_JOBS}" \
  --parallel-output-dir "${RESULTS_DIR_COLOC}/eval_ml_logs" \
  -o "${EVAL_SCRIPT}"

echo "  → ${EVAL_SCRIPT}"

# Generate eval_ml_long_distance.sh
echo "Generating eval_ml_long_distance.sh..."
apptainer exec "${GPU_ARGS[@]}" --bind "${REPO_ROOT}:${REPO_ROOT}" --bind "${DATA_DIR_LONG_DISTANCE}:/data/" "${PYTHON_SIF}" \
  generate-evaluate-ml-jobs \
  "/data/" \
  "${MODEL_OUTPUT}" \
  --duration 5 \
  --duration 10 \
  --duration 20 \
  --duration 30 \
  --duration 60 \
  --sampling-rate 10 \
  --sampling-rate 20 \
  --sampling-rate 40 \
  --sampling-rate 60 \
  --rolling-window 1 \
  --rolling-window 5 \
  --rolling-window 10 \
  --rolling-window 50 \
  --rolling-window 100 \
  --rolling-window 150 \
  --trainride-start-seconds 0 \
  --trainride-start-seconds 60 \
  --trainride-start-seconds 300 \
  --trainride-start-seconds 600 \
  --trainride-start-seconds 900 \
  --hparams-path "${HPARAMS_PATH}" \
  --test-fraction "${TEST_FRACTION}" \
  --threshold 0.0 \
  --batch-size 1024 \
  --seed "${SEED}" \
  --results-dir "${RESULTS_DIR_COLOC_LONG_DISTANCE}" \
  --python-sif "${PYTHON_SIF}" \
  --bind-path "${DATA_DIR_LONG_DISTANCE}:/data/" \
  --parallel-jobs "${PARALLEL_JOBS}" \
  --parallel-output-dir "${RESULTS_DIR_COLOC_LONG_DISTANCE}/eval_ml_logs" \
  -o "${EVAL_SCRIPT_LONG_DISTANCE}"

echo "  → ${EVAL_SCRIPT_LONG_DISTANCE}"

# Generate eval_ml_regional.sh
echo "Generating eval_ml_regional.sh..."
apptainer exec "${GPU_ARGS[@]}" --bind "${REPO_ROOT}:${REPO_ROOT}" --bind "${DATA_DIR_REGIONAL}:/data/" "${PYTHON_SIF}" \
  generate-evaluate-ml-jobs \
  "/data/" \
  "${MODEL_OUTPUT}" \
  --duration 5 \
  --duration 10 \
  --duration 20 \
  --duration 30 \
  --duration 60 \
  --sampling-rate 10 \
  --sampling-rate 20 \
  --sampling-rate 40 \
  --sampling-rate 60 \
  --rolling-window 1 \
  --rolling-window 5 \
  --rolling-window 10 \
  --rolling-window 50 \
  --rolling-window 100 \
  --rolling-window 150 \
  --trainride-start-seconds 0 \
  --trainride-start-seconds 60 \
  --trainride-start-seconds 300 \
  --trainride-start-seconds 600 \
  --trainride-start-seconds 900 \
  --hparams-path "${HPARAMS_PATH}" \
  --test-fraction "${TEST_FRACTION}" \
  --threshold 0.0 \
  --batch-size 1024 \
  --seed "${SEED}" \
  --results-dir "${RESULTS_DIR_COLOC_REGIONAL}" \
  --python-sif "${PYTHON_SIF}" \
  --bind-path "${DATA_DIR_REGIONAL}:/data/" \
  --parallel-jobs "${PARALLEL_JOBS}" \
  --parallel-output-dir "${RESULTS_DIR_COLOC_REGIONAL}/eval_ml_logs" \
  -o "${EVAL_SCRIPT_REGIONAL}"

echo "  → ${EVAL_SCRIPT_REGIONAL}"

# Generate eval_majority_ml.sh
echo "Generating eval_majority_ml.sh..."
apptainer exec "${GPU_ARGS[@]}" --bind "${REPO_ROOT}:${REPO_ROOT}" --bind "${DATA_DIR}:/data/" "${PYTHON_SIF}" \
  generate-majority-vote-jobs \
  "/data/" \
  "${MODEL_OUTPUT}" \
  --duration 5 \
  --duration 10 \
  --duration 20 \
  --duration 30 \
  --duration 60 \
  --sampling-rate 10 \
  --sampling-rate 20 \
  --sampling-rate 40 \
  --sampling-rate 60 \
  --rolling-window 1 \
  --rolling-window 5 \
  --rolling-window 10 \
  --rolling-window 50 \
  --rolling-window 100 \
  --rolling-window 150 \
  --trainride-start-seconds 0 \
  --trainride-start-seconds 60 \
  --trainride-start-seconds 300 \
  --trainride-start-seconds 600 \
  --trainride-start-seconds 900 \
  --hparams-path "${HPARAMS_PATH}" \
  --test-fraction "${TEST_FRACTION}" \
  --threshold 0.0 \
  --batch-size 1024 \
  --seed "${SEED}" \
  --results-dir "${RESULTS_DIR_COLOC}" \
  --python-sif "${PYTHON_SIF}" \
  --bind-path "${DATA_DIR}:/data/" \
  --parallel-jobs "${PARALLEL_JOBS}" \
  --parallel-output-dir "${RESULTS_DIR_COLOC}/majority_logs" \
  -o "${MAJORITY_SCRIPT}"

echo "  → ${MAJORITY_SCRIPT}"

chmod +x "${EVAL_SCRIPT}" "${EVAL_SCRIPT_LONG_DISTANCE}" "${EVAL_SCRIPT_REGIONAL}" "${MAJORITY_SCRIPT}"

echo ""
echo "Job scripts generated."

# ── Step 2/6: Run eval_ml.sh ────────────────────────────────────────────────
echo ""
echo "=============================================================================="
echo " Step 2/6: Run eval_ml.sh [all trains]"
echo "=============================================================================="
echo ""

bash "${EVAL_SCRIPT}"

echo ""
echo "eval_ml.sh complete."

# ── Step 3/6: Run eval_ml_long_distance.sh ──────────────────────────────────
echo ""
echo "=============================================================================="
echo " Step 3/6: Run eval_ml_long_distance.sh"
echo "=============================================================================="
echo ""

bash "${EVAL_SCRIPT_LONG_DISTANCE}"

echo ""
echo "eval_ml_long_distance.sh complete."

# ── Step 4/6: Run eval_ml_regional.sh ───────────────────────────────────────
echo ""
echo "=============================================================================="
echo " Step 4/6: Run eval_ml_regional.sh"
echo "=============================================================================="
echo ""

bash "${EVAL_SCRIPT_REGIONAL}"

echo ""
echo "eval_ml_regional.sh complete."

# ── Step 5/6: Run eval_majority_ml.sh ───────────────────────────────────────
echo ""
echo "=============================================================================="
echo " Step 5/6: Run eval_majority_ml.sh"
echo "=============================================================================="
echo ""

bash "${MAJORITY_SCRIPT}"

echo ""
echo "eval_majority_ml.sh complete."

# ── Step 6/6: Compute majority-vote metrics ─────────────────────────────────
echo ""
echo "=============================================================================="
echo " Step 6/6: Compute majority-vote metrics: MCC, F1, etc."
echo "=============================================================================="
echo ""

apptainer exec "${GPU_ARGS[@]}" --bind "${REPO_ROOT}:${REPO_ROOT}" "${PYTHON_SIF}" \
  evaluate-majority-ml metrics \
  --results-dir "${RESULTS_DIR_COLOC}"

echo ""
echo "Majority-vote metrics computation complete."

echo ""
echo "=============================================================================="
echo " Coloc ML evaluation - COMPLETE"
echo "=============================================================================="
echo " Outputs:"
echo "   Eval results:       ${RESULTS_DIR_COLOC}"
echo "   Long distance:      ${RESULTS_DIR_COLOC_LONG_DISTANCE}"
echo "   Regional:           ${RESULTS_DIR_COLOC_REGIONAL}"
echo "   Eval script:        ${EVAL_SCRIPT}"
echo "   Long distance:      ${EVAL_SCRIPT_LONG_DISTANCE}"
echo "   Regional:           ${EVAL_SCRIPT_REGIONAL}"
echo "   Majority script:    ${MAJORITY_SCRIPT}"
echo "=============================================================================="
