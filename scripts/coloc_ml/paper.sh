#!/usr/bin/env bash
set -euo pipefail

# Generates paper artifacts from the coloc ML evaluation results.
#
# Run scripts/coloc_ml/evaluate.sh first; this reads its results.
#
# Usage:
#   bash scripts/coloc_ml/paper.sh
#
# Environment variables (with defaults):
#   PAPER RESULTS_DIR   - Directory for coloc ML evaluation results (default: ./results/coloc_ml)
#   PAPER_FIGURE_DIR    - Directory for output figures
#   PAPER_VARS_DIR      - Directory for LaTeX variable files
#   PAPER_TABLES_DIR    - Directory for LaTeX table files
#   PAPER_RESULTS_CSV         - Path to coloc ML evaluation results CSV (default: ./results/coloc_ml/evaluation_results.results_ml.csv)
#   RESULTS_LONG_CSV    - Path to coloc ML evaluation results CSV for long-distance trains (default: ./results/coloc_ml/long_distance/evaluation_results.results_ml.csv)
#   RESULTS_REGIONAL_CSV- Path to coloc ML evaluation results CSV for regional trains (default: ./results/coloc_ml/regional/evaluation_results.results_ml.csv)
#   PAPER_MAJORITY_CSV        - Path to majority-vote evaluation results CSV (default: ./results/coloc_ml/majority_vote/master_metrics_by_k.csv)
#   PAPER_HPARAMS_PATH  - Model hyperparameters; the paper figures show the dataset configuration
#                         the model was trained on (default: ./coloc_model/model_hparams.yaml)
#   PAPER_HZ            - Sampling rate (Hz) of the paper figures (default: hz from the hyperparameters)
#   PAPER_CHUNK         - Chunk size (s) of the paper figures (default: chunk_size_seconds from the hyperparameters)
#   PAPER_WINDOW        - Rolling window of the majority-vote plot (default: window_size_seconds from the hyperparameters)
#   PYTHON_SIF          - Path to Apptainer image

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="${PYTHON_SIF:-./python.sif}"

RESULTS_DIR="${PAPER_RESULTS_DIR:-${REPO_ROOT}/results/}"

FIGURE_DIR="${PAPER_FIGURE_DIR:-${RESULTS_DIR}/paper/figures}"
DB_PATH="${COLOC_DB:-${RESULTS_DIR}/coloc_ml/hyperparam_search.db}"
VARS_DIR="${PAPER_VARS_DIR:-${RESULTS_DIR}/paper/vars}"
TABLES_DIR="${PAPER_TABLES_DIR:-${RESULTS_DIR}/paper/tables}"

# Result CSVs
RESULTS_CSV="${PAPER_RESULTS_CSV:-${RESULTS_DIR}/coloc_ml/evaluation_results.results_ml.csv}"
MAJORITY_CSV="${PAPER_MAJORITY_CSV:-${RESULTS_DIR}/coloc_ml/majority_vote/master_metrics_by_k.csv}"

# ── Helper: parse a simple YAML key from a flat YAML file ─────────────────────
yaml_get() {
  local file="$1" key="$2"
  grep -E "^${key}:" "$file" | head -1 | sed "s/^${key}:[[:space:]]*//" | tr -d '"' | tr -d "'"
}

# Dataset configuration shown in the paper figures: the one the model was trained on
HPARAMS_PATH="${PAPER_HPARAMS_PATH:-${REPO_ROOT}/coloc_model/model_hparams.yaml}"
PAPER_HZ="${PAPER_HZ:-$(yaml_get "${HPARAMS_PATH}" hz)}"
PAPER_CHUNK="${PAPER_CHUNK:-$(yaml_get "${HPARAMS_PATH}" chunk_size_seconds)}"
PAPER_WINDOW="${PAPER_WINDOW:-$(yaml_get "${HPARAMS_PATH}" window_size_seconds)}"

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

mkdir -p "${FIGURE_DIR}" "${VARS_DIR}" "${TABLES_DIR}"

echo "=============================================================================="
echo " Coloc ML Pipeline - paper.sh"
echo "=============================================================================="
echo " Python image:     ${PYTHON_SIF}"
echo " Figures:          ${FIGURE_DIR}"
echo " LaTeX vars:       ${VARS_DIR}"
echo " Tables:           ${TABLES_DIR}"
echo " Results CSV:      ${RESULTS_CSV}"
echo " Paper config:     ${PAPER_CHUNK}s chunks, window ${PAPER_WINDOW}, ${PAPER_HZ} Hz"
echo "=============================================================================="

# ── Helper ────────────────────────────────────────────────────────────────────
run_apptainer() {
  apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
    "$@"
}

# ── 1. MCC heatmap for the paper (pdf) ───────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo " 1. Generating the MCC paper heatmap"
echo "──────────────────────────────────────────────────────────────────────────────"

# Only the single-row MCC heatmap goes into the paper. plot-metric-heatmaps can
# still produce the grid, the wide paper variant and the other metrics when run
# by hand -- those are on by default and only switched off here.
for fmt in pdf; do
  echo "  Format: ${fmt}"
  run_apptainer plot-metric-heatmaps \
    "${RESULTS_CSV}" \
    --out-dir "${FIGURE_DIR}" \
    --metrics mcc \
    --no-grid-heatmaps \
    --no-paper-heatmaps \
    --paper-single-heatmaps \
    --paper-chunk "${PAPER_CHUNK}" \
    --paper-hz "${PAPER_HZ}" \
    --output-format "${fmt}"
done

# ── 2. Majority-vote plots (pdf) ─────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo " 2. Generating majority-vote plots"
echo "──────────────────────────────────────────────────────────────────────────────"

for fmt in pdf; do
  echo "  Format: ${fmt}"
  run_apptainer plot-majority-ml \
    --metrics-csv "${MAJORITY_CSV}" \
    --chunk-size "${PAPER_CHUNK}" \
    --window-size "${PAPER_WINDOW}" \
    --hz "${PAPER_HZ}" \
    --out-dir "${FIGURE_DIR}" \
    --output-format "${fmt}"
done

# ── 3. LaTeX tables ──────────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo " 3. Generating LaTeX tables"
echo "──────────────────────────────────────────────────────────────────────────────"

# 3a. Long vs regional comparison table
echo "  Generating long vs regional comparison table ..."
run_apptainer generate-longvsregional-table 

# 3b. LaTeX vars for ML results
echo "  Generating LaTeX vars for ML results ..."
run_apptainer generate-latex-vars ml-results \
  "${RESULTS_CSV}" "${VARS_DIR}/coloc_ml.tex"

# 3c. Hyperparameter search table -- only when the Optuna study is present.
if [[ -f "${DB_PATH}" ]]; then
  echo "  Generating hyperparameter search table ..."
  run_apptainer generate-hparam-search-table --db "${DB_PATH}" > "${TABLES_DIR}/hparam_search_table.tex"
else
  echo "  Skipping hyperparameter search table: ${DB_PATH} not found"
  echo "  (run scripts/coloc_ml/hyperparameter_search.sh to produce it)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================================================="
echo " Done."
echo "=============================================================================="
echo " Figures:   ${FIGURE_DIR}"
echo " Vars:      ${VARS_DIR}"
echo " Tables:    ${TABLES_DIR}"
echo " Paper:     ${REPO_ROOT}/results/paper/coloc_ml"
echo "=============================================================================="
