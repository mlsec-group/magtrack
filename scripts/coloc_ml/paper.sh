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
#   RESULTS_CSV         - Path to coloc ML evaluation results CSV (default: ./test/coloc_ml/evaluation_results.results_ml.csv)
#   MAJORITY_CSV        - Path to majority-vote evaluation results CSV (default: ./test/coloc_ml/majority_vote/master_metrics_by_k.csv)
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
    --chunk-size 60 \
    --window-size 1 \
    --hz 10 \
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
