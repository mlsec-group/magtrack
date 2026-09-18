#!/usr/bin/env bash
set -euo pipefail

# Times both colocation approaches per recording pair, across recording lengths
# and sampling rates, and writes one row per timed pair.
#
#   dtw          one DTW over the two stitched L-second traces
#   ml_majority  one batched forward pass over that pair's k chunk-pairs plus
#                the rolling majority vote over the k predictions
#
# Both run on CPU: the distance approach has no GPU path, so timing them on the
# same device keeps the ratio meaningful.

# Paths
DATA_DIR="${BENCH_DATA_DIR:-datasets/coloc_datasets/normalized_trace_all_trains}"
RESULTS_CSV="${BENCH_RESULTS_CSV:-results/benchmark/inference_times.csv}"
MODEL_PATH="${BENCH_MODEL_PATH:-coloc_model/colocation_net.pth}"
HPARAMS_PATH="${BENCH_HPARAMS_PATH:-coloc_model/model_hparams.yaml}"

# Parameter grid
RECORDING_LENGTHS=(${BENCH_RECORDING_LENGTHS:-10 30 60 300 600 900})
SAMPLING_RATES=(${BENCH_SAMPLING_RATES:-10 20 40 60})

# Runtime options
CHUNK_DURATION="${BENCH_CHUNK_DURATION:-5}"
DATASETS_PER_CELL="${BENCH_DATASETS_PER_CELL:-10}"
PAIRS="${BENCH_PAIRS:-50}"
REPEATS="${BENCH_REPEATS:-5}"
WARMUP="${BENCH_WARMUP:-3}"
RADIUS="${BENCH_RADIUS:-1}"
SEED="${BENCH_SEED:-magtrack}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SIF="python.sif"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  echo "Apptainer image not found: ${PYTHON_SIF}"
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Dataset directory not found: ${DATA_DIR}"
  echo "Generate it first with scripts/dataset_creation/coloc_datasets.sh"
  exit 1
fi

for path in "${MODEL_PATH}" "${HPARAMS_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Not found: ${path}"
    echo "Run scripts/coloc_ml/train.sh first, or set BENCH_MODEL_PATH / BENCH_HPARAMS_PATH."
    exit 1
  fi
done

mkdir -p "$(dirname "${RESULTS_CSV}")"

LENGTH_ARGS=()
for length in "${RECORDING_LENGTHS[@]}"; do
  LENGTH_ARGS+=(--recording-length "${length}")
done
RATE_ARGS=()
for rate in "${SAMPLING_RATES[@]}"; do
  RATE_ARGS+=(--sampling-rate "${rate}")
done

echo "Using apptainer image: ${PYTHON_SIF}"
echo "Dataset dir:  ${DATA_DIR}"
echo "Model:        ${MODEL_PATH}"
echo "Lengths:      ${RECORDING_LENGTHS[*]} s"
echo "Sampling:     ${SAMPLING_RATES[*]} Hz"
echo "Chunks:       ${CHUNK_DURATION}s  (vote length = recording length / ${CHUNK_DURATION})"
echo "Per cell:     ${DATASETS_PER_CELL} dataset(s) x ${PAIRS} pair(s) x ${REPEATS} repeat(s)"
echo "Results CSV:  ${RESULTS_CSV}"

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${PYTHON_SIF}" \
  benchmark-inference \
  --data-dir "${DATA_DIR}" \
  --output "${RESULTS_CSV}" \
  --model-path "${MODEL_PATH}" \
  --hparams-path "${HPARAMS_PATH}" \
  "${LENGTH_ARGS[@]}" \
  "${RATE_ARGS[@]}" \
  --chunk-duration "${CHUNK_DURATION}" \
  --datasets-per-cell "${DATASETS_PER_CELL}" \
  --pairs "${PAIRS}" \
  --repeats "${REPEATS}" \
  --warmup "${WARMUP}" \
  --radius "${RADIUS}" \
  --seed "${SEED}"

echo
echo "Done."
echo "  ${RESULTS_CSV}"
echo "Next: scripts/benchmark/paper.sh"
