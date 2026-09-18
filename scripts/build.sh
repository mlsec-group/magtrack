#!/usr/bin/env bash
set -euo pipefail

# Builds the Apptainer image that every other script runs its commands in.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_SIF="${PYTHON_SIF:-${REPO_ROOT}/python.sif}"
APPTAINER_DEF="${APPTAINER_DEF:-${REPO_ROOT}/apptainer.def}"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "apptainer not found; install it first: https://apptainer.org/"
  exit 1
fi

if [[ ! -f "${APPTAINER_DEF}" ]]; then
  echo "Definition file not found: ${APPTAINER_DEF}"
  exit 1
fi

BUILD_ARGS=()
if [[ -f "${PYTHON_SIF}" ]]; then
  echo "Rebuilding existing image: ${PYTHON_SIF}"
  BUILD_ARGS+=(--force)
else
  echo "Building image: ${PYTHON_SIF}"
fi
echo "Definition:    ${APPTAINER_DEF}"
echo

cd "${REPO_ROOT}"
apptainer build "${BUILD_ARGS[@]}" "${PYTHON_SIF}" "${APPTAINER_DEF}"

echo
echo "Done. Run commands with:  apptainer exec ${PYTHON_SIF##*/} <command>"
echo "Verify the setup with:    scripts/check.sh"
