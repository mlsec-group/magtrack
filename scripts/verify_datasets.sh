#!/usr/bin/env bash
set -euo pipefail

# Verifies the generated datasets against the checksums written by prepare_datasets.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

failed=0
for type in tmd_datasets coloc_datasets coloc_baseline; do
  sums="results/datasets/${type}.sha256"
  if [[ ! -f "${sums}" ]]; then
    echo "MISSING  ${sums} (run scripts/prepare_datasets.sh first)"
    failed=1
  elif sha256sum -c --quiet "${sums}"; then
    echo "OK       ${type} ($(wc -l < "${sums}") datasets)"
  else
    echo "FAILED   ${type}"
    failed=1
  fi
done

exit "${failed}"
