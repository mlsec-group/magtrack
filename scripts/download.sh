#!/usr/bin/env bash
set -euo pipefail

# Downloads the raw magnetometer recordings from Zenodo and extracts them into
# the layout the rest of the artifact expects:
#
#   traintrack-dataset/      train recordings + traintrack.yml
#   no-trainride-dataset/    non-train recordings + no_trainride_data.yml
#
# Needs only curl, unzip and coreutils.
# The download is resumable.

# ── Source ────────────────────────────────────────────────────────────────────
ZENODO_RECORD="${ZENODO_RECORD:-22206115}"
BASE_URL="${ZENODO_BASE_URL:-https://zenodo.org/records/${ZENODO_RECORD}/files}"
URL_SUFFIX="${ZENODO_URL_SUFFIX-?download=1}"

# ── The record ────────────────────────────────────────────────────────────────
FILES=(
  no_trainride_data_001.zip
  no_trainride_data_002.zip
  no_trainride_data_003.zip
  no_trainride_data.yml
  traintrack_data_001.zip
  traintrack_data_002.zip
  traintrack_data_003.zip
  traintrack_data_004.zip
  traintrack_data_005.zip
  traintrack_data_006.zip
  traintrack_data_007.zip
  traintrack_data_008.zip
  traintrack_data_009.zip
  traintrack_data_010.zip
  traintrack_data_011.zip
  traintrack_data_012.zip
  traintrack_data_013.zip
  traintrack_data_014.zip
  traintrack_data_015.zip
  traintrack_stats.yml
  traintrack.yml
)

META_FILES=(sha256sums.txt uncompress.sh)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ZENODO_DEST_DIR:-${REPO_ROOT}/traintrack-zenodo}"
OUTPUT_DIR="${ZENODO_OUTPUT_DIR:-${REPO_ROOT}}"

# ── Runtime options ───────────────────────────────────────────────────────────
SKIP_EXTRACT="${ZENODO_SKIP_EXTRACT:-0}"   # 1 = download only
VERIFY_ONLY="${ZENODO_VERIFY_ONLY:-0}"     # 1 = only re-check what is on disk
FILE_FILTER="${ZENODO_FILE_FILTER:-}"      # regex, for partial downloads
RETRIES="${ZENODO_RETRIES:-5}"

for tool in curl; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Required tool not found: ${tool}"
    exit 1
  fi
done

mkdir -p "${DEST_DIR}"

# Download unless the file is already there.
fetch() {
  local name="$1"
  local final="${DEST_DIR}/${name}" part="${DEST_DIR}/${name}.part"

  if [[ -f "${final}" ]]; then
    echo "  have    ${name}"
    return 0
  fi

  echo "  get     ${name}"
  if ! curl -fSL --progress-bar --retry "${RETRIES}" --retry-delay 5 --retry-connrefused \
            -C - -o "${part}" "${BASE_URL}/${name}${URL_SUFFIX}"; then
    echo
    echo "ERROR: downloading ${name} failed."
    echo "Whatever arrived is kept at ${part}; re-run this script to resume it."
    return 1
  fi
  mv -f "${part}" "${final}"
}

# ── Which files ───────────────────────────────────────────────────────────────
DATA_FILES=()
for name in "${FILES[@]}"; do
  if [[ -n "${FILE_FILTER}" ]] && ! grep -qE "${FILE_FILTER}" <<<"${name}"; then
    continue
  fi
  DATA_FILES+=("${name}")
done

if (( ${#DATA_FILES[@]} == 0 )); then
  echo "ERROR: no files selected."
  exit 1
fi

echo "=============================================================================="
echo " Zenodo record: ${ZENODO_RECORD}"
echo " Download to:   ${DEST_DIR}"
echo " Extract into:  ${OUTPUT_DIR}"
echo " Files:         ${#DATA_FILES[@]} of ${#FILES[@]}"
echo "=============================================================================="

# ── Re-check an existing download ─────────────────────────────────────────────
if (( VERIFY_ONLY )); then
  if [[ ! -f "${DEST_DIR}/sha256sums.txt" ]]; then
    echo "ERROR: ${DEST_DIR}/sha256sums.txt is missing; nothing to check against."
    exit 1
  fi
  echo
  echo "=== verifying against sha256sums.txt ==="
  ( cd "${DEST_DIR}" && sha256sum -c sha256sums.txt )
  exit 0
fi

# ── Manifests and the unpacking script ────────────────────────────────────────
echo
echo "=== manifests ==="
for name in "${META_FILES[@]}"; do
  rm -f "${DEST_DIR}/${name}"
  fetch "${name}"
done

# ── Data files ────────────────────────────────────────────────────────────────
echo
echo "=== ${#DATA_FILES[@]} file(s) ==="
n=0
for name in "${DATA_FILES[@]}"; do
  n=$(( n + 1 ))
  printf "[%2d/%2d]" "${n}" "${#DATA_FILES[@]}"
  fetch "${name}"
done

# ── Extract ───────────────────────────────────────────────────────────────────
if (( SKIP_EXTRACT )); then
  echo
  echo "Download complete. Skipping extraction (ZENODO_SKIP_EXTRACT=1)."
  echo "Run it yourself with:  bash ${DEST_DIR}/uncompress.sh ${OUTPUT_DIR}"
  exit 0
fi

if [[ ! -f "${DEST_DIR}/uncompress.sh" ]]; then
  echo "ERROR: ${DEST_DIR}/uncompress.sh is missing; cannot extract."
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo
  echo "ERROR: extracting needs 'unzip', which is not installed."
  echo "  Debian/Ubuntu:  sudo apt install unzip"
  echo "  Fedora/RHEL:    sudo dnf install unzip"
  echo "  Arch:           sudo pacman -S unzip"
  echo "The download is complete; re-run this script once unzip is available,"
  echo "or run:  bash ${DEST_DIR}/uncompress.sh ${OUTPUT_DIR}"
  exit 1
fi

echo
echo "=== extracting ==="
echo "uncompress.sh verifies every file against sha256sums.txt before unpacking."
echo

if ! bash "${DEST_DIR}/uncompress.sh" "${OUTPUT_DIR}"; then
  echo
  echo "ERROR: extraction failed."
  echo "If it reported a checksum mismatch, delete the file it named in"
  echo "${DEST_DIR} and re-run this script to fetch it again."
  exit 1
fi

echo
echo "Done."
echo "  ${OUTPUT_DIR}/traintrack-dataset"
echo "  ${OUTPUT_DIR}/no-trainride-dataset"
echo "Next: scripts/dataset_creation/coloc_datasets.sh"
