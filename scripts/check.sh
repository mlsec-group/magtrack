#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SIF="${PYTHON_SIF:-${REPO_ROOT}/python.sif}"

REQUIRED_GIB="${CHECK_REQUIRED_GIB:-500}"
CHECK_PATH="${CHECK_PATH:-${REPO_ROOT}}"
RECOMMENDED_RAM_GIB="${CHECK_RECOMMENDED_RAM_GIB:-64}"

fails=0
warns=0

if [[ -t 1 ]]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_FAIL=$'\033[31m'; C_OFF=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_FAIL=""; C_OFF=""
fi

pass() { printf '  %sok%s    %s\n' "${C_OK}" "${C_OFF}" "$*"; }
warn() { printf '  %swarn%s  %s\n' "${C_WARN}" "${C_OFF}" "$*"; warns=$(( warns + 1 )); }
fail() { printf '  %sFAIL%s  %s\n' "${C_FAIL}" "${C_OFF}" "$*"; fails=$(( fails + 1 )); }
note() { printf '        %s\n' "$*"; }

echo "=============================================================================="
echo " Artifact preflight check"
echo " Repository: ${REPO_ROOT}"
echo "=============================================================================="

# ── 1. Host tools ─────────────────────────────────────────────────────────────
echo
echo "Host tools"

for entry in \
  "apptainer:runs every pipeline" \
  "curl:scripts/download.sh" \
  "unzip:unpacking the Zenodo archives" \
  "sha256sum:verifying the download"
do
  tool="${entry%%:*}"; why="${entry#*:}"
  if command -v "${tool}" >/dev/null 2>&1; then
    pass "${tool}"
  else
    fail "${tool} is not installed  (needed for: ${why})"
  fi
done

if (( BASH_VERSINFO[0] < 4 )); then
  fail "bash ${BASH_VERSION} is too old; the scripts need bash 4 or newer"
else
  pass "bash ${BASH_VERSION%%(*}"
fi

# ── 2. Repository scripts ─────────────────────────────────────────────────────
echo
echo "Repository scripts"

n_scripts=0
not_executable=()
while IFS= read -r script; do
  n_scripts=$(( n_scripts + 1 ))
  [[ -x "${script}" ]] || not_executable+=("${script#${REPO_ROOT}/}")
done < <(find "${REPO_ROOT}/scripts" -type f -name '*.sh' 2>/dev/null | sort)

if (( n_scripts == 0 )); then
  fail "no scripts found under ${REPO_ROOT}/scripts"
elif (( ${#not_executable[@]} == 0 )); then
  pass "all ${n_scripts} scripts are executable"
else
  fail "${#not_executable[@]} of ${n_scripts} scripts are not executable:"
  for script in "${not_executable[@]}"; do
    note "${script}"
  done
  note "fix with:  chmod +x ${not_executable[*]}"
fi

# ── 3. Container image ────────────────────────────────────────────────────────
echo
echo "Container image"

if [[ ! -f "${PYTHON_SIF}" ]]; then
  fail "${PYTHON_SIF} not found"
  note "build it with:  scripts/build.sh"
elif ! command -v apptainer >/dev/null 2>&1; then
  fail "cannot inspect ${PYTHON_SIF}: apptainer is not installed"
elif ! apptainer inspect "${PYTHON_SIF}" >/dev/null 2>&1; then
  fail "${PYTHON_SIF} exists but apptainer cannot read it"
  note "rebuild it with:  scripts/build.sh"
else
  built="$(apptainer inspect "${PYTHON_SIF}" 2>/dev/null \
           | sed -n 's/^org.label-schema.build-date: //p')"
  pass "${PYTHON_SIF##*/} (built ${built:-unknown})"

  newer="$(find "${REPO_ROOT}/src" "${REPO_ROOT}/requirements.txt" "${REPO_ROOT}/pyproject.toml" \
           -newer "${PYTHON_SIF}" 2>/dev/null | head -1)"
  if [[ -n "${newer}" ]]; then
    warn "the image is older than the source (e.g. ${newer#${REPO_ROOT}/})"
    note "rebuild so the containers pick it up:  scripts/build.sh"
  fi

  expected=(create-colocation-datasets evaluate-coloc-distance create-tmd-dataset
            evaluate-tmd evaluate-nor-tmd traintrack-dataset plot-results
            generate-latex-vars generate-tables train-ml-model
            hyperparameter-search ml-inference evaluate-majority-ml
            benchmark-inference fast-evaluation generate-evaluate-ml-jobs
            generate-majority-vote-jobs parallel unzip)
  missing=()
  read -r -a missing <<<"$(apptainer exec "${PYTHON_SIF}" bash -c '
      for c in "$@"; do command -v "$c" >/dev/null 2>&1 || printf "%s " "$c"; done
    ' _ "${expected[@]}" 2>/dev/null)"
  if (( ${#missing[@]} == 0 )); then
    pass "all expected commands present in the image"
  else
    fail "missing inside the image: ${missing[*]}"
    note "rebuild after adding them:  scripts/build.sh"
  fi
fi

# ── 4. Disk space ─────────────────────────────────────────────────────────────
echo
echo "Disk space"

avail_kib="$(df -Pk "${CHECK_PATH}" 2>/dev/null | tail -1 | tr -s ' ' | cut -d' ' -f4)"
if [[ -z "${avail_kib//[0-9]/}" && -n "${avail_kib}" ]]; then
  avail_gib=$(( avail_kib / 1024 / 1024 ))
  if (( avail_gib >= REQUIRED_GIB )); then
    pass "${avail_gib} GiB free at ${CHECK_PATH} (need about ${REQUIRED_GIB} GiB)"
  else
    fail "${avail_gib} GiB free at ${CHECK_PATH}, about ${REQUIRED_GIB} GiB needed"
    note "consider deleting the downloaded data after extraction to free up space."
  fi
else
  warn "could not determine the free space at ${CHECK_PATH}"
fi

# ── 5. GPU ────────────────────────────────────────────────────────────────────
echo
echo "GPU"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "no nvidia-smi; the learning-based pipeline will run on CPU"
  note "that is supported - the coloc_ml scripts detect this and fall back"
elif ! nvidia-smi -L >/dev/null 2>&1; then
  warn "nvidia-smi is installed but reports no usable GPU; CPU will be used"
else
  n_gpus="$(nvidia-smi -L | wc -l)"
  pass "${n_gpus} GPU(s): $(nvidia-smi -L | head -1 | cut -d'(' -f1)"

  # A GPU on the host is only useful if the container can reach it.
  if [[ -f "${PYTHON_SIF}" ]] && command -v apptainer >/dev/null 2>&1; then
    if cuda="$(apptainer exec --nv "${PYTHON_SIF}" python3 -c \
                 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>/dev/null)"
    then
      case "${cuda}" in
        True*) pass "torch sees CUDA inside the container (${cuda#True })" ;;
        *)     warn "the container starts with --nv but torch reports no CUDA device"
               note "the pipelines will fall back to CPU" ;;
      esac
    else
      warn "could not run torch with --nv inside the container"
      note "the pipelines will fall back to CPU"
    fi
  fi
fi

# ── 6. Memory ─────────────────────────────────────────────────────────────────
echo
echo "Memory and CPU"

mem_kib="$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo 2>/dev/null)"
if [[ -n "${mem_kib}" ]]; then
  mem_mib=$(( mem_kib / 1024 ))
  mem_gib="$(LC_ALL=C awk -v m="${mem_mib}" 'BEGIN { printf "%.1f", m / 1024 }')"
  min_mib=$(( RECOMMENDED_RAM_GIB * 1024 * 94 / 100 ))
  if (( mem_mib >= min_mib )); then
    pass "${mem_gib} GiB RAM"
  else
    warn "${mem_gib} GiB RAM, ${RECOMMENDED_RAM_GIB} GiB recommended"
    note "the colocation evaluation holds a whole dataset in memory per worker"
  fi
else
  warn "could not determine the amount of RAM"
fi

if command -v nproc >/dev/null 2>&1; then
  pass "$(nproc) CPU core(s)"
fi

# ── 7. Data ───────────────────────────────────────────────────────────────────
echo
echo "Datasets"

list_zip_files() {
  awk '
    /^zip_files:[[:space:]]*$/ { inlist = 1; next }
    inlist && /^-[[:space:]]/  { sub(/^-[[:space:]]*/, ""); sub(/[[:space:]]+$/, ""); print; next }
    inlist                     { exit }
  ' "$1"
}

check_dataset() {
  local dir="$1" yml="$2" what="$3"
  local path="${REPO_ROOT}/${dir}/${yml}"

  if [[ ! -f "${path}" ]]; then
    warn "${dir}/${yml} not found (${what})"
    note "download it with:  scripts/download.sh"
    return
  fi

  local names total=0 missing=0 examples=()
  mapfile -t names < <(list_zip_files "${path}")
  total="${#names[@]}"

  if (( total == 0 )); then
    warn "${dir}/${yml} lists no recordings under 'zip_files:'"
    return
  fi

  local name
  for name in "${names[@]}"; do
    if [[ ! -f "${REPO_ROOT}/${dir}/${name}.zip" ]]; then
      missing=$(( missing + 1 ))
      if (( ${#examples[@]} < 3 )); then
        examples+=("${name}.zip")
      fi
    fi
  done

  if (( missing == 0 )); then
    pass "${dir}: all ${total} recording(s) listed in ${yml} are present"
  else
    fail "${dir}: ${missing} of ${total} recording(s) listed in ${yml} are missing"
    note "for example: ${examples[*]}"
    note "re-run scripts/download.sh; it resumes and only fetches what is absent"
  fi

  local on_disk
  on_disk="$(find "${REPO_ROOT}/${dir}" -maxdepth 1 -name '*.zip' 2>/dev/null | wc -l)"
  if (( on_disk > total )); then
    note "$(( on_disk - total )) .zip file(s) in the directory are not listed in ${yml}"
  fi
}

check_dataset "traintrack-dataset"   "traintrack.yml"        "train recordings"
check_dataset "no-trainride-dataset" "no_trainride_data.yml" "non-train recordings"

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "=============================================================================="
if (( fails > 0 )); then
  echo " ${fails} check(s) FAILED, ${warns} warning(s)."
  echo " Fix the failures above before running the pipelines."
  echo "=============================================================================="
  exit 1
fi
if (( warns > 0 )); then
  echo " All required checks passed, with ${warns} warning(s)."
else
  echo " All checks passed."
fi
echo "=============================================================================="
exit 0
