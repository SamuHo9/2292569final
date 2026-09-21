#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_SLICER="${PYTHON_SLICER:-/c/Program Files/SlicerSALT 6.0.0/bin/PythonSlicer.exe}"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
LAUNCHER="$PROJECT_ROOT/ICP/run_icp_with_reference.py"

if [[ ! -f "$PYTHON_SLICER" ]]; then
  printf 'PythonSlicer not found: %s\n' "$PYTHON_SLICER" >&2
  exit 1
fi
if [[ ! -f "$LAUNCHER" ]]; then
  printf 'ICP launcher not found: %s\n' "$LAUNCHER" >&2
  exit 1
fi

printf 'Project: %s\n' "$PROJECT_ROOT"
printf 'Run ID: %s\n' "$RUN_ID"
printf 'Input left: %s\n' "$(cygpath -w "$PROJECT_ROOT/../merged_ds005602_ds004469/left_hippocampus")"
printf 'Input right: %s\n' "$(cygpath -w "$PROJECT_ROOT/../merged_ds005602_ds004469/right_hippocampus")"

run_icp() {
  local side="$1"
  local extra_args=("${@:2}")
  "$PYTHON_SLICER" "$LAUNCHER" --side "$side" --run-id "$RUN_ID" "${extra_args[@]}"
}

# Check both sides before starting either batch, so an input/path problem fails fast.
run_icp left --check-only
run_icp right --check-only
run_icp left
run_icp right

printf 'Both ICP batches completed. Results: %s/reruns/icp_legacy_reference_%s\n' "$PROJECT_ROOT" "$RUN_ID"
