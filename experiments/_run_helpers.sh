# Shared helpers for per-model run scripts. Sourced from run_*.sh.
# Defines:
#   run_one <run_name> <log_subdir> -- <command...>
# which echoes a timestamped START/END frame and tees stdout+stderr to
# results/logs/<log_subdir>/<run_name>.log so a failed GPU run still has a
# trail.

set -euo pipefail

# Resolve repo root once, regardless of where the caller cd'd.
__SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${__SCRIPT_DIR}/.." && pwd)"

run_one() {
  local run_name="$1"; shift
  local log_subdir="$1"; shift
  if [[ "${1:-}" != "--" ]]; then
    echo "run_one: expected -- between log_subdir and command" >&2
    return 64
  fi
  shift

  local log_dir="${REPO_ROOT}/results/logs/${log_subdir}"
  mkdir -p "${log_dir}"
  local logfile="${log_dir}/${run_name}.log"

  local ts_start ts_end
  ts_start="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  echo "=== START: ${run_name} | ${ts_start} ==="
  # Run; tee stdout+stderr. Use PIPESTATUS to detect non-zero from the python
  # process even though tee always succeeds.
  set +e
  ( cd "${REPO_ROOT}" && "$@" ) 2>&1 | tee "${logfile}"
  local status=${PIPESTATUS[0]}
  set -e
  ts_end="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  if [[ ${status} -eq 0 ]]; then
    echo "=== END:   ${run_name} | ${ts_end} | OK ==="
  else
    echo "=== END:   ${run_name} | ${ts_end} | FAIL (exit ${status}) ==="
    # Don't abort the whole sweep on one failure — let later runs proceed.
    # Caller can re-check with: python -m experiments.status --pending-only
  fi
  return 0
}

# Default seed list for one-shot RQ1 scan. Override on CLI:
#   SEEDS="42 123 2024" bash experiments/run_ffd.sh
: "${SEEDS:=42}"
