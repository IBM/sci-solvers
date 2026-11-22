#!/usr/bin/env bash
# (C) Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0
#
# Run one (molecule, solver config) pair up a ladder of subspace sizes, stopping at the
# first size that fails: a larger size can only need more time and memory. Each size is
# one run_matrix call in its own directory, so the scan can be resumed (--resume) and its
# results summarized with sci_solvers_benchmark.summarize.
#
# Usage:
#   benchmark/scripts/run_ladder.sh <molecule> <config.yaml> --base-dir DIR --output-dir DIR \
#       [--sizes a,b,c] [--num-runs N] [--run-timeout SEC] [--size-timeout SEC] [--resume]
#
# Sizes are bitstrings drawn from the quantum-sampled input, or n_alpha for a synthetic
# input (which picks the hf_product_<n_alpha>.npz file). Writes, under --output-dir:
#   size_<N>/per_run/...  size_<N>/solver.log  ladder.log (one PASS/FAIL line per size)
#
# Run it in an allocation with the CPU, memory and GPU limits you want measured; the
# script itself starts no jobs.
set -euo pipefail

QUANTUM_SIZES="100,200,500,1000,2000,5000,6000,7000,8000,10000,12000,15000"
SYNTHETIC_SIZES="100,200,500,1000,2000,3000,4000,6000,8000,12000,15000,18000"

usage() {
  sed -n '6,20p' "$0" | sed 's/^# \{0,1\}//' >&2
}

[[ $# -ge 2 ]] || { usage; exit 2; }
MOL="$1"; CFG="$2"; shift 2
BASE_DIR="" OUTPUT_DIR="" SIZES="" NUM_RUNS=3 RUN_TIMEOUT=21600 SIZE_TIMEOUT=0 RESUME=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --num-runs) NUM_RUNS="$2"; shift 2 ;;
    --run-timeout) RUN_TIMEOUT="$2"; shift 2 ;;
    --size-timeout) SIZE_TIMEOUT="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "${BASE_DIR}" && -n "${OUTPUT_DIR}" ]] || { usage; exit 2; }
[[ -f "${CFG}" ]] || { echo "no such config: ${CFG}" >&2; exit 2; }
if [[ -z "${SIZES}" ]]; then
  if [[ -d "${BASE_DIR}/molecules/${MOL}/hf_expansion" ]]; then
    SIZES="${SYNTHETIC_SIZES}"
  else
    SIZES="${QUANTUM_SIZES}"
  fi
fi
mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/ladder.log"

# Prints "ok" if every per-run JSON under $1 has status "ok" and at least $2 timed runs.
size_ok() {
  python3 - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
paths = list(Path(sys.argv[1]).glob("per_run/**/*.json"))
good = bool(paths)
for p in paths:
    d = json.loads(p.read_text())
    good &= d.get("status") == "ok" and (d.get("results") or {}).get("num_runs", 0) >= int(sys.argv[2])
print("ok" if good else "no")
PY
}

echo "=== ladder: molecule=${MOL} config=${CFG} sizes=${SIZES} num_runs=${NUM_RUNS} run_timeout=${RUN_TIMEOUT}s ===" | tee -a "${LOG}"
IFS=',' read -ra LADDER <<< "${SIZES}"
for SIZE in "${LADDER[@]}"; do
  DIR="${OUTPUT_DIR}/size_${SIZE}"
  if [[ -d "${DIR}" ]]; then
    if ${RESUME} && [[ "$(size_ok "${DIR}" "${NUM_RUNS}")" == ok ]]; then
      echo "PASS size=${SIZE} (resumed)" | tee -a "${LOG}"
      continue
    fi
    rm -rf "${DIR}"
  fi
  mkdir -p "${DIR}"
  CMD=(python -m sci_solvers_benchmark.run_matrix --config_file "${CFG}"
       --base_dir "${BASE_DIR}" --output_dir "${DIR}"
       --molecules "[\"${MOL}\"]" --subsample_size "${SIZE}"
       --num_runs "${NUM_RUNS}" --run_timeout "${RUN_TIMEOUT}")
  if [[ "${SIZE_TIMEOUT}" != 0 ]]; then
    CMD=(timeout --signal=TERM --kill-after=60s "${SIZE_TIMEOUT}" "${CMD[@]}")
  fi
  START=${SECONDS}
  set +e
  "${CMD[@]}" > "${DIR}/solver.log" 2>&1
  CODE=$?
  set -e
  WALL=$((SECONDS - START))
  if [[ ${CODE} -eq 0 && "$(size_ok "${DIR}" "${NUM_RUNS}")" == ok ]]; then
    echo "PASS size=${SIZE} wall=${WALL}s" | tee -a "${LOG}"
  else
    # 124: --size-timeout fired; 137: killed (often the kernel's out-of-memory killer).
    echo "FAIL size=${SIZE} wall=${WALL}s exit=${CODE}" | tee -a "${LOG}"
    tail -5 "${DIR}/solver.log" | tee -a "${LOG}"
    break
  fi
done
