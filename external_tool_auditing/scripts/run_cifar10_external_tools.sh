#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p external_tool_auditing/results

export OMP_NUM_THREADS="${BENCH_THREADS:-4}"
export MKL_NUM_THREADS="${BENCH_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${BENCH_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${BENCH_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${BENCH_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1

.venv/bin/python external_tool_auditing/cifar10_external_tool_audit.py "$@" \
  2>&1 | tee external_tool_auditing/results/cifar10_external_tool_audit.log
