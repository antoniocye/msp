#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/../.."

results_dir="external_tool_auditing/results/paper_independent_exact"
mkdir -p "$results_dir"

export OMP_NUM_THREADS="${BENCH_THREADS:-2}"
export MKL_NUM_THREADS="${BENCH_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${BENCH_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${BENCH_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${BENCH_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1

.venv/bin/python external_tool_auditing/cifar10_external_tool_audit.py \
  --results-dir "$results_dir" \
  --basis-kinds sklearn_sdl saelens_sae spd \
  --query-policy independent_external \
  --query-mode external \
  --seeds 20260521 \
  --train-n 6000 \
  --epochs 1 \
  --concept-calib-n 2500 \
  --build-n 6000 \
  --select-n 8000 \
  --ref-n 10000 \
  --risk-eval-n 6000 \
  --intervention-n 5000 \
  --pool-n 3072 \
  --query-count 12 \
  --max-candidates 1800 \
  --rate-min 0.001 \
  --rate-max 0.04 \
  --dictionary-components 48 \
  --budgets 512 2048 \
  --reps 3 \
  --torch-threads "${BENCH_THREADS:-2}" \
  --torch-inter-op-threads 1 \
  "$@" 2>&1 | tee "$results_dir/cifar10_external_tool_audit.log"
