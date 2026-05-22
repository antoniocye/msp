#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p experiments/results

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1

exec /usr/bin/nice -n 15 .venv/bin/python experiments/src/sparse_concept_rare_event_suite.py \
  --datasets fashion_mnist cifar10 cifar100 \
  --model-type resnet18 \
  --use-public-weights \
  --freeze-backbone \
  --seeds 20260521 20260522 20260523 \
  --dictionary-seeds 0 1 2 \
  --train-n 12000 \
  --epochs 2 \
  --concept-calib-n 5000 \
  --build-n 12000 \
  --select-n 20000 \
  --ref-n 30000 \
  --risk-eval-n 12000 \
  --intervention-n 8000 \
  --pool-n 8192 \
  --query-count 100 \
  --max-candidates 20000 \
  --query-mode external \
  --rate-min 0.001 \
  --rate-max 0.04 \
  --budgets 64 128 256 512 1024 2048 \
  --reps 8 \
  --success-ratio-threshold 0.80 \
  --success-ci-max 1.0 \
  --max-break-even-queries 20 \
  --ablation-gap 1.05 \
  --torch-threads 2 \
  --torch-inter-op-threads 1
