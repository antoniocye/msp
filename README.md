# Sparse Concept Rare-Event Auditing

This repo contains one small empirical result and the code that produced it. The question is whether a sparse activation basis, built once, can help estimate rare classifier-failure rates with fewer labels than uniform Monte Carlo sampling.

The main writeup is [sparse-concept-rare-event-auditing-paper.md](sparse-concept-rare-event-auditing-paper.md). The archived result is a CIFAR-10 run with a frozen ResNet-18 backbone initialized from public ImageNet weights. At 512 labels per query, the sparse-internal estimator reached `0.561`x the RMSE of same-budget Monte Carlo over 12 rare-failure queries. At 2048 labels per query, the sparse-internal result was not clearly better than Monte Carlo.

![CIFAR-10 archived run: RMSE ratios by budget](experiments/results/stronger/cifar10_sparse_concept_main_ratios.png)

## Layout

| Path | Purpose |
| --- | --- |
| `sparse-concept-rare-event-auditing-paper.md` | self-contained report |
| `sparse-concept-rare-event-auditing-references.bib` | references |
| `experiments/src/sparse_concept_rare_event_suite.py` | experiment runner, estimators, analysis, and plots |
| `experiments/results/stronger/cifar10_*` | archived CIFAR-10 tables and figures used in the report |
| `scripts/run_external_query_benchmark.sh` | low-priority local runner for the larger external-query benchmark |
| `requirements.txt` | Python dependencies |

Downloaded datasets, virtual environments, logs, checkpoints, and fresh local outputs are ignored by git.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The first full run downloads datasets into `data/`.

## Reproduce the Archived CIFAR-10 Run

```bash
.venv/bin/python experiments/src/sparse_concept_rare_event_suite.py \
  --dataset cifar10 \
  --model-type resnet18 \
  --use-public-weights \
  --freeze-backbone \
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
  --budgets 512 2048 \
  --reps 3
```

Fresh outputs are written to `experiments/results/sparse_concept_*`. The root paper is not overwritten. The archived paper artifacts remain under `experiments/results/stronger/cifar10_*`.

## Larger External-Query Benchmark

The larger benchmark is intended to stress-test the idea with more external queries, multiple model and dictionary seeds, and stronger baselines. It is long-running and is not needed to reproduce the archived paper result.

```bash
.venv/bin/python experiments/src/sparse_concept_rare_event_suite.py \
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
  --budgets 64 128 256 512 1024 2048 \
  --reps 8 \
  --torch-threads 6 \
  --torch-inter-op-threads 1
```

For a lower-priority local run with thread caps, checkpoints, and logs:

```bash
scripts/run_external_query_benchmark.sh > experiments/results/external_query_benchmark.log 2>&1
```

The runner uses `BENCH_THREADS` if set; otherwise it uses the machine's logical CPU count minus two.

## Archived Result

Under the paper's label-matched accounting, the sparse-internal estimator beat label-matched Monte Carlo after 5 audited queries in the archived CIFAR-10 run. The result is intentionally scoped: it supports this query distribution and accounting convention, not a general claim that sparse bases beat sampling everywhere.
