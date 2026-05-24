# Sparse Concept Rare-Event Auditing

This repo studies whether reusable model structure can compete with Monte Carlo sampling for rare-event auditing.

Main paper: [Amortized Rare-Event Auditing Under a Competing-With-Sampling Criterion](sparse-concept-rare-event-auditing-paper.md)

## Contents

- `sparse-concept-rare-event-auditing-paper.md`: self-contained paper and results.
- `external_tool_auditing/`: external-tool experiment harness for SAE, sparse dictionary learning, and SPD.
- `external_tool_auditing/results/paper_independent_budget_sweep/`: artifact bundle referenced by the paper.
- `experiments/src/sparse_concept_rare_event_suite.py`: shared CIFAR-10 auditing utilities.

## Reproduce

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -r external_tool_auditing/requirements-external.txt

external_tool_auditing/scripts/setup_external_tools.sh
external_tool_auditing/scripts/run_paper_budget_sweep_external_tools.sh
```

Post-processing commands are listed in the paper's reproducibility appendix.
