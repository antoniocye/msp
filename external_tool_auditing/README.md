# External Tool Auditing

Experiment harness for the paper: [Amortized Rare-Event Auditing Under a Competing-With-Sampling Criterion](../sparse-concept-rare-event-auditing-paper.md).

It runs the CIFAR-10 rare-event audit with:

- `sklearn_sdl`: sklearn sparse dictionary learning
- `saelens_sae`: SAELens sparse autoencoder
- `spd`: Goodfire Stochastic Parameter Decomposition

## Run

```bash
external_tool_auditing/scripts/setup_external_tools.sh
external_tool_auditing/scripts/run_paper_budget_sweep_external_tools.sh
```

The paper-referenced artifacts are in `external_tool_auditing/results/paper_independent_budget_sweep/`. Post-processing commands are listed in the paper appendix.
