# External Tool CIFAR-10 Auditing Runs

This folder runs the rare-event auditing comparison with external basis tools:

- `sklearn_sdl`: `sklearn.decomposition.MiniBatchDictionaryLearning`
- `saelens_sae`: SAELens `StandardTrainingSAE`
- `spd`: Goodfire SPD `ComponentModel` / `optimize` on the trained CIFAR-10 linear head

The original repo runner is not modified. This runner imports its CIFAR-10 data/model/query/estimator
utilities and swaps the learned internal basis.

## Quick Run

Install the external tool checkouts used by the runner:

```bash
external_tool_auditing/scripts/setup_external_tools.sh
```

```bash
external_tool_auditing/scripts/run_cifar10_external_tools.sh
```

Outputs are written under `external_tool_auditing/results/`.

## Notes

By default, the runner uses `--query-policy independent_external`: it selects rare queries once,
before fitting any SAE/SDL/SPD basis, using only labels, predictions, confidence, and pixel/image
concepts. The same frozen query set is then evaluated for every basis. This is the unbiased setup
for comparing representations. The older representation-native setup is still available with
`--query-policy basis_specific`.

SPD is parameter decomposition rather than an activation reconstruction dictionary. In this run it
decomposes the trained penultimate-to-logit `head` linear layer, and its learned causal-importance
gates are used as the reusable internal basis codes.

After an independent-query budget sweep, post-process the basis-independent controls before reading
the comparison plot. This makes baselines such as `output_comp`, `output_active`, `pca_comp`,
`random_comp`, and `mc` identical across tool panels, while preserving the original internal
SAE/SDL/SPD estimates:

```bash
.venv/bin/python external_tool_auditing/unify_basis_independent_controls.py \
  --results-dir external_tool_auditing/results/paper_independent_budget_sweep

.venv/bin/python external_tool_auditing/rate_binned_rmse.py \
  --results-dir external_tool_auditing/results/paper_independent_budget_sweep

.venv/bin/python external_tool_auditing/plot_budget_trends.py \
  --results-dir external_tool_auditing/results/paper_independent_budget_sweep \
  --ci-file cifar10_external_tool_ci_unified_controls.csv \
  --single-plot \
  --metric rmse \
  --output cifar10_external_tool_budget_trends_unified_single.png
```
