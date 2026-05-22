# Competing With Sampling for Rare-Failure Auditing Using Sparse Concept Bases

Date: 2026-05-22

## Abstract

We test whether a reusable mechanistic explanation can reduce the cost of estimating rare failure rates. Following the "competing with sampling" framing, the target is not to classify individual failures directly, but to estimate the mean of a detector \(C_c\) over inputs \(x\), for many randomly sampled query contexts \(c\). The explanation algorithm \(E\) builds a sparse concept basis once from model activations. The estimator \(\mathbb{G}\) then uses that explanation to allocate labels more efficiently than same-budget Monte Carlo.

On CIFAR-10, using a ResNet-18 initialized with public ImageNet weights and a trained linear head, the sparse-internal estimator achieved an RMSE ratio of `0.561` against Monte Carlo at 512 labels per query over 12 rare-failure queries, with a model-query bootstrap interval of `[0.433, 0.753]`. Including the one-time build labels, it beat label-matched Monte Carlo after 5 audited queries. At 2048 labels per query, the result was directionally positive but not decisive: RMSE ratio `0.970`, interval `[0.698, 1.247]`.

This supports a narrow claim: for a fixed distribution of rare, concept-conditioned classifier failures, a sparse activation basis can serve as reusable audit infrastructure and can compete with sampling at low audit budgets. It does not establish a general theorem about mechanistic explanations or all rare failures.

## Formal Setup

Let \(M_\theta\) be a trained classifier and let \((x, y) \sim D\) be a labeled input from the deployment distribution. A query context \(c\) specifies a rare-failure predicate, such as "the sparse activation component is high, the true class is deer, and the model is wrong." The detector is

\[
C_c(M_\theta, x, y) \in \{0, 1\}.
\]

The audit target is the rare event rate

\[
p_c = \mathbb{E}_{(x,y)\sim D}[C_c(M_\theta, x, y)].
\]

The experiment instantiates the ARC-style objects as follows.

| Object | In this experiment |
| --- | --- |
| \(C\) | The family of rare-failure detectors \(C_c\). A detector may refer to model prediction, confidence, true label, hand-defined image concepts, and sparse activation predicates. |
| \(E\) | The one-time explanation builder. Given \(M_\theta\) and a build stream, it learns a sparse dictionary over penultimate activations, fits reusable atom heads, and stores a query compiler. Its output is the explanation \(\pi\). |
| \(\mathbb{G}\) | The audit estimator. Given \((M_\theta, \pi, c)\) and a label budget, it compiles \(c\) into a risk score, stratifies an unlabeled pool, labels a small sample, and returns a stratified estimate of \(p_c\). |

The sampling baseline estimates \(p_c\) by labeling uniformly sampled examples. A method "competes with sampling" empirically when its squared error is no worse than Monte Carlo at the same label budget, averaged over the query distribution. We report RMSE ratios, so values below `1.0` are better than sampling and correspond to squared-error ratios equal to the square of the reported number.

## Explanation And Estimators

The explanation \(\pi\) has three parts.

1. A sparse dictionary \(D_z\) over penultimate activations \(h(x)\). Each input receives sparse codes \(z(x)\), and component-threshold atoms such as \(z_j(x) > \tau_j\) become reusable internal concepts.
2. Atom heads for reusable predicates: true class, predicted class, model error, confidence thresholds, simple image concepts, and sparse-basis threshold atoms.
3. A compiler from Boolean query descriptions to scalar risk scores. Clauses are multiplied as approximate atom probabilities; disjunctions are combined by noisy OR.

The main explanation-based estimator is `sparse_internal`: it can directly use sparse codes from \(\pi\) when a query mentions sparse-basis predicates. The main baselines are:

| Method | Information used to rank examples |
| --- | --- |
| `mc` | no ranking; uniform sampling |
| `random_stratified` | random score with the same stratified estimator |
| `output_active` | model output confidence and predicted class |
| `output_comp` | learned output-only atom composition |
| `ase_output` | output-only active surrogate estimation |
| `input_concept_comp` | hand-defined image concepts and output atoms |
| `sparse_internal` | sparse explanation \(\pi\), atom heads, and compiled query score |

All methods estimate the same event labels \(C_c(M_\theta,x,y)\). The risk score only changes which examples are labeled.

## Experimental Design

The model is a ResNet-18 with public ImageNet weights and a frozen backbone. A CIFAR-10 classification head is trained on 6000 examples for one epoch. The evaluation accuracy is `0.681`.

| Quantity | Value |
| --- | --- |
| Dataset | CIFAR-10 |
| Model | ResNet-18, public ImageNet weights, frozen backbone |
| Seed | `20260521` |
| Sparse dictionary components | `48` |
| Query-selection stream | `8000` examples |
| Reference stream | `10000` examples |
| Build stream | `6000` examples |
| Queries | `12` |
| Rare-rate filter | `0.001` to `0.04` |
| Budgets | `512`, `2048` labels per query |
| Replicates | `3` |

The query generator samples families before seeing reference rates and keeps only queries whose selection-stream rates fall inside the rare-rate filter.

| Query family | Queries | Mean reference rate |
| --- | --- | --- |
| Sparse-basis class error | 4 | 0.0261 |
| Sparse-basis confusion | 1 | 0.0043 |
| Image-concept class error | 4 | 0.0062 |
| Image-concept confusion | 2 | 0.0026 |
| Random sparse-basis DNF | 1 | 0.0034 |

This distribution is intentionally not just output-confidence auditing. Most queries mention internal sparse-basis predicates or image concepts, then ask whether the classifier fails on that subpopulation.

## Results

RMSE ratios are relative to same-budget Monte Carlo. Intervals bootstrap model-query cells.

| Budget | Method | RMSE ratio vs MC | Bootstrap interval |
| --- | --- | --- | --- |
| 512 | `sparse_internal` | `0.561` | `[0.433, 0.753]` |
| 512 | `output_active` | `0.587` | `[0.368, 0.953]` |
| 512 | `output_comp` | `0.595` | `[0.464, 0.689]` |
| 512 | `ase_output` | `0.848` | `[0.688, 1.202]` |
| 512 | `input_concept_comp` | `0.973` | `[0.652, 1.439]` |
| 512 | `mc` | `1.000` | `[1.000, 1.000]` |
| 2048 | `output_comp` | `0.780` | `[0.546, 1.362]` |
| 2048 | `sparse_internal` | `0.970` | `[0.698, 1.247]` |
| 2048 | `output_active` | `0.993` | `[0.827, 1.419]` |
| 2048 | `mc` | `1.000` | `[1.000, 1.000]` |
| 2048 | `input_concept_comp` | `1.044` | `[0.781, 1.378]` |
| 2048 | `ase_output` | `1.205` | `[0.820, 2.137]` |

![RMSE ratios](experiments/results/stronger/cifar10_sparse_concept_main_ratios.png)

At 512 labels per query, `sparse_internal` is the strongest estimator and cleanly beats sampling: its RMSE is `56.1%` of Monte Carlo, or `31.5%` of Monte Carlo MSE. This is the clearest "competing with sampling" result in the experiment. At 2048 labels, Monte Carlo is already stronger, and the sparse-internal advantage is not statistically clear.

## Amortizing The Explanation Cost

The explanation \(\pi\) is not free. The sparse dictionary and atom bank are built using a 6000-example build stream. To test amortization, we compare each reusable method to a hypothetical Monte Carlo estimator with the same average label count per audited query: `build_n / audited_queries + budget`.

| Budget | Method | First query count beating label-matched MC | Final audited queries | Final RMSE ratio vs label-matched MC |
| --- | --- | --- | --- | --- |
| 512 | `sparse_internal` | 5 | 12 | `0.899` |
| 512 | `output_comp` | 6 | 12 | `0.954` |
| 512 | `input_concept_comp` | 1 | 12 | `1.561` |
| 2048 | `output_comp` | 2 | 12 | `0.711` |
| 2048 | `sparse_internal` | 4 | 12 | `0.883` |
| 2048 | `input_concept_comp` | 2 | 12 | `0.951` |

The result supports amortization: after a handful of queries, the one-time explanation cost is paid back under this query distribution.

## Query-Family Behavior

At 512 labels, the sparse-internal estimator is strongest on sparse-basis class errors: RMSE ratio `0.506` over four queries. Output-only composition is strongest on the single sparse-basis confusion query (`0.150`) and on the random sparse-basis DNF query (`0.355`). The image-concept compositor is strongest on image-concept class errors (`0.397`) and image-concept confusions (`0.203`).

This is the expected pattern. The sparse explanation helps most when the query actually refers to sparse internal structure. Output-only methods remain competitive when model confidence and predicted class already expose the failure. Hand-defined concepts help when the query is explicitly image-concept-defined.

![Family breakdown](experiments/results/stronger/cifar10_sparse_concept_family_breakdown.png)

## Dictionary And Intervention Checks

The learned dictionary has reconstruction \(R^2 = 0.251\), code density `0.167`, average active components `8.0`, and split-half stability cosine `0.402`. This is not evidence that every component is human-legible. It is evidence that the basis is sparse enough and stable enough to act as a reusable coordinate system for this audit workload.

As a directional intervention check, we selected six common class-confusion pairs, found the sparse component whose decoder direction most increased the target-vs-source class logit margin, and edited penultimate activations along that component. Adding the selected component increased the target confusion rate for all six pairs; subtracting it decreased the target confusion rate for all six pairs.

![Activation interventions](experiments/results/stronger/cifar10_sparse_concept_interventions.png)

## What The Result Supports

The positive claim is narrow:

1. \(E\) can build a reusable sparse explanation \(\pi\) from activations without fitting a new risk model per query.
2. \(\mathbb{G}\), given \((M_\theta,\pi,c)\), can reduce rare-event estimation error relative to sampling for this pre-specified CIFAR-10 query distribution at low label budgets.
3. The build cost can amortize across a modest number of related audits.

The result does not show that sparse components are complete causal explanations, that the estimator wins on arbitrary rare events, or that the approach removes the need for labels. The estimator still uses real event labels; the explanation improves where to spend them.

## Reproducibility

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the archived CIFAR-10 configuration:

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

Archived result files are under `experiments/results/stronger/cifar10_*`.

## Conclusion

This experiment gives a concrete, limited instance of competing with sampling. A sparse activation basis learned once from a trained model can be reused as explanation advice \(\pi\). Given that advice, \(\mathbb{G}\) can estimate rare, concept-conditioned failure rates more accurately than uniform sampling at low label budgets, and the build cost can be amortized over multiple related audits. The main remaining question is robustness: whether the same pattern holds across more seeds, stronger classifiers, and externally meaningful concept distributions.

## References

- Alignment Research Center, Competing with sampling, 2022. https://www.alignment.org/blog/competing-with-sampling/
- Kossen et al., Active Testing: Sample-Efficient Model Evaluation, ICML 2021. https://arxiv.org/abs/2103.05331
- Kossen et al., Active Surrogate Estimators, NeurIPS 2022. https://arxiv.org/abs/2202.06881
- Chen et al., Mandoline: Model Evaluation under Distribution Shift, ICML 2021. https://proceedings.mlr.press/v139/chen21i.html
- Geiger et al., Inducing Causal Structure for Interpretable Neural Networks, ICML 2022. https://proceedings.mlr.press/v162/geiger22a.html
- Marks et al., Sparse Feature Circuits, 2024. https://arxiv.org/abs/2403.19647
