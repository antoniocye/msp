# CIFAR-10 External Tool Auditing Run

Elapsed seconds: `1744.0`

This run compares external reusable bases on CIFAR-10. The rare-event queries are selected once, before fitting any SAE/SDL/SPD basis, using only labels, predictions, confidence, and pixel/image concepts.

Query policy: `independent_external`

## Internal Method RMSE Ratios

| basis_kind | budget | model_queries | rmse_ratio_vs_mc | rmse_ratio_ci_low | rmse_ratio_ci_high |
| --- | --- | --- | --- | --- | --- |
| saelens_sae | 64 | 12 | 1.038000 | 0.607236 | 1.326109 |
| saelens_sae | 128 | 12 | 0.482272 | 0.326683 | 0.636703 |
| saelens_sae | 256 | 12 | 0.618403 | 0.361451 | 1.021279 |
| saelens_sae | 512 | 12 | 0.734509 | 0.610854 | 0.899280 |
| saelens_sae | 1024 | 12 | 0.766468 | 0.582208 | 1.026384 |
| saelens_sae | 2048 | 12 | 0.970593 | 0.704305 | 1.210234 |
| sklearn_sdl | 64 | 12 | 0.911862 | 0.612821 | 1.200165 |
| sklearn_sdl | 128 | 12 | 0.560224 | 0.358489 | 1.010028 |
| sklearn_sdl | 256 | 12 | 0.975854 | 0.621036 | 1.244894 |
| sklearn_sdl | 512 | 12 | 0.907202 | 0.592020 | 1.501536 |
| sklearn_sdl | 1024 | 12 | 0.831150 | 0.576719 | 1.182028 |
| sklearn_sdl | 2048 | 12 | 0.813476 | 0.575049 | 1.018950 |
| spd | 64 | 12 | 1.299010 | 0.661642 | 1.876850 |
| spd | 128 | 12 | 0.600067 | 0.487205 | 0.834949 |
| spd | 256 | 12 | 0.597952 | 0.499055 | 0.781940 |
| spd | 512 | 12 | 0.835410 | 0.585916 | 1.158992 |
| spd | 1024 | 12 | 0.929131 | 0.575615 | 1.302734 |
| spd | 2048 | 12 | 0.863112 | 0.685442 | 1.045967 |

## Full Summary

| basis_kind | budget | method | model_queries | rmse_ratio_vs_mc | effective_mc_multiplier |
| --- | --- | --- | --- | --- | --- |
| saelens_sae | 64 | pca_comp | 12 | 0.544351 | 3.374752 |
| saelens_sae | 64 | per_query_rf | 12 | 0.687403 | 2.116298 |
| saelens_sae | 64 | output_comp | 12 | 0.744662 | 1.803358 |
| saelens_sae | 64 | input_concept_comp | 12 | 0.765813 | 1.705119 |
| saelens_sae | 64 | random_comp | 12 | 0.767961 | 1.695591 |
| saelens_sae | 64 | output_active | 12 | 0.996423 | 1.007193 |
| saelens_sae | 64 | mc | 12 | 1.000000 | 1.000000 |
| saelens_sae | 64 | internal | 12 | 1.038000 | 0.928122 |
| saelens_sae | 64 | random_stratified | 12 | 1.373816 | 0.529837 |
| saelens_sae | 64 | embedding_ase | 12 | 1.467304 | 0.464472 |
| saelens_sae | 64 | ase_output | 12 | 1.630403 | 0.376192 |
| saelens_sae | 64 | embedding_comp | 12 | 2.439462 | 0.168040 |
| saelens_sae | 128 | input_concept_comp | 12 | 0.444882 | 5.052550 |
| saelens_sae | 128 | internal | 12 | 0.482272 | 4.299484 |
| saelens_sae | 128 | random_comp | 12 | 0.498096 | 4.030645 |
| saelens_sae | 128 | per_query_rf | 12 | 0.519403 | 3.706731 |
| saelens_sae | 128 | pca_comp | 12 | 0.524096 | 3.640646 |
| saelens_sae | 128 | output_comp | 12 | 0.643024 | 2.418496 |
| saelens_sae | 128 | output_active | 12 | 0.686813 | 2.119937 |
| saelens_sae | 128 | embedding_comp | 12 | 0.765469 | 1.706650 |
| saelens_sae | 128 | random_stratified | 12 | 0.866744 | 1.331122 |
| saelens_sae | 128 | mc | 12 | 1.000000 | 1.000000 |
| saelens_sae | 128 | embedding_ase | 12 | 1.052241 | 0.903170 |
| saelens_sae | 128 | ase_output | 12 | 1.084924 | 0.849575 |
| saelens_sae | 256 | pca_comp | 12 | 0.373675 | 7.161643 |
| saelens_sae | 256 | output_comp | 12 | 0.503558 | 3.943666 |
| saelens_sae | 256 | per_query_rf | 12 | 0.510904 | 3.831081 |
| saelens_sae | 256 | input_concept_comp | 12 | 0.585917 | 2.912912 |
| saelens_sae | 256 | internal | 12 | 0.618403 | 2.614906 |
| saelens_sae | 256 | output_active | 12 | 0.642560 | 2.421989 |
| saelens_sae | 256 | random_comp | 12 | 0.855796 | 1.365400 |
| saelens_sae | 256 | embedding_ase | 12 | 0.878078 | 1.296982 |
| saelens_sae | 256 | ase_output | 12 | 0.892121 | 1.256472 |
| saelens_sae | 256 | mc | 12 | 1.000000 | 1.000000 |
| saelens_sae | 256 | random_stratified | 12 | 1.066184 | 0.879702 |
| saelens_sae | 256 | embedding_comp | 12 | 1.326274 | 0.568504 |
| saelens_sae | 512 | per_query_rf | 12 | 0.458251 | 4.762048 |
| saelens_sae | 512 | pca_comp | 12 | 0.491365 | 4.141820 |
| saelens_sae | 512 | output_comp | 12 | 0.502042 | 3.967530 |
| saelens_sae | 512 | output_active | 12 | 0.694699 | 2.072079 |
| saelens_sae | 512 | internal | 12 | 0.734509 | 1.853554 |
| saelens_sae | 512 | input_concept_comp | 12 | 0.746210 | 1.795882 |
| saelens_sae | 512 | random_stratified | 12 | 0.830317 | 1.450480 |
| saelens_sae | 512 | random_comp | 12 | 0.891541 | 1.258107 |
| saelens_sae | 512 | mc | 12 | 1.000000 | 1.000000 |
| saelens_sae | 512 | embedding_comp | 12 | 1.062243 | 0.886242 |
| saelens_sae | 512 | embedding_ase | 12 | 1.216720 | 0.675489 |
| saelens_sae | 512 | ase_output | 12 | 2.716195 | 0.135543 |
| saelens_sae | 1024 | output_comp | 12 | 0.466053 | 4.603934 |
| saelens_sae | 1024 | per_query_rf | 12 | 0.515578 | 3.761929 |
| saelens_sae | 1024 | pca_comp | 12 | 0.580570 | 2.966817 |
| saelens_sae | 1024 | input_concept_comp | 12 | 0.632358 | 2.500767 |
| saelens_sae | 1024 | output_active | 12 | 0.711257 | 1.976727 |
| saelens_sae | 1024 | ase_output | 12 | 0.756552 | 1.747121 |
| saelens_sae | 1024 | internal | 12 | 0.766468 | 1.702205 |
| saelens_sae | 1024 | random_comp | 12 | 0.860037 | 1.351966 |
| saelens_sae | 1024 | random_stratified | 12 | 0.939076 | 1.133963 |
| saelens_sae | 1024 | embedding_comp | 12 | 0.957459 | 1.090835 |
| saelens_sae | 1024 | mc | 12 | 1.000000 | 1.000000 |
| saelens_sae | 1024 | embedding_ase | 12 | 1.113543 | 0.806466 |
| saelens_sae | 2048 | input_concept_comp | 12 | 0.729712 | 1.878008 |
| saelens_sae | 2048 | per_query_rf | 12 | 0.814226 | 1.508377 |
| saelens_sae | 2048 | output_comp | 12 | 0.875449 | 1.304784 |
| saelens_sae | 2048 | embedding_comp | 12 | 0.877425 | 1.298912 |
| saelens_sae | 2048 | random_comp | 12 | 0.881149 | 1.287956 |
| saelens_sae | 2048 | output_active | 12 | 0.936800 | 1.139478 |
| saelens_sae | 2048 | random_stratified | 12 | 0.962180 | 1.080158 |
| saelens_sae | 2048 | internal | 12 | 0.970593 | 1.061514 |
| saelens_sae | 2048 | mc | 12 | 1.000000 | 1.000000 |
| saelens_sae | 2048 | pca_comp | 12 | 1.026686 | 0.948690 |
| saelens_sae | 2048 | embedding_ase | 12 | 1.186910 | 0.709846 |
| saelens_sae | 2048 | ase_output | 12 | 1.229282 | 0.661754 |
| sklearn_sdl | 64 | output_comp | 12 | 0.619506 | 2.605609 |
| sklearn_sdl | 64 | random_comp | 12 | 0.773290 | 1.672302 |
| sklearn_sdl | 64 | embedding_comp | 12 | 0.856855 | 1.362026 |
| sklearn_sdl | 64 | internal | 12 | 0.911862 | 1.202657 |
| sklearn_sdl | 64 | input_concept_comp | 12 | 0.935915 | 1.141636 |
| sklearn_sdl | 64 | mc | 12 | 1.000000 | 1.000000 |
| sklearn_sdl | 64 | pca_comp | 12 | 1.217787 | 0.674307 |
| sklearn_sdl | 64 | random_stratified | 12 | 1.267644 | 0.622308 |
| sklearn_sdl | 64 | output_active | 12 | 1.384353 | 0.521803 |
| sklearn_sdl | 64 | embedding_ase | 12 | 1.572999 | 0.404150 |
| sklearn_sdl | 64 | ase_output | 12 | 2.126531 | 0.221135 |
| sklearn_sdl | 64 | per_query_rf | 12 | 2.525931 | 0.156732 |
| sklearn_sdl | 128 | per_query_rf | 12 | 0.396339 | 6.365994 |
| sklearn_sdl | 128 | embedding_comp | 12 | 0.560162 | 3.186935 |
| sklearn_sdl | 128 | internal | 12 | 0.560224 | 3.186229 |
| sklearn_sdl | 128 | output_active | 12 | 0.625117 | 2.559042 |
| sklearn_sdl | 128 | random_comp | 12 | 0.684113 | 2.136704 |
| sklearn_sdl | 128 | input_concept_comp | 12 | 0.793358 | 1.588774 |
| sklearn_sdl | 128 | ase_output | 12 | 0.995881 | 1.008290 |
| sklearn_sdl | 128 | mc | 12 | 1.000000 | 1.000000 |
| sklearn_sdl | 128 | pca_comp | 12 | 1.030615 | 0.941471 |
| sklearn_sdl | 128 | random_stratified | 12 | 1.112667 | 0.807736 |
| sklearn_sdl | 128 | embedding_ase | 12 | 1.485229 | 0.453329 |
| sklearn_sdl | 128 | output_comp | 12 | 1.632768 | 0.375104 |
| sklearn_sdl | 256 | per_query_rf | 12 | 0.554327 | 3.254375 |
| sklearn_sdl | 256 | input_concept_comp | 12 | 0.579423 | 2.978577 |
| sklearn_sdl | 256 | pca_comp | 12 | 0.639175 | 2.447709 |
| sklearn_sdl | 256 | random_stratified | 12 | 0.849027 | 1.387258 |
| sklearn_sdl | 256 | internal | 12 | 0.975854 | 1.050099 |
| sklearn_sdl | 256 | mc | 12 | 1.000000 | 1.000000 |
| sklearn_sdl | 256 | output_comp | 12 | 1.032173 | 0.938631 |
| sklearn_sdl | 256 | output_active | 12 | 1.050353 | 0.906420 |
| sklearn_sdl | 256 | ase_output | 12 | 1.237711 | 0.652772 |
| sklearn_sdl | 256 | random_comp | 12 | 1.286279 | 0.604407 |
| sklearn_sdl | 256 | embedding_ase | 12 | 1.434484 | 0.485969 |
| sklearn_sdl | 256 | embedding_comp | 12 | 1.595372 | 0.392894 |
| sklearn_sdl | 512 | per_query_rf | 12 | 0.667364 | 2.245300 |
| sklearn_sdl | 512 | pca_comp | 12 | 0.863644 | 1.340696 |
| sklearn_sdl | 512 | embedding_comp | 12 | 0.883941 | 1.279832 |
| sklearn_sdl | 512 | internal | 12 | 0.907202 | 1.215044 |
| sklearn_sdl | 512 | random_comp | 12 | 0.987618 | 1.025232 |
| sklearn_sdl | 512 | mc | 12 | 1.000000 | 1.000000 |
| sklearn_sdl | 512 | input_concept_comp | 12 | 1.039222 | 0.925941 |
| sklearn_sdl | 512 | output_comp | 12 | 1.126917 | 0.787438 |
| sklearn_sdl | 512 | output_active | 12 | 1.218497 | 0.673521 |
| sklearn_sdl | 512 | embedding_ase | 12 | 1.402346 | 0.508498 |
| sklearn_sdl | 512 | random_stratified | 12 | 1.695060 | 0.348041 |
| sklearn_sdl | 512 | ase_output | 12 | 2.918656 | 0.117391 |

## Query Distribution

| basis_kind | family | queries | mean_reference_rate | mean_select_rate |
| --- | --- | --- | --- | --- |
| saelens_sae | class_confusion | 2 | 0.001200 | 0.001250 |
| saelens_sae | concept_class_error | 6 | 0.008417 | 0.008646 |
| saelens_sae | confidence_error | 1 | 0.008400 | 0.007750 |
| saelens_sae | output_fp | 3 | 0.003167 | 0.003000 |
| sklearn_sdl | class_confusion | 2 | 0.001200 | 0.001250 |
| sklearn_sdl | concept_class_error | 6 | 0.008417 | 0.008646 |
| sklearn_sdl | confidence_error | 1 | 0.008400 | 0.007750 |
| sklearn_sdl | output_fp | 3 | 0.003167 | 0.003000 |
| spd | class_confusion | 2 | 0.001200 | 0.001250 |
| spd | concept_class_error | 6 | 0.008417 | 0.008646 |
| spd | confidence_error | 1 | 0.008400 | 0.007750 |
| spd | output_fp | 3 | 0.003167 | 0.003000 |

## Frozen Rare Events

| query | family | select_rate | uses_basis | uses_concept | description |
| --- | --- | --- | --- | --- | --- |
| 0 | concept_class_error | 0.009500 | False | True | (image concept: edge_high AND label in {frog} AND model error) |
| 1 | output_fp | 0.005625 | False | False | (prediction in {automobile} AND label not in {automobile} AND confidence > 0.70) |
| 2 | concept_class_error | 0.017250 | False | True | (image concept: bottom_heavy AND label in {cat} AND model error) |
| 3 | class_confusion | 0.001125 | False | False | (label in {frog} AND prediction in {truck}) |
| 4 | output_fp | 0.001250 | False | False | (prediction in {truck} AND label not in {truck} AND confidence > 0.85) |
| 5 | output_fp | 0.002125 | False | False | (prediction in {cat} AND label not in {cat} AND confidence > 0.70) |
| 6 | concept_class_error | 0.005875 | False | True | (image concept: brightness_high AND label in {dog} AND model error) |
| 7 | concept_class_error | 0.004875 | False | True | (image concept: top_heavy AND label in {cat} AND model error) |
| 8 | class_confusion | 0.001375 | False | False | (label in {dog} AND prediction in {airplane}) |
| 9 | concept_class_error | 0.004500 | False | True | (image concept: right_heavy AND label in {ship} AND model error) |
| 10 | concept_class_error | 0.009875 | False | True | (image concept: brightness_high AND label in {cat} AND model error) |
| 11 | confidence_error | 0.007750 | False | False | (label in {deer} AND confidence > 0.70 AND model error) |

## Basis Diagnostics

| basis_kind | seed | dictionary_seed | eval_accuracy | eval_nll | train_seconds | basis_code_density | basis_active_components | basis_reconstruction_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sklearn_sdl | 20260521 | 0 | 0.681333 | 0.950644 | 223.470623 | 0.166667 | 8.000000 | 0.323937 |
| saelens_sae | 20260521 | 0 | 0.681333 | 0.950644 | 223.470623 | 0.166667 | 8.000000 | 0.267840 |
| spd | 20260521 | 0 | 0.681333 | 0.950644 | 223.470623 | 0.997743 | 47.891667 | not reached |

## Risk Ranking

| basis_kind | method | cells | mean_auroc | mean_average_precision | mean_top_decile_lift |
| --- | --- | --- | --- | --- | --- |
| saelens_sae | per_query_rf | 12 | 0.971754 | 0.716577 | 9.389151 |
| saelens_sae | embedding_comp | 12 | 0.958803 | 0.353394 | 8.321385 |
| saelens_sae | output_comp | 12 | 0.896499 | 0.156958 | 6.364641 |
| saelens_sae | pca_comp | 12 | 0.921437 | 0.114265 | 6.882046 |
| saelens_sae | internal | 12 | 0.886074 | 0.054798 | 5.999735 |
| saelens_sae | random_comp | 12 | 0.812727 | 0.050248 | 4.351676 |
| saelens_sae | input_concept_comp | 12 | 0.831395 | 0.037661 | 4.572616 |
| saelens_sae | output_active | 12 | 0.823934 | 0.031971 | 4.522854 |
| sklearn_sdl | per_query_rf | 12 | 0.971754 | 0.716577 | 9.389151 |
| sklearn_sdl | embedding_comp | 12 | 0.958803 | 0.353394 | 8.321385 |
| sklearn_sdl | output_comp | 12 | 0.896499 | 0.156958 | 6.364641 |
| sklearn_sdl | pca_comp | 12 | 0.921437 | 0.114265 | 6.882046 |
| sklearn_sdl | internal | 12 | 0.871422 | 0.075736 | 5.383788 |
| sklearn_sdl | random_comp | 12 | 0.812727 | 0.050248 | 4.351676 |
| sklearn_sdl | input_concept_comp | 12 | 0.831395 | 0.037661 | 4.572616 |
| sklearn_sdl | output_active | 12 | 0.823934 | 0.031971 | 4.522854 |
| spd | per_query_rf | 12 | 0.971754 | 0.716577 | 9.389151 |
| spd | embedding_comp | 12 | 0.958803 | 0.353394 | 8.321385 |
| spd | output_comp | 12 | 0.896499 | 0.156958 | 6.364641 |
| spd | pca_comp | 12 | 0.921437 | 0.114265 | 6.882046 |
| spd | internal | 12 | 0.903125 | 0.078068 | 6.959438 |
| spd | random_comp | 12 | 0.812727 | 0.050248 | 4.351676 |
| spd | input_concept_comp | 12 | 0.831395 | 0.037661 | 4.572616 |
| spd | output_active | 12 | 0.823934 | 0.031971 | 4.522854 |

## Configuration

```json
{
  "results_dir": "external_tool_auditing/results/paper_independent_budget_sweep",
  "basis_kinds": [
    "sklearn_sdl",
    "saelens_sae",
    "spd"
  ],
  "query_policy": "independent_external",
  "seeds": [
    20260521
  ],
  "dictionary_seeds": [
    0
  ],
  "train_n": 6000,
  "epochs": 1,
  "concept_calib_n": 2500,
  "build_n": 6000,
  "select_n": 8000,
  "ref_n": 10000,
  "risk_eval_n": 6000,
  "intervention_n": 5000,
  "pool_n": 3072,
  "query_count": 12,
  "max_candidates": 1800,
  "rate_min": 0.001,
  "rate_max": 0.04,
  "query_mode": "external",
  "dictionary_components": 48,
  "dictionary_alpha": 0.35,
  "budgets": [
    64,
    128,
    256,
    512,
    1024,
    2048
  ],
  "reps": 3,
  "torch_threads": 6,
  "torch_inter_op_threads": 1,
  "sklearn_alpha": 0.35,
  "sklearn_max_iter": 350,
  "sae_l1": 0.03,
  "sae_steps": 180,
  "sae_batch_size": 256,
  "sae_lr": 0.001,
  "spd_importance_coeff": 0.03,
  "spd_steps": 120,
  "spd_batch_size": 256,
  "spd_lr": 0.003
}
```