# Unified Basis-Independent Controls

This report is post-processed from the completed raw run. No SAE/SDL/SPD internal fits or internal estimates were rerun.

All non-`internal` methods are treated as basis-independent controls for this independent-query run. They are aggregated once by frozen query signature across the existing repeated control estimates, then copied into each basis panel.

The `internal` estimates are not rerun. Their RMSE numerators come from the original per-tool internal rows, but their ratios are recomputed against the same unified MC denominator used by the controls.

Unified control methods: `ase_output`, `embedding_ase`, `embedding_comp`, `input_concept_comp`, `mc`, `output_active`, `output_comp`, `pca_comp`, `per_query_rf`, `random_comp`, `random_stratified`.

## Internal Rows

| basis_kind | budget | model_queries | rmse_ratio_vs_mc | rmse_ratio_ci_low | rmse_ratio_ci_high |
| --- | --- | --- | --- | --- | --- |
| saelens_sae | 64 | 12 | 1.126506 | 0.567339 | 1.494760 |
| saelens_sae | 128 | 12 | 0.544102 | 0.440037 | 0.638077 |
| saelens_sae | 256 | 12 | 0.700793 | 0.428660 | 0.950256 |
| saelens_sae | 512 | 12 | 0.782217 | 0.693999 | 0.856344 |
| saelens_sae | 1024 | 12 | 0.901551 | 0.652129 | 1.144702 |
| saelens_sae | 2048 | 12 | 0.970593 | 0.689829 | 1.232159 |
| sklearn_sdl | 64 | 12 | 0.856625 | 0.630785 | 0.996428 |
| sklearn_sdl | 128 | 12 | 0.509494 | 0.341193 | 0.784474 |
| sklearn_sdl | 256 | 12 | 0.830370 | 0.516696 | 1.038783 |
| sklearn_sdl | 512 | 12 | 0.746171 | 0.499991 | 1.086461 |
| sklearn_sdl | 1024 | 12 | 0.741855 | 0.517534 | 0.957860 |
| sklearn_sdl | 2048 | 12 | 0.813476 | 0.574164 | 1.024215 |
| spd | 64 | 12 | 1.259223 | 0.658740 | 1.749313 |
| spd | 128 | 12 | 0.569291 | 0.479800 | 0.764160 |
| spd | 256 | 12 | 0.595475 | 0.478499 | 0.696192 |
| spd | 512 | 12 | 0.911086 | 0.663971 | 1.207606 |
| spd | 1024 | 12 | 0.841254 | 0.551781 | 1.089608 |
| spd | 2048 | 12 | 0.863112 | 0.670093 | 1.033947 |

## Global Controls

| budget | method | model_queries | rmse_ratio_vs_mc | rmse_ratio_ci_low | rmse_ratio_ci_high |
| --- | --- | --- | --- | --- | --- |
| 64 | input_concept_comp | 12 | 0.788549 | 0.542746 | 1.003744 |
| 64 | output_comp | 12 | 0.809357 | 0.508259 | 1.104604 |
| 64 | pca_comp | 12 | 0.849359 | 0.488921 | 1.148731 |
| 64 | random_comp | 12 | 0.944969 | 0.734744 | 1.171861 |
| 64 | mc | 12 | 1.000000 | 1.000000 | 1.000000 |
| 64 | output_active | 12 | 1.063872 | 0.648591 | 1.396305 |
| 64 | per_query_rf | 12 | 1.477587 | 0.549553 | 2.733760 |
| 64 | embedding_ase | 12 | 1.527581 | 1.176611 | 1.835080 |
| 64 | random_stratified | 12 | 1.545202 | 1.168576 | 1.825515 |
| 64 | embedding_comp | 12 | 1.619535 | 0.367695 | 2.534087 |
| 64 | ase_output | 12 | 2.139216 | 1.185416 | 2.946387 |
| 128 | per_query_rf | 12 | 0.498813 | 0.417147 | 0.601641 |
| 128 | input_concept_comp | 12 | 0.577933 | 0.329077 | 1.007351 |
| 128 | output_active | 12 | 0.632460 | 0.581589 | 0.730932 |
| 128 | random_comp | 12 | 0.663228 | 0.504001 | 0.958209 |
| 128 | pca_comp | 12 | 0.680906 | 0.420553 | 1.167686 |
| 128 | embedding_comp | 12 | 0.853139 | 0.397653 | 1.497361 |
| 128 | ase_output | 12 | 0.968350 | 0.699202 | 1.435818 |
| 128 | random_stratified | 12 | 0.971023 | 0.839528 | 1.322700 |
| 128 | mc | 12 | 1.000000 | 1.000000 | 1.000000 |
| 128 | output_comp | 12 | 1.041530 | 0.675155 | 1.174438 |
| 128 | embedding_ase | 12 | 1.250511 | 0.913352 | 1.799412 |
| 256 | per_query_rf | 12 | 0.505365 | 0.415642 | 0.586775 |
| 256 | pca_comp | 12 | 0.533222 | 0.454097 | 0.618665 |
| 256 | input_concept_comp | 12 | 0.593868 | 0.480280 | 0.705078 |
| 256 | output_comp | 12 | 0.752974 | 0.495397 | 0.923796 |
| 256 | output_active | 12 | 0.877250 | 0.631426 | 1.084366 |
| 256 | random_comp | 12 | 0.953099 | 0.635424 | 1.232308 |
| 256 | random_stratified | 12 | 0.975383 | 0.798213 | 1.113768 |
| 256 | mc | 12 | 1.000000 | 1.000000 | 1.000000 |
| 256 | embedding_ase | 12 | 1.018676 | 0.857734 | 1.175302 |
| 256 | ase_output | 12 | 1.057226 | 0.903383 | 1.157242 |
| 256 | embedding_comp | 12 | 1.207527 | 0.480395 | 1.799595 |
| 512 | per_query_rf | 12 | 0.520766 | 0.434878 | 0.625219 |
| 512 | pca_comp | 12 | 0.616887 | 0.513465 | 0.742167 |
| 512 | output_comp | 12 | 0.835395 | 0.680801 | 0.909435 |
| 512 | input_concept_comp | 12 | 0.843876 | 0.673604 | 1.100297 |
| 512 | random_comp | 12 | 0.893876 | 0.746251 | 1.059069 |
| 512 | mc | 12 | 1.000000 | 1.000000 | 1.000000 |
| 512 | embedding_comp | 12 | 1.031680 | 0.695157 | 1.213356 |
| 512 | output_active | 12 | 1.032225 | 0.758085 | 1.316501 |
| 512 | random_stratified | 12 | 1.169996 | 1.057914 | 1.329589 |
| 512 | embedding_ase | 12 | 1.618854 | 1.132189 | 1.959364 |
| 512 | ase_output | 12 | 2.279163 | 1.088065 | 3.447739 |
| 1024 | per_query_rf | 12 | 0.575241 | 0.419502 | 0.708329 |
| 1024 | output_comp | 12 | 0.642778 | 0.490060 | 0.789947 |
| 1024 | input_concept_comp | 12 | 0.728451 | 0.528300 | 0.941015 |
| 1024 | pca_comp | 12 | 0.785091 | 0.556569 | 1.005958 |
| 1024 | output_active | 12 | 0.813116 | 0.631708 | 1.003598 |
| 1024 | random_comp | 12 | 0.936732 | 0.739362 | 1.130658 |
| 1024 | mc | 12 | 1.000000 | 1.000000 | 1.000000 |
| 1024 | embedding_comp | 12 | 1.005567 | 0.686037 | 1.319679 |
| 1024 | random_stratified | 12 | 1.082788 | 0.838586 | 1.329238 |
| 1024 | ase_output | 12 | 1.242433 | 0.733396 | 1.777821 |
| 1024 | embedding_ase | 12 | 1.282637 | 0.933501 | 1.632154 |
| 2048 | per_query_rf | 12 | 0.820359 | 0.619030 | 0.998850 |
| 2048 | input_concept_comp | 12 | 0.846732 | 0.666403 | 1.050985 |
| 2048 | output_comp | 12 | 0.875197 | 0.687659 | 1.050158 |
| 2048 | output_active | 12 | 0.892816 | 0.704813 | 1.076618 |
| 2048 | mc | 12 | 1.000000 | 1.000000 | 1.000000 |
| 2048 | pca_comp | 12 | 1.016305 | 0.680176 | 1.370054 |
| 2048 | random_comp | 12 | 1.031790 | 0.787586 | 1.285249 |
| 2048 | embedding_comp | 12 | 1.041544 | 0.741723 | 1.355022 |
| 2048 | random_stratified | 12 | 1.103879 | 0.868367 | 1.329886 |
| 2048 | ase_output | 12 | 1.197484 | 0.959613 | 1.442297 |
| 2048 | embedding_ase | 12 | 1.251658 | 0.968195 | 1.553580 |
