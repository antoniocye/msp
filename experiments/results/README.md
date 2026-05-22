# Result Artifacts

The `stronger/cifar10_*` files are the archived result package used by the main report.

| File pattern | Contents |
| --- | --- |
| `cifar10_sparse_concept_summary.csv` | aggregate RMSE ratios versus Monte Carlo |
| `cifar10_sparse_concept_ci.csv` | bootstrap intervals for RMSE ratios |
| `cifar10_sparse_concept_break_even.csv` | amortized build-cost comparison |
| `cifar10_sparse_concept_family_summary.csv` | query-family analysis |
| `cifar10_sparse_concept_dictionary.csv` | sparse dictionary diagnostics |
| `cifar10_sparse_concept_interventions.csv` | activation intervention checks |
| `cifar10_sparse_concept_*png` | figures used by the report |
| `cifar10_sparse_concept_results.json` | run configuration and row count |

Fresh decisive-benchmark runs also write `sparse_concept_verdict.csv` and include the same verdict rows in `sparse_concept_results.json`.
