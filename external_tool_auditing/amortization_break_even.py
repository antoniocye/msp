from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_BUILD_LABELS = {
    "internal": 6000,
    "output_comp": 6000,
    "input_concept_comp": 6000,
    "embedding_comp": 6000,
    "pca_comp": 6000,
    "random_comp": 6000,
    "per_query_rf": 6000,
    "mc": 0,
    "random_stratified": 0,
    "output_active": 0,
    "ase_output": 0,
    "embedding_ase": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--finite-population-n", type=float, default=10000.0)
    p.add_argument("--max-k", type=int, default=1_000_000)
    p.add_argument("--n-boot", type=int, default=1000)
    return p.parse_args()


def mc_rmse(event_rates: pd.Series, n: float, finite_population_n: float) -> float:
    if n <= 0:
        return math.nan
    if finite_population_n > 0:
        fpc = max((finite_population_n - n) / (finite_population_n - 1.0), 0.0)
    else:
        fpc = 1.0
    return math.sqrt(float((event_rates * (1.0 - event_rates) / n * fpc).mean()))


def break_even_k(method_rmse: float, event_rates: pd.Series, budget: int, build_labels: int, finite_population_n: float, max_k: int) -> int | None:
    if build_labels <= 0:
        baseline = mc_rmse(event_rates, budget, finite_population_n)
        return 1 if method_rmse < baseline else None
    same_budget_baseline = mc_rmse(event_rates, budget, finite_population_n)
    if method_rmse >= same_budget_baseline:
        return None
    lo, hi = 1, 1
    while hi < max_k and method_rmse >= mc_rmse(event_rates, budget + build_labels / hi, finite_population_n):
        hi *= 2
    if hi >= max_k and method_rmse >= mc_rmse(event_rates, budget + build_labels / hi, finite_population_n):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if method_rmse < mc_rmse(event_rates, budget + build_labels / mid, finite_population_n):
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    runs = pd.read_csv(results_dir / "cifar10_external_tool_runs.csv")
    observed_k = int(runs["query"].nunique())

    df = runs.copy()
    df["basis_label"] = df["basis_kind"]
    df.loc[df["method"] != "internal", "basis_label"] = "global"
    cell = (
        df.groupby(["basis_label", "method", "budget", "query"], as_index=False)
        .agg(squared_error=("squared_error", "mean"), event_rate=("reference_rate", "mean"))
    )

    rows: list[dict[str, object]] = []
    for (basis, method, budget), sub in cell.groupby(["basis_label", "method", "budget"]):
        build_labels = DEFAULT_BUILD_LABELS.get(str(method))
        if build_labels is None:
            continue
        method_rmse = math.sqrt(float(sub["squared_error"].mean()))
        k = break_even_k(method_rmse, sub["event_rate"], int(budget), build_labels, args.finite_population_n, args.max_k)
        labels_at_break_even = math.nan if k is None else float(budget) + build_labels / k
        labels_at_observed_k = float(budget) + build_labels / observed_k if build_labels else float(budget)
        mc_at_observed_k = mc_rmse(sub["event_rate"], labels_at_observed_k, args.finite_population_n)
        mc_same_budget = mc_rmse(sub["event_rate"], float(budget), args.finite_population_n)
        rows.append(
            {
                "basis": basis,
                "method": method,
                "budget": int(budget),
                "observed_queries": observed_k,
                "build_labels": build_labels,
                "method_rmse": method_rmse,
                "same_budget_mc_rmse_theory": mc_same_budget,
                "ratio_vs_same_budget_mc_theory": method_rmse / mc_same_budget if mc_same_budget else math.nan,
                "ratio_at_observed_queries": method_rmse / mc_at_observed_k if mc_at_observed_k else math.nan,
                "break_even_queries_extrapolated": k,
                "labels_per_query_at_break_even": labels_at_break_even,
            }
        )

    out = pd.DataFrame(rows).sort_values(["method", "basis", "budget"])
    out_path = results_dir / "cifar10_external_tool_amortization_break_even.csv"
    out.to_csv(out_path, index=False)
    boot_rows = []
    internal = runs[runs["method"] == "internal"].copy()
    rng_seed = 20260523
    try:
        import numpy as np

        rng = np.random.default_rng(rng_seed)
        for (basis, budget), sub in internal.groupby(["basis_kind", "budget"]):
            query_ids = sorted(sub["query"].unique())
            by_query = {q: qdf for q, qdf in sub.groupby("query")}
            vals = []
            for _ in range(args.n_boot):
                sampled_queries = rng.choice(query_ids, size=len(query_ids), replace=True)
                se_parts = []
                p_parts = []
                for q in sampled_queries:
                    qdf = by_query[int(q)]
                    reps = sorted(qdf["rep"].unique())
                    sampled_reps = rng.choice(reps, size=len(reps), replace=True)
                    rep_means = [
                        float(qdf.loc[qdf["rep"] == int(rep), "squared_error"].mean())
                        for rep in sampled_reps
                    ]
                    se_parts.append(float(pd.Series(rep_means).mean()))
                    p_parts.append(float(qdf["reference_rate"].iloc[0]))
                method_rmse = math.sqrt(float(pd.Series(se_parts).mean()))
                k = break_even_k(method_rmse, pd.Series(p_parts), int(budget), DEFAULT_BUILD_LABELS["internal"], args.finite_population_n, args.max_k)
                vals.append(float("nan") if k is None else float(k))
            finite = pd.Series(vals, dtype=float).dropna()
            point = out[(out["method"] == "internal") & (out["basis"] == basis) & (out["budget"] == budget)]["break_even_queries_extrapolated"].iloc[0]
            boot_rows.append(
                {
                    "basis": basis,
                    "method": "internal",
                    "budget": int(budget),
                    "point_break_even_queries": point,
                    "finite_boot_fraction": float(len(finite) / len(vals)),
                    "break_even_q05": float(finite.quantile(0.05)) if len(finite) else math.nan,
                    "break_even_q50": float(finite.quantile(0.50)) if len(finite) else math.nan,
                    "break_even_q95": float(finite.quantile(0.95)) if len(finite) else math.nan,
                }
            )
    except ImportError:
        boot_rows = []
    if boot_rows:
        boot_path = results_dir / "cifar10_external_tool_amortization_break_even_bootstrap.csv"
        pd.DataFrame(boot_rows).sort_values(["basis", "budget"]).to_csv(boot_path, index=False)
        print(boot_path)
    print(out_path)


if __name__ == "__main__":
    main()
