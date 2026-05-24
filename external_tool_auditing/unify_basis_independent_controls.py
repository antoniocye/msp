from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASIS_SPECIFIC_METHODS = {"internal"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    return p.parse_args()


def query_key(df: pd.DataFrame) -> pd.Series:
    parts = [
        df["seed"].astype(str),
        df["dictionary_seed"].astype(str),
        df["query_signature"].astype(str),
    ]
    return parts[0] + "|" + parts[1] + "|" + parts[2]


def global_control_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    controls = run_df[~run_df["method"].isin(BASIS_SPECIFIC_METHODS)].copy()
    controls["query_key"] = query_key(controls)
    cell = (
        controls.groupby(["budget", "method", "query_key"], as_index=False)
        .agg(
            squared_error=("squared_error", "mean"),
            abs_error=("abs_error", "mean"),
            reference_rate=("reference_rate", "mean"),
        )
    )
    out = (
        cell.groupby(["method", "budget"], as_index=False)
        .agg(
            cells=("squared_error", "count"),
            model_queries=("query_key", "nunique"),
            mean_reference_rate=("reference_rate", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
            mae=("abs_error", "mean"),
        )
    )
    mc = out[out["method"] == "mc"][["budget", "rmse"]].rename(columns={"rmse": "mc_rmse"})
    out = out.merge(mc, on="budget", how="left")
    out["rmse_ratio_vs_mc"] = out["rmse"] / out["mc_rmse"]
    out["effective_mc_multiplier"] = 1.0 / np.square(out["rmse_ratio_vs_mc"])
    return out.sort_values(["budget", "rmse_ratio_vs_mc"])


def global_mc_cells(run_df: pd.DataFrame) -> pd.DataFrame:
    controls = run_df[run_df["method"] == "mc"].copy()
    controls["query_key"] = query_key(controls)
    return (
        controls.groupby(["budget", "query_key"], as_index=False)
        .agg(squared_error=("squared_error", "mean"))
        .rename(columns={"squared_error": "mc_squared_error"})
    )


def global_control_ci(run_df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260523)
    controls = run_df[~run_df["method"].isin(BASIS_SPECIFIC_METHODS)].copy()
    controls["query_key"] = query_key(controls)
    cell = (
        controls.groupby(["budget", "query_key", "method"], as_index=False)
        .agg(squared_error=("squared_error", "mean"))
    )
    rows: list[dict[str, object]] = []
    for budget, group in cell.groupby("budget"):
        ids = np.array(sorted(group["query_key"].unique()))
        by_method = {
            method: sub.set_index("query_key")["squared_error"].reindex(ids).to_numpy()
            for method, sub in group.groupby("method")
        }
        if "mc" not in by_method:
            raise RuntimeError(f"missing mc rows for budget {budget}")
        mc = by_method["mc"]
        for method, errors in by_method.items():
            vals = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                idx = rng.integers(0, len(ids), size=len(ids))
                denom = float(np.sqrt(np.mean(mc[idx])))
                numer = float(np.sqrt(np.mean(errors[idx])))
                vals[i] = numer / denom if denom > 0.0 else np.nan
            rmse = float(np.sqrt(np.mean(errors)))
            mc_rmse = float(np.sqrt(np.mean(mc)))
            rows.append(
                {
                    "budget": int(budget),
                    "method": method,
                    "model_queries": len(ids),
                    "rmse": rmse,
                    "mc_rmse": mc_rmse,
                    "rmse_ratio_vs_mc": rmse / mc_rmse,
                    "rmse_ratio_ci_low": float(np.nanquantile(vals, 0.025)),
                    "rmse_ratio_ci_high": float(np.nanquantile(vals, 0.975)),
                }
            )
    return pd.DataFrame(rows).sort_values(["budget", "rmse_ratio_vs_mc"])


def internal_summary_against_global_mc(run_df: pd.DataFrame, mc_cells: pd.DataFrame) -> pd.DataFrame:
    internal = run_df[run_df["method"].isin(BASIS_SPECIFIC_METHODS)].copy()
    internal["query_key"] = query_key(internal)
    cell = (
        internal.groupby(["basis_kind", "budget", "method", "query_key"], as_index=False)
        .agg(
            squared_error=("squared_error", "mean"),
            abs_error=("abs_error", "mean"),
            reference_rate=("reference_rate", "mean"),
        )
        .merge(mc_cells, on=["budget", "query_key"], how="left")
    )
    if cell["mc_squared_error"].isna().any():
        raise RuntimeError("missing global MC cells for internal rows")
    out = (
        cell.groupby(["basis_kind", "method", "budget"], as_index=False)
        .agg(
            cells=("squared_error", "count"),
            model_queries=("query_key", "nunique"),
            mean_reference_rate=("reference_rate", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
            mc_rmse=("mc_squared_error", lambda s: float(np.sqrt(np.mean(s)))),
            mae=("abs_error", "mean"),
        )
    )
    out["rmse_ratio_vs_mc"] = out["rmse"] / out["mc_rmse"]
    out["effective_mc_multiplier"] = 1.0 / np.square(out["rmse_ratio_vs_mc"])
    return out.sort_values(["basis_kind", "budget", "rmse_ratio_vs_mc"])


def internal_ci_against_global_mc(run_df: pd.DataFrame, mc_cells: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260524)
    internal = run_df[run_df["method"].isin(BASIS_SPECIFIC_METHODS)].copy()
    internal["query_key"] = query_key(internal)
    cell = (
        internal.groupby(["basis_kind", "budget", "query_key", "method"], as_index=False)
        .agg(squared_error=("squared_error", "mean"))
        .merge(mc_cells, on=["budget", "query_key"], how="left")
    )
    if cell["mc_squared_error"].isna().any():
        raise RuntimeError("missing global MC cells for internal rows")
    rows: list[dict[str, object]] = []
    for (basis_kind, budget), group in cell.groupby(["basis_kind", "budget"]):
        ids = np.array(sorted(group["query_key"].unique()))
        mc = group.drop_duplicates("query_key").set_index("query_key")["mc_squared_error"].reindex(ids).to_numpy()
        by_method = {
            method: sub.set_index("query_key")["squared_error"].reindex(ids).to_numpy()
            for method, sub in group.groupby("method")
        }
        for method, errors in by_method.items():
            vals = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                idx = rng.integers(0, len(ids), size=len(ids))
                denom = float(np.sqrt(np.mean(mc[idx])))
                numer = float(np.sqrt(np.mean(errors[idx])))
                vals[i] = numer / denom if denom > 0.0 else np.nan
            rmse = float(np.sqrt(np.mean(errors)))
            mc_rmse = float(np.sqrt(np.mean(mc)))
            rows.append(
                {
                    "basis_kind": basis_kind,
                    "budget": int(budget),
                    "method": method,
                    "model_queries": len(ids),
                    "rmse": rmse,
                    "mc_rmse": mc_rmse,
                    "rmse_ratio_vs_mc": rmse / mc_rmse,
                    "rmse_ratio_ci_low": float(np.nanquantile(vals, 0.025)),
                    "rmse_ratio_ci_high": float(np.nanquantile(vals, 0.975)),
                }
            )
    return pd.DataFrame(rows).sort_values(["basis_kind", "budget", "rmse_ratio_vs_mc"])


def copy_controls_to_basis(global_df: pd.DataFrame, basis_kinds: list[str]) -> pd.DataFrame:
    rows = []
    for basis_kind in basis_kinds:
        sub = global_df.copy()
        sub.insert(0, "basis_kind", basis_kind)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(fmt(row[col]) for col in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep, *rows])


def write_report(results_dir: Path, unified_ci: pd.DataFrame, global_ci: pd.DataFrame, control_methods: list[str]) -> Path:
    internal = unified_ci[unified_ci["method"] == "internal"].copy()
    lines = [
        "# Unified Basis-Independent Controls",
        "",
        "This report is post-processed from the completed raw run. No SAE/SDL/SPD internal fits or internal estimates were rerun.",
        "",
        "All non-`internal` methods are treated as basis-independent controls for this independent-query run. They are aggregated once by frozen query signature across the existing repeated control estimates, then copied into each basis panel.",
        "",
        "The `internal` estimates are not rerun. Their RMSE numerators come from the original per-tool internal rows, but their ratios are recomputed against the same unified MC denominator used by the controls.",
        "",
        "Unified control methods: `" + "`, `".join(control_methods) + "`.",
        "",
        "## Internal Rows",
        "",
        markdown_table(internal[["basis_kind", "budget", "model_queries", "rmse_ratio_vs_mc", "rmse_ratio_ci_low", "rmse_ratio_ci_high"]]),
        "",
        "## Global Controls",
        "",
        markdown_table(global_ci[["budget", "method", "model_queries", "rmse_ratio_vs_mc", "rmse_ratio_ci_low", "rmse_ratio_ci_high"]]),
        "",
    ]
    out = results_dir / "cifar10_external_tool_unified_controls_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    run_df = pd.read_csv(results_dir / "cifar10_external_tool_runs.csv")
    raw_ci = pd.read_csv(results_dir / "cifar10_external_tool_ci.csv")
    queries_path = results_dir / "cifar10_external_tool_queries.csv"
    if queries_path.exists():
        queries = pd.read_csv(queries_path)
        if "uses_basis" in queries and queries["uses_basis"].astype(str).str.lower().eq("true").any():
            raise RuntimeError("cannot safely unify controls for a run with basis-dependent rare-event queries")

    basis_kinds = list(raw_ci["basis_kind"].drop_duplicates())
    control_methods = sorted(m for m in run_df["method"].unique() if m not in BASIS_SPECIFIC_METHODS)

    mc_cells = global_mc_cells(run_df)
    global_ci = global_control_ci(run_df, args.n_boot)
    global_summary = global_control_summary(run_df)
    internal_ci = internal_ci_against_global_mc(run_df, mc_cells, args.n_boot)
    internal_summary = internal_summary_against_global_mc(run_df, mc_cells)

    unified_ci = pd.concat([internal_ci, copy_controls_to_basis(global_ci, basis_kinds)], ignore_index=True)
    unified_ci = unified_ci.sort_values(["basis_kind", "budget", "rmse_ratio_vs_mc"])

    unified_summary = pd.concat([internal_summary, copy_controls_to_basis(global_summary, basis_kinds)], ignore_index=True)
    unified_summary = unified_summary.sort_values(["basis_kind", "budget", "rmse_ratio_vs_mc"])

    ci_out = results_dir / "cifar10_external_tool_ci_unified_controls.csv"
    summary_out = results_dir / "cifar10_external_tool_summary_unified_controls.csv"
    global_ci_out = results_dir / "cifar10_external_tool_global_controls_ci.csv"
    global_summary_out = results_dir / "cifar10_external_tool_global_controls_summary.csv"
    unified_ci.to_csv(ci_out, index=False)
    unified_summary.to_csv(summary_out, index=False)
    global_ci.to_csv(global_ci_out, index=False)
    global_summary.to_csv(global_summary_out, index=False)
    report_out = write_report(results_dir, unified_ci, global_ci, control_methods)

    print(ci_out)
    print(summary_out)
    print(global_ci_out)
    print(report_out)


if __name__ == "__main__":
    main()
