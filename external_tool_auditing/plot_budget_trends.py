from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--ci-file", default="cifar10_external_tool_ci.csv")
    p.add_argument("--output", default="cifar10_external_tool_budget_trends.png")
    p.add_argument("--methods", nargs="+", default=["internal", "output_comp", "output_active", "pca_comp", "random_comp", "mc"])
    p.add_argument("--metric", choices=["ratio", "rmse"], default="ratio")
    p.add_argument("--single-plot", action="store_true", help="Overlay shared baselines and tool internals in one axis.")
    return p.parse_args()


def display_name(value: str) -> str:
    names = {
        "saelens_sae": "SAELens SAE",
        "sklearn_sdl": "sklearn SDL",
        "spd": "SPD",
        "mc": "MC",
        "output_comp": "output comp",
        "output_active": "output active",
        "pca_comp": "PCA comp",
        "random_comp": "random comp",
        "input_concept_comp": "input concept comp",
        "embedding_comp": "embedding comp",
        "per_query_rf": "per-query RF",
        "random_stratified": "random stratified",
        "ase_output": "ASE output",
        "embedding_ase": "embedding ASE",
    }
    return names.get(value, value)


def collapse_baseline(ci: pd.DataFrame, method: str) -> pd.DataFrame:
    sub = ci[ci["method"] == method].copy()
    if sub.empty:
        return sub
    return (
        sub.groupby("budget", as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_ratio_vs_mc=("rmse_ratio_vs_mc", "mean"),
            rmse_ratio_ci_low=("rmse_ratio_ci_low", "mean"),
            rmse_ratio_ci_high=("rmse_ratio_ci_high", "mean"),
        )
        .sort_values("budget")
    )


def metric_config(metric: str) -> tuple[str, str, str]:
    if metric == "rmse":
        return "rmse", "RMSE vs true event rate", "CIFAR-10 rare-event RMSE against true rates"
    return "rmse_ratio_vs_mc", "RMSE ratio vs MC", "CIFAR-10 external basis budget sweep"


def plot_single_axis(ci: pd.DataFrame, methods: list[str], out: Path, metric: str) -> None:
    y_col, y_label, title = metric_config(metric)
    internal_colors = {
        "saelens_sae": "#d55e00",
        "sklearn_sdl": "#0072b2",
        "spd": "#009e73",
    }
    baseline_styles = {
        "mc": {"color": "#222222", "linestyle": "--", "marker": None, "linewidth": 1.4},
        "output_comp": {"color": "#687385", "linestyle": "-", "marker": "o", "linewidth": 1.4},
        "output_active": {"color": "#8f758f", "linestyle": ":", "marker": "s", "linewidth": 1.4},
        "pca_comp": {"color": "#77866f", "linestyle": "-.", "marker": "^", "linewidth": 1.4},
        "random_comp": {"color": "#9a9a9a", "linestyle": (0, (3, 2)), "marker": "D", "linewidth": 1.2},
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    baseline_methods = [m for m in methods if m != "internal"]
    for method in baseline_methods:
        sub = collapse_baseline(ci, method)
        if sub.empty:
            continue
        style = baseline_styles.get(method, {"color": "#9a9a9a", "linestyle": "--", "marker": "o", "linewidth": 1.1}).copy()
        marker = style.pop("marker")
        ax.plot(
            sub["budget"],
            sub[y_col],
            label=f"baseline: {display_name(method)}",
            marker=marker,
            markersize=4 if marker else 0,
            alpha=0.72,
            zorder=2,
            **style,
        )

    internal = ci[ci["method"] == "internal"].copy()
    for basis_kind, sub in internal.groupby("basis_kind"):
        sub = sub.sort_values("budget")
        color = internal_colors.get(basis_kind, "#b65f1a")
        ax.plot(
            sub["budget"],
            sub[y_col],
            marker="o",
            markersize=6,
            linewidth=3.2,
            label=display_name(basis_kind),
            color=color,
            zorder=4,
        )

    if metric == "ratio":
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.9, alpha=0.65, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(ci["budget"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("label budget")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    ci_path = results_dir / args.ci_file
    ci = pd.read_csv(ci_path)
    methods = [m for m in args.methods if m in set(ci["method"])]
    out = results_dir / args.output
    if args.single_plot:
        plot_single_axis(ci, methods, out, args.metric)
        print(out)
        return

    y_col, y_label, title = metric_config(args.metric)
    basis_kinds = list(ci["basis_kind"].drop_duplicates())
    colors = {
        "internal": "#b65f1a",
        "output_comp": "#355c9a",
        "output_active": "#766fb0",
        "pca_comp": "#7d9a35",
        "random_comp": "#8d8d8d",
        "mc": "#222222",
    }
    fig, axes = plt.subplots(1, len(basis_kinds), figsize=(4.3 * len(basis_kinds), 4.2), sharey=True)
    if len(basis_kinds) == 1:
        axes = [axes]
    for ax, basis_kind in zip(axes, basis_kinds, strict=False):
        sub_basis = ci[ci["basis_kind"] == basis_kind]
        for method in methods:
            sub = sub_basis[sub_basis["method"] == method].sort_values("budget")
            if sub.empty:
                continue
            ax.plot(
                sub["budget"],
                sub[y_col],
                marker="o",
                linewidth=1.8,
                label=method,
                color=colors.get(method),
            )
            if method == "internal" and args.metric == "ratio":
                ax.fill_between(
                    sub["budget"].to_numpy(),
                    sub["rmse_ratio_ci_low"].to_numpy(),
                    sub["rmse_ratio_ci_high"].to_numpy(),
                    color=colors["internal"],
                    alpha=0.16,
                    linewidth=0,
                )
        if args.metric == "ratio":
            ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub_basis["budget"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(basis_kind)
        ax.set_xlabel("label budget")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
