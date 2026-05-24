from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--output",
        default="cifar10_external_tool_rate_binned_rmse.csv",
        help="Output CSV filename, relative to --results-dir unless absolute.",
    )
    return parser.parse_args()


def query_key(df: pd.DataFrame) -> pd.Series:
    parts = [
        df["seed"].astype(str),
        df["dictionary_seed"].astype(str),
        df["query_signature"].astype(str),
    ]
    return parts[0] + "|" + parts[1] + "|" + parts[2]


def rate_bin(rate: float) -> str:
    if rate <= 0.003:
        return "<=0.003"
    if rate <= 0.008:
        return "0.003-0.008"
    return ">0.008"


def rmse(series: pd.Series) -> float:
    return float(np.sqrt(np.mean(series)))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = results_dir / output_path

    runs = pd.read_csv(results_dir / "cifar10_external_tool_runs.csv")
    runs["query_key"] = query_key(runs)
    bins = (
        runs.groupby("query_key")["reference_rate"]
        .mean()
        .map(rate_bin)
        .rename("rate_bin")
        .reset_index()
    )

    mc_cells = (
        runs[runs["method"] == "mc"]
        .groupby(["budget", "query_key"], as_index=False)
        .agg(squared_error=("squared_error", "mean"))
        .merge(bins, on="query_key", how="left")
    )
    internal_cells = (
        runs[runs["method"] == "internal"]
        .groupby(["basis_kind", "budget", "query_key"], as_index=False)
        .agg(squared_error=("squared_error", "mean"))
        .merge(bins, on="query_key", how="left")
    )

    basis_names = {
        "saelens_sae": "SAELens SAE",
        "sklearn_sdl": "sklearn sparse DL",
        "spd": "SPD",
    }
    rows: list[dict[str, object]] = []
    for budget in [128, 512, 2048]:
        for bin_name in ["<=0.003", "0.003-0.008", ">0.008"]:
            row: dict[str, object] = {"budget": budget, "rate_bin": bin_name}
            mc_sub = mc_cells[(mc_cells["budget"] == budget) & (mc_cells["rate_bin"] == bin_name)]
            row["MC"] = rmse(mc_sub["squared_error"])
            for basis_kind, label in basis_names.items():
                sub = internal_cells[
                    (internal_cells["budget"] == budget)
                    & (internal_cells["rate_bin"] == bin_name)
                    & (internal_cells["basis_kind"] == basis_kind)
                ]
                row[label] = rmse(sub["squared_error"])
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
