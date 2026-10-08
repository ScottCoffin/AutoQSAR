from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GENRA_GLOBAL_R2 = 0.61
GENRA_GLOBAL_RMSE = 0.58
GENRA_SOURCE = "Helman et al. 2019, Comput Toxicol, doi:10.1016/j.comtox.2019.100097"


def build_comparison(metrics_path: Path, out_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path, low_memory=False)
    for column in ["test_r2", "test_rmse", "test_mae", "cv_r2", "cv_rmse"]:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    rows = metrics.loc[metrics["test_r2"].notna() & metrics["test_rmse"].notna()].copy()
    rows["genra_global_r2"] = GENRA_GLOBAL_R2
    rows["genra_global_rmse"] = GENRA_GLOBAL_RMSE
    rows["delta_r2_vs_genra_global"] = rows["test_r2"] - GENRA_GLOBAL_R2
    rows["delta_rmse_vs_genra_global"] = rows["test_rmse"] - GENRA_GLOBAL_RMSE
    keep = [
        "dataset",
        "model",
        "workflow",
        "ensemble_method",
        "n_molecules",
        "n_train",
        "n_test",
        "split_strategy",
        "target_transform",
        "test_r2",
        "test_rmse",
        "test_mae",
        "cv_r2",
        "cv_rmse",
        "genra_global_r2",
        "genra_global_rmse",
        "delta_r2_vs_genra_global",
        "delta_rmse_vs_genra_global",
    ]
    available = [column for column in keep if column in rows.columns]
    comparison = rows[available].sort_values(["test_rmse", "test_r2"], ascending=[True, False]).reset_index(drop=True)
    comparison.to_csv(out_dir / "genra_comparison.csv", index=False)
    return comparison


def write_markdown(comparison: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Acute LD50 QSARena vs GenRA paper baseline",
        "",
        f"Paper baseline: {GENRA_SOURCE}.",
        "GenRA global R2 = 0.61, RMSE = 0.58; Monte Carlo CV R2 range 0.47-0.62; local-domain R2 up to 0.91.",
        "",
        "QSARena values use the package benchmark split/protocol, not the paper's exact GenRA setup.",
        "",
    ]
    if comparison.empty:
        lines.append("No QSARena rows with both test_r2 and test_rmse were found.")
    else:
        best_rmse = comparison.iloc[0]
        best_r2 = comparison.sort_values(["test_r2", "test_rmse"], ascending=[False, True]).iloc[0]
        lines.extend(
            [
                "## Best QSARena rows",
                "",
                (
                    f"Best RMSE: {best_rmse['model']} test_R2={best_rmse['test_r2']:.3f}, "
                    f"test_RMSE={best_rmse['test_rmse']:.3f}, "
                    f"delta_R2_vs_GenRA={best_rmse['delta_r2_vs_genra_global']:.3f}, "
                    f"delta_RMSE_vs_GenRA={best_rmse['delta_rmse_vs_genra_global']:.3f}."
                ),
                (
                    f"Best R2: {best_r2['model']} test_R2={best_r2['test_r2']:.3f}, "
                    f"test_RMSE={best_r2['test_rmse']:.3f}, "
                    f"delta_R2_vs_GenRA={best_r2['delta_r2_vs_genra_global']:.3f}, "
                    f"delta_RMSE_vs_GenRA={best_r2['delta_rmse_vs_genra_global']:.3f}."
                ),
                "",
                "## Ranked by test RMSE",
                "",
                comparison.head(30).to_markdown(index=False),
            ]
        )
    (out_dir / "genra_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="full_dataset_qsar_ready_ld50lm")
    args = parser.parse_args()
    metrics_path = args.run_dir / args.dataset_name / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    comparison = build_comparison(metrics_path, args.run_dir)
    write_markdown(comparison, args.run_dir)
    print(f"wrote {args.run_dir / 'genra_comparison.csv'}")
    print(f"wrote {args.run_dir / 'genra_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
