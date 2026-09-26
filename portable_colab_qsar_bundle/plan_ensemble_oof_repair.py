"""Plan the out-of-fold (OOF) ensemble repair without training anything.

Reads a seeded run directory (metrics.csv + predictions.csv per dataset) and reports, for every
candidate ensemble member, where its out-of-fold predictions would come from if the runner were
launched with ``--ensemble-member-selection-split oof``:

    saved_oof      OOF rows already in predictions.csv            (free)
    unimol_cvdata  Uni-Mol's own internal-fold predictions (cv.data) (free, read from disk)
    cpu_refit      K fold refits on the CPU                        (conventional, ChemML, MapLight)
    gpu_refit      K fold refits on the GPU                        (Chemprop; Uni-Mol without cv.data)
    excluded       not a member (fusion output, API-metered TabPFN, no refit path, or GPU refit
                   disallowed by --scope cpu)

Nothing is fitted and no GPU is touched, so it is safe to run on an allocation that is short of
credits. Full models are never retrained by the OOF stage; only fold models are.

    python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py \
        benchmark_results/qsarena_benchmark_oof_ensemble \
        --scope cpu --folds 5 \
        --source-run benchmark_results/autoqsar_benchmark_20260623_153839

Writes ``ensemble_oof_plan.csv`` into the run directory and prints a summary. Exit code 2 means
the plan would retrain on the GPU although ``--scope cpu`` was requested, or that a Uni-Mol member
has no usable cv.data (which would trigger a GPU refit under ``--scope all``); check before launching.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner
except ImportError:  # script-style use from inside the bundle directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner

# Median per-model, per-dataset A100 wall-clock of the full training (chemprop_fixed / canonical
# runs). A fold refit trains on (K-1)/K of the data; these give an order-of-magnitude estimate only.
MEDIAN_FULL_FIT_SECONDS = {"chemprop": 281.0, "unimol": 374.0, "maplight_gnn": 137.0, "chemml": 40.0, "conventional": 7.0}


def member_kind(model_name: str, workflow: str) -> str:
    name = str(model_name).strip()
    lower = name.lower()
    workflow_lower = str(workflow).strip().lower()
    if runner.is_fusion_payload(name, {"workflow": workflow}):
        return "fusion"
    if name.endswith(" GA") or workflow_lower.startswith("tuned"):
        return "ga_tuned"
    if lower.startswith("chemprop"):
        return "chemprop"
    if lower.startswith("uni-mol"):
        return "unimol"
    if lower.startswith("maplight + gnn"):
        return "maplight_gnn"
    if lower.startswith("chemml"):
        return "chemml"
    if lower.startswith("tabpfn"):
        return "tabpfn"
    return "conventional"


def unimol_candidates(model_name: str, metrics: pd.DataFrame, dataset_dir: Path, source_run: Path | None,
                      random_seed: int, unimol_model_size: str) -> list[Path]:
    subdir = (
        Path("unimol_v1") / f"seed_{random_seed}"
        if model_name == "Uni-Mol V1"
        else Path("unimol_v2") / f"{unimol_model_size}_seed_{random_seed}"
    )
    candidates: list[Path] = []
    if "unimol_model_dir" in metrics.columns:
        rows = metrics.loc[metrics["model"].astype(str) == model_name, "unimol_model_dir"].dropna().astype(str)
        candidates.extend(Path(value) for value in rows if value.strip() and value.strip().lower() != "nan")
    candidates.append(dataset_dir / subdir)
    if source_run is not None:
        candidates.append(source_run / dataset_dir.name / subdir)
    return candidates


def plan_dataset(dataset_dir: Path, args: argparse.Namespace) -> list[dict]:
    metrics_path, predictions_path = dataset_dir / "metrics.csv", dataset_dir / "predictions.csv"
    if not (metrics_path.exists() and predictions_path.exists()):
        return [{"dataset": dataset_dir.name, "model": "", "kind": "", "n_train": 0, "source": "missing_inputs",
                 "fold_trainings": 0, "detail": "metrics.csv or predictions.csv missing: seed the run first"}]
    metrics = pd.read_csv(metrics_path, low_memory=False)
    payloads = runner.rebuild_prediction_payloads([pd.read_csv(predictions_path, low_memory=False)])
    rows = []
    for model_name, payload in payloads.items():
        workflow = str(payload.get("workflow", ""))
        kind = member_kind(model_name, workflow)
        n_train = len(payload["train"])
        row = {"dataset": dataset_dir.name, "model": model_name, "kind": kind, "n_train": n_train,
               "source": "", "fold_trainings": 0, "detail": ""}
        if kind == "fusion":
            row.update(source="excluded", detail="fusion output (CFA/ensemble) is never a member")
        elif payload.get("oof") is not None:
            row.update(source="saved_oof", detail="OOF rows already in predictions.csv")
        elif kind == "unimol":
            reasons = []
            for candidate in unimol_candidates(model_name, metrics, dataset_dir, args.source_run,
                                               args.random_seed, args.unimol_model_size):
                oof, reason = runner.load_unimol_saved_oof(candidate, n_train=n_train,
                                                           reference_train_pred=payload.get("train"))
                if oof is not None:
                    row.update(source="unimol_cvdata", detail=str(candidate))
                    break
                reasons.append(reason)
            else:
                if args.scope == "all":
                    row.update(source="gpu_refit", fold_trainings=args.folds,
                               detail="NO usable cv.data -> would refit Uni-Mol per fold: " + "; ".join(reasons))
                else:
                    row.update(source="excluded", detail="no usable cv.data and --scope cpu: " + "; ".join(reasons))
        elif kind == "chemprop":
            if args.scope == "all":
                row.update(source="gpu_refit", fold_trainings=args.folds, detail="Chemprop fold refits")
            else:
                row.update(source="excluded", detail="--scope cpu: Chemprop left out of the ensemble")
        elif kind == "tabpfn":
            if args.allow_api_refits:
                row.update(source="api_refit", fold_trainings=args.folds, detail="metered Prior Labs credits")
            else:
                row.update(source="excluded", detail="metered API; pass --ensemble-oof-allow-api-refits to include")
        elif kind == "ga_tuned":
            row.update(source="excluded", detail="GA-tuned rows have no OOF refit path")
        else:
            row.update(source="cpu_refit", fold_trainings=args.folds, detail=f"{kind} fold refits")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scope", choices=["all", "cpu"], default="cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--allow-api-refits", action="store_true")
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--unimol-model-size", default="84m")
    args = parser.parse_args(argv)

    dataset_dirs = sorted(p for p in args.run_dir.iterdir() if p.is_dir() and (p / "metrics.csv").exists())
    rows = [row for dataset_dir in dataset_dirs for row in plan_dataset(dataset_dir, args)]
    plan = pd.DataFrame(rows)
    out_path = args.run_dir / "ensemble_oof_plan.csv"
    plan.to_csv(out_path, index=False)

    print(f"Datasets: {len(dataset_dirs)}   scope={args.scope}   folds={args.folds}   plan -> {out_path}")
    if plan.empty:
        print("No prediction payloads found: seed the run directory first (prepare_chemprop_repair_run.py).")
        return 2
    print("\nMembers by OOF source:")
    print(plan.groupby(["source", "kind"]).size().rename("members").to_string())
    fold_trainings = plan.groupby("source")["fold_trainings"].sum()
    print("\nFold trainings (each fits a new fold model; no full model is retrained):")
    print(fold_trainings.to_string())
    gpu = plan[plan["source"] == "gpu_refit"]
    if not gpu.empty:
        seconds = sum(
            MEDIAN_FULL_FIT_SECONDS.get(kind, 300.0) * (args.folds - 1) / args.folds * count
            for kind, count in gpu.groupby("kind")["fold_trainings"].sum().items()
        )
        print(f"\nEstimated GPU fold-training time: ~{seconds / 3600:.0f} h (median-based; large datasets dominate)")
    missing = plan[plan["source"] == "missing_inputs"]
    unimol_gpu = gpu[gpu["kind"] == "unimol"]
    status = 0
    if not missing.empty:
        print(f"\nSTOP: {len(missing)} dataset(s) lack metrics.csv/predictions.csv.")
        status = 2
    if args.scope == "cpu" and not gpu.empty:
        print("\nSTOP: GPU refits planned under --scope cpu (should be impossible).")
        status = 2
    if not unimol_gpu.empty:
        print(f"\nSTOP: {len(unimol_gpu)} Uni-Mol member(s) have no usable cv.data and would be refitted on the GPU:")
        print(unimol_gpu[["dataset", "model", "detail"]].to_string(index=False))
        status = 2
    excluded_unimol = plan[(plan["kind"] == "unimol") & (plan["source"] == "excluded")]
    if not excluded_unimol.empty:
        print(f"\nNote: {len(excluded_unimol)} Uni-Mol member(s) will be left out (no usable cv.data).")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
