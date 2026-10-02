"""Post-run checks for benchmark_results/qsarena_benchmark_oof_ensemble (AGENTS.md step 7 + 2026-09-29/10-01 fixes).

    python tools/verify_oof_ensemble_run.py            # exit 1 on any hard failure

Hard checks: both ensemble rows per dataset, OOF member selection, no ensemble errors; base-model rows
unchanged versus git HEAD (primary_metric_value of every non-ensemble/non-fusion model); feature
selections unchanged versus git HEAD. Reported (soft): Uni-Mol membership in regression ensembles,
clipping notes, Chemprop "Selected descriptors" membership, ensembles far worse than their best member.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("benchmark_results/qsarena_benchmark_oof_ensemble")
FUSION = ("Ensemble", "CFA")


def git_csv(path: Path) -> pd.DataFrame | None:
    out = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], capture_output=True, text=True, encoding="utf-8")
    return pd.read_csv(io.StringIO(out.stdout)) if out.returncode == 0 and out.stdout else None


def main() -> int:
    hard, soft = [], []
    stats = {"unimol_in_reg": 0, "reg_with_unimol_valid": 0, "clipped": 0, "seldesc_member": 0, "seldesc_valid": 0}
    for d in sorted(p.parent for p in RUN.glob("*/metrics.csv")):
        m = pd.read_csv(d / "metrics.csv", low_memory=False)
        ens = m[m["model"].astype(str).str.startswith("Ensemble")].drop_duplicates("model", keep="last")
        if len(ens) != 2:
            hard.append(f"{d.name}: {len(ens)} ensemble rows")
        if (ens.get("ensemble_member_selection_split", pd.Series(dtype=str)).astype(str) != "oof").any():
            hard.append(f"{d.name}: ensemble not built on OOF")
        if ens["error"].notna().any():
            hard.append(f"{d.name}: ensemble error {ens['error'].dropna().iloc[0][:80]}")
        members = " | ".join(ens["ensemble_members"].astype(str))
        notes = " | ".join(ens["ensemble_member_filter_notes"].astype(str))
        valid = m[m["error"].isna()].drop_duplicates("model", keep="last")
        regression = "test_rmse" in valid and valid["test_rmse"].notna().any()
        if regression and (valid["model"] == "Uni-Mol V1").any():
            stats["reg_with_unimol_valid"] += 1
            stats["unimol_in_reg"] += "Uni-Mol V1" in members
        stats["clipped"] += "clipped to the training target range" in notes
        seldesc = "Chemprop v2 (D-MPNN + Selected descriptors, ensemble=3)"
        if (valid["model"] == seldesc).any():
            stats["seldesc_valid"] += 1
            stats["seldesc_member"] += seldesc in members
        # ensembles far worse than the best member (lower-is-better RMSE for regression; AUROC otherwise)
        base = valid[~valid["model"].astype(str).str.startswith(FUSION)]
        col, lower = ("test_rmse", True) if regression else ("test_roc_auc", False)
        if col in base and base[col].notna().any() and ens[col].notna().any():
            best = base[col].min() if lower else base[col].max()
            worst_ens = ens[col].max() if lower else ens[col].min()
            ratio = worst_ens / best if lower else best / max(worst_ens, 1e-9)
            if ratio > 1.25:
                soft.append(f"{d.name}: worst ensemble {col}={worst_ens:.3f} vs best member {best:.3f} (x{ratio:.2f})")
        # base rows unchanged versus HEAD
        old = git_csv(d / "metrics.csv")
        if old is not None:
            def base_vals(frame):
                f = frame[frame["error"].isna() & ~frame["model"].astype(str).str.startswith(FUSION)]
                return f.drop_duplicates("model", keep="last").set_index("model")["primary_metric_value"]

            a, b = base_vals(old), base_vals(m)
            common = a.index.intersection(b.index)
            diff = [k for k in common if not np.isclose(a[k], b[k], rtol=0, atol=1e-12, equal_nan=True)]
            if diff:
                hard.append(f"{d.name}: base-model metric changed for {diff[:3]}")
            if len(common) < len(a):
                hard.append(f"{d.name}: base models missing vs HEAD: {sorted(set(a.index) - set(b.index))[:3]}")
        old_sel = git_csv(d / "selected_features.csv")
        if old_sel is not None and (d / "selected_features.csv").exists():
            if not old_sel.equals(pd.read_csv(d / "selected_features.csv")):
                hard.append(f"{d.name}: selected_features.csv differs from HEAD")
    print(f"Uni-Mol V1 in regression ensembles: {stats['unimol_in_reg']}/{stats['reg_with_unimol_valid']}")
    print(f"datasets with clipping notes: {stats['clipped']}")
    print(f"Chemprop Selected-descriptors in ensembles where valid: {stats['seldesc_member']}/{stats['seldesc_valid']}")
    for line in soft:
        print("WARN", line)
    for line in hard:
        print("FAIL", line)
    print(f"{'PASS' if not hard else 'FAIL'}: {len(hard)} hard failure(s), {len(soft)} warning(s)")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
