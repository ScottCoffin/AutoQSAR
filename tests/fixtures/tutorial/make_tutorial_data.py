"""Regenerate the tutorial example data shipped in ``qsarena/examples/data``.

The structures are the 96 molecules of ``tests/fixtures/tdc_tiny`` (sampled from TDC); every target
is SYNTHETIC — a fixed formula of RDKit descriptors plus seeded noise — so the examples are small,
license-free and have a learnable signal. The output is deterministic; ``tests/docs`` checks that
running this script reproduces the committed files byte for byte.

    python tests/fixtures/tutorial/make_tutorial_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "qsarena" / "examples" / "data"
TINY = REPO / "tests" / "fixtures" / "tdc_tiny"


def pool() -> list[str]:
    smiles: list[str] = []
    for name in ("caco2_wang", "herg"):
        for part in ("train_val", "test"):
            smiles += pd.read_csv(TINY / name / f"{part}.csv")["Drug"].astype(str).tolist()
    seen, out = set(), []
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen:
            seen.add(canonical)
            out.append(text)
    return out


def descriptors(text: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(text)
    return {
        "logp": Crippen.MolLogP(mol),
        "mw": Descriptors.MolWt(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def write(frame: pd.DataFrame, relative: str) -> None:
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.4f")


def main() -> None:
    rng = np.random.default_rng(20260925)
    molecules = pool()
    d = pd.DataFrame([descriptors(s) for s in molecules])

    # 1. solubility.csv — regression, two targets, plus the problems the tutorial demonstrates:
    #    one salt, one unparseable SMILES and one duplicated structure.
    sol_smiles = molecules[:44]
    sol = d.iloc[:44]
    log_s = 0.8 - 0.65 * sol.logp - 0.006 * (sol.mw - 300) + 0.25 * sol.hbd + rng.normal(0, 0.25, len(sol))
    log_d = 0.9 * sol.logp - 0.004 * sol.tpsa + rng.normal(0, 0.2, len(sol))
    solubility = pd.DataFrame({
        "compound_id": [f"SOL-{i + 1:03d}" for i in range(len(sol))],
        "smiles": sol_smiles,
        "logS": log_s.round(4),
        "logD": log_d.round(4),
    })
    extras = pd.DataFrame({
        "compound_id": ["SOL-045", "SOL-046", "SOL-047"],
        # sodium salt of the first molecule's acid form is not needed: a plain salt pair suffices
        "smiles": [f"{sol_smiles[0]}.Cl", "this_is_not_a_smiles", sol_smiles[1]],
        "logS": [float(log_s.iloc[0]), -3.0, float(log_s.iloc[1])],
        "logD": [float(log_d.iloc[0]), 1.0, float(log_d.iloc[1])],
    })
    write(pd.concat([solubility, extras], ignore_index=True), "solubility.csv")

    # 2. bbb.csv — binary classification (0/1) from a TPSA/logP rule with 10% label noise.
    bbb_smiles = molecules[40:84]
    b = d.iloc[40:84]
    label = ((b.tpsa < 80) & (b.logp > 1.0)).astype(int).to_numpy()
    flip = rng.random(len(label)) < 0.10
    label = np.where(flip, 1 - label, label)
    write(pd.DataFrame({"molecule": [f"BBB-{i + 1:03d}" for i in range(len(b))], "smiles": bbb_smiles,
                        "bbb_penetrant": label}), "bbb.csv")

    # 3. batch_dir/ — two regression CSVs with auto-detectable column names (smiles, target).
    perm = d.iloc[10:50]
    write(pd.DataFrame({"smiles": molecules[10:50],
                        "target": (-4.6 + 0.35 * perm.logp - 0.012 * perm.tpsa + rng.normal(0, 0.2, len(perm))).round(4)}),
          "batch_dir/permeability.csv")
    lipo = d.iloc[50:90]
    write(pd.DataFrame({"smiles": molecules[50:90],
                        "target": (0.85 * lipo.logp + 0.3 * lipo.rings - 0.5 + rng.normal(0, 0.2, len(lipo))).round(4)}),
          "batch_dir/lipophilicity.csv")

    # 4. broken.csv — deliberately malformed (no SMILES column) to show batch failure handling.
    write(pd.DataFrame({"name": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]}), "broken.csv")

    # 5. batch_manifest.csv — three datasets, one malformed, with per-dataset overrides.
    manifest = pd.DataFrame([
        {"dataset_name": "solubility", "path": "solubility.csv", "smiles_col": "smiles", "target_col": "logS",
         "id_col": "compound_id", "task": "regression", "split": "random", "test_fraction": "0.25"},
        {"dataset_name": "bbb", "path": "bbb.csv", "smiles_col": "smiles", "target_col": "bbb_penetrant",
         "id_col": "molecule", "task": "classification", "split": "random", "test_fraction": ""},
        {"dataset_name": "broken", "path": "broken.csv", "smiles_col": "", "target_col": "", "id_col": "",
         "task": "", "split": "", "test_fraction": ""},
    ])
    manifest.to_csv(OUT / "batch_manifest.csv", index=False, lineterminator="\n")
    print(f"wrote tutorial example data to {OUT}")


if __name__ == "__main__":
    main()
