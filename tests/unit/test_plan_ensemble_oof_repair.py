from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from portable_colab_qsar_bundle import plan_ensemble_oof_repair as planner
from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner


def _seed_run(tmp_path):
    run = tmp_path / "run"
    ds = run / "toy"
    ds.mkdir(parents=True)
    n = 12
    smiles = pd.Series([f"C{'C' * i}" for i in range(n)])
    y = pd.Series(np.linspace(0, 1, n))
    test_smiles, y_test = pd.Series(["N", "NN"]), pd.Series([0.2, 0.8])
    frames, metrics = [], []
    unimol_dir = ds / "unimol_v1" / "seed_13"
    unimol_dir.mkdir(parents=True)
    joblib.dump((y.to_numpy() + 0.05).reshape(-1, 1), unimol_dir / "cv.data")
    for model, workflow in [
        ("Random forest", "conventional"),
        ("Chemprop v2 (D-MPNN, ensemble=3)", "Chemprop v2"),
        ("Uni-Mol V1", "Uni-Mol"),
        ("Uni-Mol V2 (84m)", "Uni-Mol"),  # no cv.data saved
        ("TabPFNRegressor", "conventional"),
        ("CFA (Combinatorial Fusion)", "cfa"),
    ]:
        frames.append(runner.prediction_frame("toy", model, workflow, "train", smiles, y, y.to_numpy()))
        frames.append(runner.prediction_frame("toy", model, workflow, "test", test_smiles, y_test, y_test.to_numpy()))
        metrics.append({"model": model, "workflow": workflow})
    pd.concat(frames).to_csv(ds / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(ds / "metrics.csv", index=False)
    return run


def test_cpu_scope_plans_no_gpu_work_and_reads_unimol_cvdata(tmp_path, capsys) -> None:
    run = _seed_run(tmp_path)
    status = planner.main([str(run), "--scope", "cpu"])
    plan = pd.read_csv(run / "ensemble_oof_plan.csv").set_index("model")
    assert plan.loc["Random forest", "source"] == "cpu_refit"
    assert plan.loc["Uni-Mol V1", "source"] == "unimol_cvdata"
    assert plan.loc["Uni-Mol V2 (84m)", "source"] == "excluded"
    assert plan.loc["Chemprop v2 (D-MPNN, ensemble=3)", "source"] == "excluded"
    assert plan.loc["TabPFNRegressor", "source"] == "excluded"
    assert plan.loc["CFA (Combinatorial Fusion)", "source"] == "excluded"
    assert (plan["source"] != "gpu_refit").all()
    assert status == 0


def test_all_scope_flags_unimol_without_cvdata_before_launch(tmp_path) -> None:
    run = _seed_run(tmp_path)
    status = planner.main([str(run), "--scope", "all"])
    plan = pd.read_csv(run / "ensemble_oof_plan.csv").set_index("model")
    assert plan.loc["Chemprop v2 (D-MPNN, ensemble=3)", "source"] == "gpu_refit"
    assert plan.loc["Uni-Mol V2 (84m)", "source"] == "gpu_refit"
    assert plan.loc["Uni-Mol V1", "source"] == "unimol_cvdata"
    assert status == 2  # a Uni-Mol GPU refit is planned: the agent must stop and check
