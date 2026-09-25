from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner


def _tdc_spec() -> runner.DatasetSpec:
    return runner.DatasetSpec(
        name="caco2_wang",
        source="PyTDC ADMET Benchmark Group: caco2_wang",
        frame=pd.DataFrame({"Drug": ["CCO"], "Y": [1.0], "split": ["train"]}),
        smiles_column="Drug",
        target_column="Y",
        recommended_split="predefined",
        benchmark_suite="tdc",
        benchmark_id="caco2_wang",
        predefined_split_column="split",
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        tdc22_multiseed_seeds="1,2",
        tdc22_multiseed_output_name="tdc22_best_model_multiseed",
        resume=True,
        output_dir=tmp_path,
        ga_models="",
    )


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "caco2_wang",
                "model": "Random forest",
                "workflow": "Conventional ML",
                "family": "Conventional ML",
                "primary_metric": "rmse",
                "primary_metric_value": 1.00,
            }
        ]
    )


def test_multiseed_summary_marks_single_seed_within_one_sd() -> None:
    metrics = pd.DataFrame(
        [
            {
                "dataset": "caco2_wang",
                "model": "Random forest",
                "workflow": "Conventional ML",
                "primary_metric": "rmse",
                "primary_metric_value": 0.9,
                "stage_duration_seconds": 10,
                "multiseed_seed": 1,
                "multiseed_selected_model": "Random forest",
            },
            {
                "dataset": "caco2_wang",
                "model": "Random forest",
                "workflow": "Conventional ML",
                "primary_metric": "rmse",
                "primary_metric_value": 1.1,
                "stage_duration_seconds": 12,
                "multiseed_seed": 2,
                "multiseed_selected_model": "Random forest",
            },
        ]
    )
    plan = pd.DataFrame(
        [
            {
                "dataset": "caco2_wang",
                "status": "planned",
                "selected_model": "Random forest",
                "main_run_primary_metric_value": 1.0,
                "main_run_winner_model": "Random forest",
                "selection_note": "overall_best_direct_model",
                "seeds": "1,2",
            }
        ]
    )
    out = runner.summarize_tdc22_multiseed_metrics(metrics, plan)
    row = out.iloc[0]
    assert row["seed_count"] == 2
    assert row["mean_primary_metric_value"] == 1.0
    assert round(row["std_primary_metric_value"], 6) == round(0.14142135623730956, 6)
    assert bool(row["single_seed_winner_within_1sd"]) is True
    assert row["metric_direction"] == "lower"


def test_tdc22_multiseed_reuses_completed_seed_metrics(tmp_path, monkeypatch) -> None:
    spec = _tdc_spec()
    args = _args(tmp_path)
    root = tmp_path / "run"
    for seed, value in [(1, 0.9), (2, 1.1)]:
        dataset_dir = root / "tdc22_best_model_multiseed" / f"seed_{seed}" / "caco2_wang"
        dataset_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "dataset": "caco2_wang",
                    "model": "Random forest",
                    "workflow": "Conventional ML",
                    "primary_metric": "rmse",
                    "primary_metric_value": value,
                    "stage_duration_seconds": 10 + seed,
                }
            ]
        ).to_csv(dataset_dir / "metrics.csv", index=False)

    def fail_run_dataset(*_args, **_kwargs):  # pragma: no cover - should never be called
        raise AssertionError("run_dataset should not be called when resumable seed metrics exist")

    monkeypatch.setattr(runner, "run_dataset", fail_run_dataset)
    payload = runner.run_tdc22_best_model_multiseed(datasets=[spec], output_dir=root, args=args, summary=_summary())
    assert payload["status"] == "completed"
    summary_path = root / "tdc22_best_model_multiseed" / "tdc22_best_model_multiseed_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "seed_count"] == 2
    assert bool(summary.loc[0, "single_seed_winner_within_1sd"]) is True


def test_tdc22_multiseed_reruns_partial_seed_metrics(tmp_path, monkeypatch) -> None:
    spec = _tdc_spec()
    args = _args(tmp_path)
    args.tdc22_multiseed_seeds = "1"
    root = tmp_path / "run"
    dataset_dir = root / "tdc22_best_model_multiseed" / "seed_1" / "caco2_wang"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "caco2_wang",
                "model": "Random forest",
                "workflow": "Conventional ML",
                "primary_metric": "rmse",
                "primary_metric_value": 0.9,
                "stage_duration_seconds": 11,
            }
        ]
    ).to_csv(dataset_dir / "metrics.csv", index=False)
    summary = pd.concat(
        [
            _summary(),
            pd.DataFrame(
                [
                    {
                        "dataset": "caco2_wang",
                        "model": "Linear regression",
                        "workflow": "Conventional ML",
                        "family": "Conventional ML",
                        "primary_metric": "rmse",
                        "primary_metric_value": 1.0,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "dataset": "caco2_wang",
                        "model": "Ensemble (Simple average)",
                        "workflow": "Ensemble",
                        "family": "Ensemble",
                        "primary_metric": "rmse",
                        "primary_metric_value": 0.8,
                        "ensemble_members": "Random forest, Linear regression",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    calls = {"count": 0}

    def fake_run_dataset(_spec, _seed_output_dir, _seed_args, **_kwargs):
        calls["count"] += 1
        rows = [
            {
                "dataset": "caco2_wang",
                "model": "Random forest",
                "workflow": "Conventional ML",
                "primary_metric": "rmse",
                "primary_metric_value": 0.9,
                "stage_duration_seconds": 11,
            },
            {
                "dataset": "caco2_wang",
                "model": "Linear regression",
                "workflow": "Conventional ML",
                "primary_metric": "rmse",
                "primary_metric_value": 1.0,
                "stage_duration_seconds": 12,
            },
        ]
        return runner.DatasetRunResult(metrics_rows=rows, prediction_tables=[], ga_history_tables=[], status="completed", elapsed_seconds=1.0)

    monkeypatch.setattr(runner, "run_dataset", fake_run_dataset)
    payload = runner.run_tdc22_best_model_multiseed(datasets=[spec], output_dir=root, args=args, summary=summary)
    assert payload["status"] == "completed"
    assert calls["count"] == 1
    table = pd.read_csv(root / "tdc22_best_model_multiseed" / "tdc22_best_model_multiseed_summary.csv")
    assert set(table["selected_model"]) == {"Random forest", "Linear regression"}


def test_independent_multiseed_writes_requested_results_csv(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source_run"
    source.mkdir()
    _summary().to_csv(source / "summary_metrics.csv", index=False)
    out_dir = tmp_path / "multiseed_run"
    results_csv = tmp_path / "results" / "tdc22_multiseed.csv"
    args = _args(out_dir)
    args.tdc22_multiseed_source_run = source
    args.tdc22_multiseed_summary_csv = results_csv

    def fake_run_dataset(_spec, seed_output_dir, seed_args, **_kwargs):
        seed = int(seed_args.random_seed)
        value = 0.9 if seed == 1 else 1.1
        row = {
            "dataset": "caco2_wang",
            "model": "Random forest",
            "workflow": "Conventional ML",
            "primary_metric": "rmse",
            "primary_metric_value": value,
            "stage_duration_seconds": 10 + seed,
        }
        return runner.DatasetRunResult(metrics_rows=[row], prediction_tables=[], ga_history_tables=[], status="completed", elapsed_seconds=1.0)

    monkeypatch.setattr(runner, "run_dataset", fake_run_dataset)
    payload = runner.run_independent_tdc22_multiseed(
        root=tmp_path,
        datasets=[_tdc_spec()],
        output_dir=out_dir,
        args=args,
    )
    assert payload["status"] == "completed"
    assert payload["source_run"] == str(source)
    assert results_csv.exists()
    table = pd.read_csv(results_csv)
    assert list(table["dataset"]) == ["caco2_wang"]
    assert table.loc[0, "seed_count"] == 2
