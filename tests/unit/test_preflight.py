"""qsarena.preflight: dataset checks, backend messages and the cost heuristic."""

from __future__ import annotations

from qsarena import preflight

SMILES = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "not_a_smiles", "CCO", "CCCC", "CCCl", "OCCO", "C1CCCCC1"]


def _codes(result):
    return {m.code: m for m in result.messages}


def test_preflight_smiles_parse_rate_and_drop_count():
    result = preflight.preflight_dataset(SMILES, list(range(10)), name="demo", minimum_rows=5)
    assert result.n_rows == 10
    assert result.n_unparseable == 1
    assert result.parse_rate == 0.9
    assert result.n_valid == 9
    message = _codes(result)["unparseable_smiles"]
    assert message.level == "warning"
    assert "1 SMILES (10.0%) cannot be parsed by RDKit and will be dropped." == message.message
    assert "--no-drop-unparseable" in message.remedy


def test_preflight_duplicate_rate():
    result = preflight.preflight_dataset(SMILES, list(range(10)), name="demo", minimum_rows=5)
    assert abs(result.duplicate_fraction - 1 / 9) < 1e-9  # CCO appears twice among 9 parsed rows
    assert "--deduplicate canonical_smiles" in _codes(result)["duplicates"].remedy


def test_preflight_class_balance_and_inferred_task():
    labels = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    result = preflight.preflight_dataset(SMILES, labels, name="demo", minimum_rows=5)
    assert result.task == "classification"
    assert result.task_source == "auto: target has exactly two values"
    assert result.class_balance == {"0": 1, "1": 8}  # the unparseable row is not counted
    many = ["C" * n for n in range(1, 21)]  # 20 valid molecules, 1 minority label = 5 %
    imbalanced = preflight.preflight_dataset(many, [0] + [1] * 19, name="demo", minimum_rows=5)
    assert "class_imbalance" in _codes(imbalanced)
    assert "--primary-metric auprc" in _codes(imbalanced)["class_imbalance"].remedy
    regression = preflight.preflight_dataset(SMILES, [0.1 * i for i in range(10)], name="demo", minimum_rows=5)
    assert regression.task == "regression" and regression.class_balance == {}


def test_preflight_threshold_binarizes_and_not_binary_is_an_error():
    result = preflight.preflight_dataset(SMILES, list(range(10)), name="demo", minimum_rows=5, classification_threshold=5)
    assert result.task == "classification" and set(result.class_balance) == {"0", "1"}
    bad = preflight.preflight_dataset(SMILES, list(range(10)), name="demo", minimum_rows=5, task="classification")
    assert _codes(bad)["not_binary"].level == "error"
    assert "--classification-threshold" in _codes(bad)["not_binary"].remedy


def test_preflight_guardrails():
    too_few = preflight.preflight_dataset(SMILES, list(range(10)), name="demo", minimum_rows=20)
    assert too_few.will_skip and _codes(too_few)["too_few_rows"].level == "error"
    small = preflight.preflight_dataset(SMILES, list(range(10)), name="demo", minimum_rows=5, test_fraction=0.2)
    assert "about 2 molecules" in _codes(small)["small_dataset"].message


def test_backend_messages_are_actionable():
    status = preflight.backend_status(gpu_available=False)
    for name in ("xgboost", "catboost", "torch", "chemprop", "unimol_tools", "tabpfn", "dgl", "gpu"):
        assert name in status and "available" in status[name]
    fake = {name: {"available": False, "remedy": info["remedy"]} for name, info in status.items()}
    messages = preflight.backend_messages(fake, {"boosting": True, "chemprop": True, "unimol_auto": True, "unimol_v2": True})
    text = " | ".join(m.message + " -> " + m.remedy for m in messages)
    assert "pip install 'qsarena[boosting]'" in text
    assert "pip install 'qsarena[graph]'" in text
    assert "GPU not detected: Uni-Mol V1/V2 are skipped" in text
    assert "Uni-Mol V2 was requested but needs a GPU" in text


def test_cost_estimate_scales_with_size_and_gpu():
    small = preflight.estimate_model_seconds("graph_nn", 1605, gpu=True)
    assert abs(small - 268.6) < 1e-6
    assert preflight.estimate_model_seconds("graph_nn", 3210, gpu=True) == 2 * small
    assert preflight.estimate_model_seconds("graph_nn", 1605, gpu=False) == 8 * small
    stages = preflight.estimate_pipeline_seconds(1000, ["conventional_ml", "conventional_ml"], gpu=False, selector_seconds=3.0)
    assert set(stages) == {"features", "feature_selection", "conventional_ml"}
    assert preflight.format_duration(30) == "30 s" and preflight.format_duration(7200) == "2.0 h"
