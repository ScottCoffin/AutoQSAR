"""RunConfig (qsarena/config.py): parsing, validation, round trips, generated docs, notebook mapping."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from qsarena import config as qc

REPO = Path(__file__).resolve().parents[2]
APPENDIX_A = REPO / "tests" / "fixtures" / "config" / "appendix_a_run.yaml"


def _parser():
    from portable_colab_qsar_bundle.run_qsarena_benchmarks import build_arg_parser

    return build_arg_parser()


def test_defaults_validate():
    cfg = qc.RunConfig.defaults()
    cfg.validate(resolved=False)
    assert cfg.split.strategy == "target_quartile"
    assert cfg.split.test_fraction == 0.2
    assert cfg.split.seed == 13
    assert cfg.models.profile == "cost_optimized"
    assert cfg.ga_tuning.mode == "off"
    assert cfg.report.formats == ["html", "md"]


def test_all_fifteen_groups_have_options():
    groups = {spec.group for spec in qc.iter_options()}
    assert groups == set(range(1, 16))


def test_run_yaml_round_trips():
    text = qc.render_example_yaml()
    cfg = qc.RunConfig.from_dict(yaml.safe_load(text))
    assert cfg.to_dict() == qc.RunConfig().to_dict()


def test_appendix_a_run_yaml_loads_cleanly():
    cfg = qc.load_run_config(APPENDIX_A)
    assert cfg.run.mode == "single"
    assert cfg.standardize.strip_salts is True
    assert "maplight" in cfg.features.families  # avalon / erg are aliases for the MapLight composite
    assert cfg.deep.unimol.v2 == "true"  # v2_84m alias
    assert cfg.deep.chemprop.variants == ["dmpnn", "dmpnn_rdkit2d", "selected_features", "cmpnn", "attentivefp"]
    assert cfg.evaluation.classification_metrics[0] == "roc_auc"  # auroc alias
    assert cfg.ga_tuning.mode == "off"  # YAML 1.1 reads bare `off` as False
    assert cfg.ensemble.exclude_negative_test_r2_members is False


def test_invalid_option_reports_field_and_allowed_values():
    with pytest.raises(qc.ConfigError) as info:
        qc.RunConfig.from_dict({"split": {"strategy": "scafold"}})
    message = str(info.value)
    assert "split.strategy" in message
    assert "target_quartile, random, scaffold, predefined" in message
    assert "Did you mean 'scaffold'" in message


def test_unknown_key_suggests_correction():
    with pytest.raises(qc.ConfigError) as info:
        qc.RunConfig.from_dict({"split": {"test_fracton": 0.3}})
    assert "split.test_fracton: unknown option" in str(info.value)
    assert "split.test_fraction" in str(info.value)


def test_out_of_range_and_wrong_type_are_rejected():
    with pytest.raises(qc.ConfigError, match="split.test_fraction: 0.9 is above the maximum 0.5"):
        qc.RunConfig.from_dict({"split": {"test_fraction": 0.9}})
    with pytest.raises(qc.ConfigError, match="split.cv_folds: 'five' is not an integer"):
        qc.RunConfig.from_dict({"split": {"cv_folds": "five"}})
    with pytest.raises(qc.ConfigError, match="locked"):
        qc.RunConfig.from_dict({"report": {"formats": ["html"]}})


def test_cross_field_rules_apply_to_the_resolved_config_only():
    cfg = qc.RunConfig.from_dict({"run": {"mode": "single"}})  # path may come from --dataset
    with pytest.raises(qc.ConfigError, match="needs input.path"):
        cfg.validate(resolved=True)


def _non_default_value(spec: qc.OptionSpec):
    """A valid value different from the default, for every option kind."""
    special = {
        "run.output_dir": "some/dir",
        "input.path": "a.csv",
        "input.target_col": ["t1", "t2"],
        "batch.source": "manifest.csv",
        "run.mode": "batch",
        "features.families": ["morgan", "rdkit"],
        "feature_selection.binary_prevalence_range": [0.01, 0.99],
        "models.enable_families": {"gradient_boosting": False, "fusion": False},
        "models.disable_models": ["Tabular CNN"],
        "models.only_models": ["SVR", "Ensemble"],
        "ga_tuning.estimators": ["catboost"],
        "ga_tuning.mode": "on",
        "deep.chemprop.variants": ["cmpnn"],
        "multiseed.seeds": [3, 7],
        "run.dataset_names": ["tdc_caco2_wang"],
        "evaluation.regression_metrics": ["rmse", "mae"],
        "evaluation.classification_metrics": ["auprc"],
        "caching.granularity": ["dataset"],
        "report.formats": ["md", "html"],
        "split.predefined_split_col": "split",
        "input.smiles_col": "SMILES",
        "input.id_col": "id",
        "input.classification_threshold": 1.5,
        "input.task": "classification",
        "evaluation.primary_metric": "mae",
        "deep.use_gpu": "false",
        "applicability_domain.method": "off",
        "ensemble.member_selection_metric": "test",
        "ensemble.simple_average": True,
    }
    if spec.key in special:
        return special[spec.key]
    if spec.kind == "bool":
        return not bool(spec.default) if spec.default is not None else False
    if spec.kind == "tristate":
        return "true"
    if spec.kind == "choice":
        return next(c for c in spec.choices if c != spec.default)
    if spec.kind == "int":
        return int((spec.default or 0) + 3) if spec.maximum is None else int(spec.minimum or 1)
    if spec.kind == "float":
        base = float(spec.default or 0.1)
        value = base * 1.5 if base else 0.5
        if spec.maximum is not None:
            value = min(value, float(spec.maximum))
            if value == base:
                value = (float(spec.minimum or 0.0) + base) / 2
        return value
    raise AssertionError(f"no test value for {spec.key} ({spec.kind})")


@pytest.mark.parametrize("spec", qc.iter_options(), ids=lambda s: s.key)
def test_each_option_round_trips_through_cli_namespace_and_manifest(spec):
    value = _non_default_value(spec)
    cfg = qc.RunConfig()
    cfg.set(spec.key, value)
    expected = cfg.get(spec.key)
    args = _parser().parse_args([])
    for dest, arg_value in cfg.to_arg_values().items():
        assert hasattr(args, dest), f"{spec.key} maps to unknown argparse dest {dest!r}"
        setattr(args, dest, arg_value)
    resolved = qc.RunConfig.from_namespace(args)
    assert resolved.get(spec.key) == expected, spec.key
    # ...and survives the JSON manifest.
    manifest = json.loads(json.dumps(resolved.to_manifest_dict(), default=str))
    node = manifest["config"]
    for part in spec.key.split("."):
        node = node[part]
    assert node == json.loads(json.dumps(expected, default=str))


def test_manifest_contains_resolved_config_and_signature():
    args = _parser().parse_args(["--dataset", "x.csv", "--test-fraction", "0.3"])
    resolved = qc.RunConfig.from_namespace(args)
    manifest = resolved.to_manifest_dict()
    assert manifest["config"]["split"]["test_fraction"] == 0.3
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["config_signature"])
    other = qc.RunConfig.from_namespace(_parser().parse_args(["--dataset", "x.csv"]))
    assert other.config_signature() != resolved.config_signature()
    # run-only options (paths, verbosity) do not change the signature
    quiet = qc.RunConfig.from_namespace(_parser().parse_args(["--dataset", "x.csv", "--test-fraction", "0.3", "--verbosity", "quiet"]))
    assert quiet.config_signature() == resolved.config_signature()


def test_options_reference_lists_all_15_groups_and_is_up_to_date():
    text = qc.render_options_reference()
    for number, name in qc.GROUPS.items():
        assert f"## {number}. {name}" in text
    for spec in qc.iter_options():
        assert f"`{spec.key}`" in text
    committed = (REPO / "docs" / "options_reference.md").read_text(encoding="utf-8")
    assert committed.replace("\r\n", "\n") == text, "run: python -m qsarena.config --write-docs"


def test_example_yaml_is_up_to_date():
    committed = (REPO / "configs" / "run.example.yaml").read_text(encoding="utf-8")
    assert committed.replace("\r\n", "\n") == qc.render_example_yaml(), "run: python -m qsarena.config --write-docs"


def test_help_lists_all_15_groups_and_every_option_flag():
    help_text = _parser().format_help()
    for number, name in qc.GROUPS.items():
        assert f"{number}. {name}:" in help_text
    for spec in qc.iter_options():
        if not spec.cli or spec.cli in {"(fixed)", "--run-chemprop-*"}:
            continue
        for flag in re.findall(r"--[a-z0-9-]+", spec.cli):
            assert flag in help_text, f"{spec.key}: {flag} missing from --help"


def test_per_dataset_overrides_reject_run_level_options():
    assert qc.per_dataset_arg_overrides({"split.cv_folds": "3"}, source="row 2") == {"cv_folds": 3}
    with pytest.raises(qc.ConfigError, match="run-level option"):
        qc.per_dataset_arg_overrides({"run.output_dir": "x"}, source="row 2")


BUILDER = REPO / "portable_colab_qsar_bundle" / "build_colab_qsar_tutorial.py"
NOTEBOOK = REPO / "portable_colab_qsar_bundle" / "colab_qsar_tutorial.ipynb"


def test_notebook_widget_keys_match_run_config_schema():
    builder = BUILDER.read_text(encoding="utf-8")
    params = set(re.findall(r"^\s*([A-Za-z_]\w*)\s*=.*#\s*@param", builder, flags=re.MULTILINE))
    keys = {spec.key for spec in qc.iter_options()}
    for key, widgets in qc.NOTEBOOK_WIDGETS.items():
        assert key in keys, key
        for widget in widgets:
            assert widget in params, f"{key} -> {widget} is not a notebook @param"
    # every mapped widget is overridable by a loaded run.yaml in the generated notebook
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for widget in qc.notebook_widget_names():
        assert f"QSARENA_WIDGET_OVERRIDES.get('{widget}'" in sources, widget
    assert "@title 0B. Optional: apply a QSARena run.yaml" in sources
    assert "@title 9E. Export the widget choices as run.yaml" in sources


def _cell(title: str) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if title in text:
            return text
    raise AssertionError(title)


def test_notebook_config_cells_load_and_export_run_yaml(tmp_path, monkeypatch):
    """Execute the generated 0B (load) and 9E (export) cells as the notebook would."""
    import sys

    monkeypatch.chdir(tmp_path)
    run_yaml = tmp_path / "in.yaml"
    run_yaml.write_text("split:\n  strategy: scaffold\n  test_fraction: 0.25\nfeatures:\n  families: [morgan, rdkit]\n", encoding="utf-8")
    namespace = {"Path": Path, "sys": sys, "IN_COLAB": True}  # Colab: the @param line is the widget value
    source = _cell("@title 0B.").replace('run_config_yaml_path = "" # @param', f'run_config_yaml_path = r"{run_yaml}" # @param')
    exec(compile(source, "cell_0B", "exec"), namespace)
    overrides = namespace["QSARENA_WIDGET_OVERRIDES"]
    assert overrides["data_split_strategy"] == "scaffold"
    assert overrides["test_fraction"] == 0.25
    assert overrides["use_morgan_features"] is True and overrides["use_maccs_keys"] is False
    namespace.update(overrides)
    exec(compile(_cell("@title 9E."), "cell_9E", "exec"), namespace)
    exported = qc.load_run_config(tmp_path / "qsarena_notebook_run.yaml")
    assert exported.split.strategy == "scaffold"
    assert exported.features.families == ["morgan", "rdkit"]
