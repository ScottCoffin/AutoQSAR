"""F1: precedence is command line > --config file > built-in defaults, including profile defaults."""

from __future__ import annotations

import pytest

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_overrides_yaml_over_defaults(tmp_path):
    cfg = _write(tmp_path / "run.yaml", "split:\n  strategy: scaffold\n  test_fraction: 0.3\ninput:\n  path: data/x.csv\n")
    args, info = runner.prepare_args(["--config", str(cfg), "--test-fraction", "0.25"])
    assert args.test_fraction == 0.25  # CLI beats the file
    assert args.split_strategy == "scaffold"  # file beats the default
    assert args.random_seed == 13  # default
    assert args.dataset == [str((tmp_path / "data" / "x.csv").resolve())]  # relative to the config file
    assert info["mode"] == "single"
    assert info["resolved_config"]["split"]["test_fraction"] == 0.25
    assert set(info["config_explicit_keys"]) == {"split.strategy", "split.test_fraction", "input.path"}


def test_config_value_beats_profile_default(tmp_path):
    cfg = _write(tmp_path / "run.yaml", "models:\n  profile: full\ndeep:\n  chemprop:\n    epochs: 7\n")
    args, _ = runner.prepare_args(["--config", str(cfg)])
    assert args.benchmark_profile == "full"
    assert args.chemprop_epochs == 7  # the full profile would otherwise set 40
    assert args.chemprop_ensemble_size == 3  # still the full-profile default


def test_family_switches_and_quick_profile(tmp_path):
    args, info = runner.prepare_args(["--benchmark-profile", "quick"])
    assert not args.run_chemprop_mpnn and not args.run_maplight_gnn and not args.run_chemml_pytorch
    assert not args.run_tabpfn and not args.run_cnn
    assert info["resolved_config"]["models"]["enable_families"]["conventional_ml"] is True
    assert info["resolved_config"]["models"]["enable_families"]["gradient_boosting"] is False
    assert runner.model_filter_allows(args, "Random forest")
    assert not runner.model_filter_allows(args, "XGBoost")
    args, _ = runner.prepare_args(["--disable-model-families", "fusion,ensemble", "--disable-model", "SVR"])
    assert not args.run_cfa and not args.run_ensemble
    assert not runner.model_filter_allows(args, "SVR")


def test_ga_mode_on_uses_estimators(tmp_path):
    cfg = _write(tmp_path / "run.yaml", "ga_tuning:\n  mode: on\n  estimators: [elastic_net]\n  max_configs: 5\n")
    args, info = runner.prepare_args(["--config", str(cfg)])
    names, meta = runner.resolve_requested_ga_models(args, tmp_path)
    assert names == ["ElasticNet"] and meta["source"] == "ga_estimators"
    assert args.ga_max_configs == 5
    assert info["resolved_config"]["ga_tuning"]["mode"] == "on"


def test_invalid_config_gives_actionable_error_and_exit_2(tmp_path, capsys):
    cfg = _write(tmp_path / "bad.yaml", "split:\n  strategy: scafold\nmodels:\n  profle: quick\n")
    assert runner.main(["--config", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "split.strategy: 'scafold' is not one of" in err
    assert "Did you mean 'scaffold'" in err
    assert "models.profle: unknown option" in err and "models.profile" in err


def test_resolved_validation_catches_missing_input(tmp_path):
    with pytest.raises(runner.qsarena_config.ConfigError, match="needs batch.source"):
        runner.prepare_args(["--run-mode", "batch"])
