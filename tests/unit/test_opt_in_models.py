"""Opt-in models added from the feature-expansion arm: XGBoost (ADMETboost features) and the CheMeleon Chemprop variant.

Both are off by default, so the canonical benchmark configuration (and every resume signature) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner
from qsarena import config as qc


def _args(*argv: str):
    return runner.build_arg_parser().parse_args(["--dataset", "x.csv", *argv])


def test_both_models_are_off_by_default():
    args = _args()
    assert args.run_admetboost_xgboost is False
    assert args.run_chemprop_chemeleon is False
    assert runner.ADMETBOOST_XGB_LABEL not in runner.selected_conventional_model_names(args)
    assert not any(s["variant_tag"] == "chemeleon" for s in runner.chemprop_variant_specs(args))


def test_chemeleon_variant_spec_alone_and_alongside_others():
    alone = runner.chemprop_variant_specs(_args("--no-run-chemprop-attentivefp", "--no-run-chemprop-selected-features",
                                                "--run-chemprop-chemeleon", "--chemprop-ensemble-size", "3"))
    assert [s["variant_tag"] for s in alone] == ["chemeleon"]
    spec = alone[0]
    assert spec["train_args"] == ["--from-foundation", "CHEMELEON"]
    assert spec["label"] == "Chemprop v2 (CheMeleon fine-tuned, ensemble=3)"
    assert qc.model_family(spec["label"]) == "graph_nn"
    both = runner.chemprop_variant_specs(_args("--run-chemprop-dmpnn", "--run-chemprop-chemeleon"))
    tags = [s["variant_tag"] for s in both]
    assert "dmpnn" in tags and tags[-1] == "chemeleon"


@pytest.mark.skipif(runner.XGBRegressor is None, reason="xgboost not installed")
def test_admetboost_xgboost_is_listed_and_uses_the_arm_settings():
    from qsarena.feature_expansion.train import XGB_PARAMS

    args = _args("--run-admetboost-xgboost")
    assert runner.ADMETBOOST_XGB_LABEL in runner.selected_conventional_model_names(args)
    assert qc.model_family(runner.ADMETBOOST_XGB_LABEL) == "gradient_boosting"
    est = runner.admetboost_xgboost_estimator(args, n_jobs=1)
    params = est.get_params()
    for key, value in XGB_PARAMS.items():
        assert params[key] == value
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 4)), columns=list("abcd"))
    models = runner.conventional_models(args, X, pd.Series(np.arange(30.0)), pd.Series(["C"] * 30))
    assert runner.ADMETBOOST_XGB_LABEL in models


def test_new_flags_do_not_enter_existing_family_signatures():
    """Adding these flags must not invalidate cached rows of any existing family on resume."""
    for keys in runner._FAMILY_SIGNATURE_ARGS.values():
        assert "run_admetboost_xgboost" not in keys and "run_chemprop_chemeleon" not in keys
    base, on = _args(), _args("--run-admetboost-xgboost", "--run-chemprop-chemeleon")
    for family in runner._FAMILY_SIGNATURE_ARGS:
        assert runner.family_arg_signature(base, family) == runner.family_arg_signature(on, family)


def test_runconfig_round_trips_the_new_options():
    cfg = qc.RunConfig()
    cfg.models.admetboost_xgboost = True
    cfg.deep.chemprop.variants = ["dmpnn", "chemeleon"]
    values = cfg.to_arg_values(explicit_only=False)
    assert values["run_admetboost_xgboost"] is True
    assert values["run_chemprop_chemeleon"] is True
    assert values["run_chemprop_attentivefp"] is False


def test_explicit_disable_model_families_survives_the_profile():
    argv = ["--dataset", "x.csv", "--benchmark-profile", "quick", "--disable-model-families", "graph_nn"]
    args = runner.build_arg_parser().parse_args(argv)
    runner.apply_benchmark_profile_defaults(args, argv)
    assert args.disabled_model_families == ["graph_nn"]
    argv = ["--dataset", "x.csv", "--benchmark-profile", "quick"]
    args = runner.build_arg_parser().parse_args(argv)
    runner.apply_benchmark_profile_defaults(args, argv)
    assert "gradient_boosting" in args.disabled_model_families
