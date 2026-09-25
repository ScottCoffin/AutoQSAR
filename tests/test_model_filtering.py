from argparse import Namespace

import pandas as pd

from portable_colab_qsar_bundle.run_qsarena_benchmarks import (
    leaderboard_comparison_by_dataset,
    model_filter_values,
)


def test_model_filter_preserves_commas_in_exact_model_names() -> None:
    label = "Chemprop v2 (D-MPNN, ensemble=1)"

    assert model_filter_values(Namespace(only_model_names=[label])) == {label}


def test_model_filter_accepts_internal_model_lists() -> None:
    labels = ["Random forest", "Voting Regressor (KNN, SVM)"]

    assert model_filter_values(Namespace(only_model_names=labels)) == set(labels)


def test_error_only_targeted_run_has_no_leaderboard_comparison() -> None:
    summary = pd.DataFrame(
        [{"dataset": "probe", "model": "Chemprop", "error": "training failed"}]
    )

    assert leaderboard_comparison_by_dataset(summary).empty
