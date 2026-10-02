"""A corrected dataset citation label must not change the stage 2/3 signature.

That signature keys the cached feature selections and every metrics.csv row's stage_config_signature,
so a display-label fix (the PODUAM misattribution, 2026-10-02) would otherwise force retraining.
"""

from __future__ import annotations

import argparse

import pandas as pd

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner


def _signature(source: str, monkeypatch) -> tuple[str, dict]:
    monkeypatch.setattr(runner, "stage23_args_payload", lambda args, spec: {})
    frame = pd.DataFrame({"canonical_smiles": ["CCO", "c1ccccc1"], "target": [0.1, 0.2]})
    spec = runner.DatasetSpec(name="poduam_pod_nc_std", source=source, frame=frame,
                              smiles_column="canonical_smiles", target_column="target")
    return runner.stage23_resume_signature(args=argparse.Namespace(), spec=spec, canonical_df=frame,
                                           input_meta={}, predefined_split=None)


def test_corrected_poduam_labels_keep_their_signature(monkeypatch) -> None:
    assert len(runner.SIGNATURE_SOURCE_LABEL_ALIASES) == 2
    for corrected, legacy in runner.SIGNATURE_SOURCE_LABEL_ALIASES.items():
        assert "von Borries" in corrected and "Aurisano" in legacy
        assert _signature(corrected, monkeypatch)[0] == _signature(legacy, monkeypatch)[0]


def test_other_labels_still_change_the_signature(monkeypatch) -> None:
    assert _signature("source A", monkeypatch)[0] != _signature("source B", monkeypatch)[0]


def test_cache_matcher_ignores_display_label_but_not_data() -> None:
    stored = {"dataset_name": "d", "dataset_source": "old label", "dataset_content_hash": "h1"}
    assert runner._stage23_payload_matches_ignoring_cache_location(stored, {**stored, "dataset_source": "new label"})
    assert not runner._stage23_payload_matches_ignoring_cache_location(stored, {**stored, "dataset_content_hash": "h2"})
