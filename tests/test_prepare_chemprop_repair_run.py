import json

import pandas as pd

from portable_colab_qsar_bundle.prepare_chemprop_repair_run import seed_repair_run


def test_seed_repair_run_copies_resume_files_but_not_model_directories(tmp_path) -> None:
    source = tmp_path / "canonical"
    dataset = source / "example"
    model_dir = dataset / "chemprop_v2" / "model"
    model_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "example", "model": "Chemprop v2 (D-MPNN, ensemble=3)", "error": None},
            {"dataset": "example", "model": "Chemprop v2 (CMPNN, ensemble=3)", "error": "failed"},
        ]
    ).to_csv(dataset / "metrics.csv", index=False)
    pd.DataFrame([{"model": "Random forest"}]).to_csv(dataset / "predictions.csv", index=False)
    (dataset / "run_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (dataset / "stage23_resume_cache.pkl").write_bytes(b"cache")
    (model_dir / "checkpoint.ckpt").write_bytes(b"model")

    destination = tmp_path / "repair"
    summary = seed_repair_run(source, destination)

    assert summary == {
        "datasets": 1,
        "chemprop_models": 2,
        "successful_chemprop_pairs": 1,
        "missing_chemprop_pairs": 1,
    }
    assert (destination / "example" / "metrics.csv").exists()
    assert (destination / "example" / "predictions.csv").exists()
    assert (destination / "example" / "stage23_resume_cache.pkl").exists()
    assert not (destination / "example" / "chemprop_v2").exists()
