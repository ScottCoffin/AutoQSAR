from __future__ import annotations

import json

from qsarena.notebooks import NOTEBOOK_FILES, copy_notebooks
from tests.conftest import command_prefix


def test_qsarena_notebooks_console_script_is_public():
    prefix = command_prefix("qsarena-notebooks")
    assert prefix


def test_copy_notebooks_writes_generated_tutorial_artifacts(tmp_path):
    written = copy_notebooks(tmp_path)
    assert [path.name for path in written] == list(NOTEBOOK_FILES)
    for name in NOTEBOOK_FILES:
        assert (tmp_path / name).exists(), name

    colab = json.loads((tmp_path / "colab_qsar_tutorial.ipynb").read_text(encoding="utf-8"))
    local = json.loads((tmp_path / "local_qsar_tutorial.ipynb").read_text(encoding="utf-8"))
    assert colab["nbformat"] == 4
    assert local["nbformat"] == 4
    assert "9F. Export notebook results to an HTML report" in (tmp_path / "colab_qsar_tutorial.ipynb").read_text(encoding="utf-8")
    assert "9F. Export notebook results to an HTML report" in (tmp_path / "local_qsar_tutorial.ipynb").read_text(encoding="utf-8")


def test_copy_notebooks_respects_overwrite(tmp_path):
    copy_notebooks(tmp_path)
    target = tmp_path / "colab_qsar_tutorial.ipynb"
    target.write_text("sentinel", encoding="utf-8")
    copy_notebooks(tmp_path)
    assert target.read_text(encoding="utf-8") == "sentinel"
    copy_notebooks(tmp_path, overwrite=True)
    assert target.read_text(encoding="utf-8") != "sentinel"
