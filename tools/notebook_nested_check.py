"""Headless check of the local notebook on one CSV (regression or binary classification).

Covers 1A-1C, 2B, 4A-4E, 7A-7B, 9A and 9F, with the nested-selection path in 4B/4C/7A.

Builds a notebook from the generated local_qsar_tutorial.ipynb cells, sets widget values through injected
cells (set_local_form_values), runs 4C with outer selection and then nested selection, then 7A, and saves the
metrics tables plus the executed notebook into <cwd>/<out_dir_name>. pip installs are blocked in the kernel.

Usage (the benchmark env must be first on PATH, because the python3 kernelspec runs `python`):
    python tools/notebook_nested_check.py <csv with smiles + target> <target_column> <out_dir_name>
Build a CSV from the committed partitions
(data/meta_analysis/dataset_partitions.csv.gz: smiles, observed), e.g. freesolv_sampl, tdc_caco2_wang (regression)
or tdc_hia_hou, tdc_bbb_martins (classification).
2026-10-06 (final build): FreeSolv 103 s, Caco-2 203 s, HIA 131 s, BBB-Martins 834 s; all pass (nested CV engaged,
test metrics identical to the outer run, 4E skipped for classification, report written).
"""
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "portable_colab_qsar_bundle" / "local_qsar_tutorial.ipynb"
csv_path, target, tag = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
out_dir = Path.cwd() / tag
out_dir.mkdir(exist_ok=True)
# Setup looks for portable_colab_qsar_bundle/qsar_workflow_core.py in the working directory and otherwise downloads the
# published copy from GitHub, so put the local one there: the check must test this checkout's code.
(out_dir / "portable_colab_qsar_bundle").mkdir(exist_ok=True)
(out_dir / "portable_colab_qsar_bundle" / "__init__.py").write_text("", encoding="utf-8")
(out_dir / "portable_colab_qsar_bundle" / "qsar_workflow_core.py").write_bytes(
    (REPO / "portable_colab_qsar_bundle" / "qsar_workflow_core.py").read_bytes()
)

src = nbformat.read(SRC, as_version=4)
cells = src.cells


def code(text):
    return nbformat.v4.new_code_cell(text)


def form_id(block):
    # Form ids are random per notebook build, so read them from the widget-control cells by block label.
    pattern = re.compile(r"ensure_local_form_display\('" + re.escape(block) + r"\. [^']*', '([0-9a-f]+)'")
    for cell in cells:
        match = pattern.search(cell.source)
        if match:
            return match.group(1)
    raise KeyError(f"no widget controls for block {block}")


def setter(block, values):
    # Set the widgets, then fail loudly if any value did not take (a silent miss runs the defaults instead).
    fid = form_id(block)
    return code(
        f"set_local_form_values({fid!r}, {values!r})\n"
        f"_widgets = LOCAL_FORM_STATE[{fid!r}]['widget_map']\n"
        f"_missed = {{k: v for k, v in {values!r}.items() if _widgets[k].value != v}}\n"
        f"assert not _missed, ('widget values not applied', {block!r}, _missed)\n"
        f"print('set', {block!r}, {fid!r})"
    )


GUARD = code(
    "import subprocess as _sp\n"
    "_cc, _run = _sp.check_call, _sp.run\n"
    "def _blocked(cmd):\n"
    "    return isinstance(cmd, (list, tuple)) and 'pip' in [str(c) for c in cmd] and 'install' in [str(c) for c in cmd]\n"
    "def _check_call(cmd, *a, **k):\n"
    "    if _blocked(cmd):\n"
    "        print('[test-guard] blocked', cmd); return 0\n"
    "    return _cc(cmd, *a, **k)\n"
    "def _run_(cmd, *a, **k):\n"
    "    if _blocked(cmd):\n"
    "        print('[test-guard] blocked', cmd); return _sp.CompletedProcess(cmd, 1, '', 'blocked by test guard')\n"
    "    return _run(cmd, *a, **k)\n"
    "_sp.check_call, _sp.run = _check_call, _run_\n"
)


def save(label, key):
    return code(
        "import pandas as _pd\n"
        f"_v = STATE.get({key!r})\n"
        f"_p = r'{out_dir}' + r'\\{label}.csv'\n"
        "if isinstance(_v, _pd.DataFrame):\n"
        "    _v.to_csv(_p, index=False); print('saved', _p, _v.shape)\n"
        "else:\n"
        "    print('not a frame:', type(_v))\n"
    )


NO_CACHE_4C = {"enable_conventional_model_cache": False, "reuse_conventional_cached_models": False,
               "run_tabular_cnn": False, "run_lightgbm": False}
seq = [
    GUARD, cells[5],
    cells[13], setter("1A", {"data_source": "File path", "dataset_file_path": str(csv_path)}), cells[14],
    cells[16], setter("1B", {"smiles_column": "smiles", "target_column": target}), cells[17],
    cells[20], setter("1C", {"missing_value_strategy": "ignore_row"}), cells[21],
    cells[28], cells[29],
    cells[37], cells[38],
    cells[41], setter("4A.5", {"enable_feature_selector_cache": False, "reuse_feature_selector_cache": False}),
    cells[42], cells[44],
    cells[49], setter("4C", {**NO_CACHE_4C, "nested_selection_cv": False}), cells[50], save("4C_outer", "traditional_results"),
    setter("4C", {**NO_CACHE_4C, "nested_selection_cv": True}), cells[50], save("4C_nested", "traditional_results"),
    code("print('nested fold columns cached:', len(STATE.get('nested_fold_columns', {})),"
         " 'train-only selector refit:', STATE.get('train_only_selector_refit') is not None)"),
    cells[105], setter("7A", {"include_unimol": False, "run_cfa_ensemble": False}), cells[106],
    save("7A_ensemble", "ensemble_results"),
    code("print('ENSEMBLE NOTES:', STATE.get('ensemble_member_filter_notes'))"),
    # Downstream blocks: plots, a regression-only block (4E: skipped for classification), prediction and the report.
    cells[52], cells[55], cells[56], cells[108],
    cells[120], cells[121], save("9A_predictions", "latest_prediction_results"),
    cells[137], cells[138],
    code("print('TASK TYPE:', STATE.get('task_type'), '| class map:', STATE.get('class_label_map'))"),
]
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(c.source) if c.cell_type == "code" else c for c in seq]
nb.metadata = src.metadata
start = time.time()
client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(out_dir)}},
                        allow_errors=False)
try:
    client.execute()
    status = "OK"
except Exception as exc:  # keep the partial notebook for diagnosis
    status = f"FAILED: {type(exc).__name__}: {str(exc)[-1500:]}"
nbformat.write(nb, out_dir / "executed.ipynb")
(out_dir / "status.txt").write_text(f"{status}\nelapsed {time.time() - start:.0f}s\n", encoding="utf-8")
print(status, f"{time.time() - start:.0f}s")
