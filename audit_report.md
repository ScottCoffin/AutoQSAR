# AutoQSAR Container Audit Report

**Branch:** feat/containerize  
**Audit date:** 2026-06-18  
**Audited by:** automated audit pass prior to containerization  

---

## 1.1 Full Dependency Surface

### GPU / CUDA-dependent packages

| Package | Version (cuda req) | Where imported | Notes |
|---|---|---|---|
| `torch` | `2.5.1+cu121` | `run_autoqsar_ga_benchmarks.py` lines 319, 1157, 4854, 5299, 5386–5387, 6050, 6052 | Lazy imports inside try/if blocks; GPU detection via `torch.cuda.is_available()` |
| `chemprop` | `2.2.3` | Spawned as subprocess via `chemprop train`/`chemprop predict` CLI | v2 line; Chemprop v2 uses PyTorch internally |
| `dgl` | `2.2.1` | Line 5384 (`import dgl`) | MapLight+GNN graph encoder |
| `dgllife` | `0.3.2` | Line 5385 (`import dgllife`) | GIN pretrained model loader |
| `tabpfn` | `7.1.1` | Lines 291–307 | Tabular foundation model; requires torch |
| `tabpfn-client` | `>=0.2.8` (unpinned in cuda req) | Line 291 | API fallback for TabPFN; still requires network |
| `tabpfn-common-utils` | `>=0.2.20` (unpinned) | Transitive dep of tabpfn | Needs exact pin — see §2 |
| `unimol-tools` | `0.1.5` | Line 6119 (`from unimol_tools import MolPredict, MolTrain`) | 3D pretrained model; auto-gated to GPU |
| `torchvision` | `0.20.1+cu121` | Transitive via torch ecosystem | Not directly imported in source |
| `torchaudio` | `2.5.1+cu121` | Transitive | Not directly imported |
| `lightning` / `pytorch-lightning` | `2.6.1` | Transitive via chemprop | |
| `torchdata` | `0.9.0` | Transitive | CUDA req only |

**GPU-optional (CPU fallback exists):**  
`catboost==1.2.10`, `xgboost==3.2.0` — both have GPU tree methods but run CPU-only with no special flag. `lightgbm` is imported in the code (line 284) but **missing from all requirements and environment files** — this is a dependency gap (see §1.4 action item).

### CPU-only scientific stack

| Package | Version (cpu req) | Role |
|---|---|---|
| `rdkit` | `2026.3.1` | Molecular standardization, all fingerprint families, descriptors |
| `scikit-learn` | `1.6.1` | ElasticNetCV selector, RF, SVM, histGBT, ensemble meta-models, CV splitters |
| `numpy` | `1.26.4` (cpu) | Numeric arrays throughout |
| `pandas` | `2.3.3` | Data frames, catalog I/O |
| `scipy` | `1.17.1` | Stats utilities |
| `catboost` | `1.2.10` | CatBoost models (CPU mode when no GPU) |
| `xgboost` | `3.2.0` | XGBoost models (CPU mode) |
| `lightgbm` | **UNLISTED** | LightGBM models — import at line 284 wraps try/except, so missing silently disables it |
| `cfanalysis` | `0.1.13` | CFA combinatorial fusion analysis |
| `pyarrow` | `23.0.1` | Parquet feature store backend |
| `pyarrow`/`fastparquet` | — | Feature cache Parquet writes |
| `pytdc` (bundled) | `1.1.15` | TDC dataset loader (bundled in `.cache/`) |
| `descriptastorus` | `2.8.0` | Descriptor utilities |
| `mhfp` | `1.9.6` | MHFP fingerprint |

---

## 1.2 Entry Points and Per-Dataset Stage Runner

### Primary entry point
`portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py`

**CLI invocation:**
```
python portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py \
  [--dataset-name <NAME>] \
  [--output-dir <PATH>] \
  [--random-seed 13] \
  [--chemprop-random-seed 42] \
  [--benchmark-profile cost_optimized|full] \
  [--device ...implicit via torch.cuda.is_available()...]
  ... (130+ additional flags)
```

**Per-dataset runner:**  
Function `run_dataset(spec: DatasetSpec, output_dir: Path, args: argparse.Namespace, dataset_position, dataset_total)` at **line 6968**.  
Called from `main()` via the dataset loop; orchestrates: feature generation → split → train-only ElasticNetCV selection → conventional models → optional GA → deep workflows (ChemML/Chemprop/MapLight+GNN/Uni-Mol) → optional CFA fusion → ensemble → artifact writes.

**`run_config.json` consumption:**  
Written to `<output_dir>/run_config.json` at line 9960, recording all resolved args plus the config signature, git hash, and profile. Not read back to continue a run — args are re-parsed from CLI on each invocation; the config file is used only for resume-matching (comparing `config_signature` fields) and for reproducibility archiving.

**Root path resolution:**  
```python
root = Path(__file__).resolve().parents[1]   # line 9713
```
All relative cache and data paths are computed from this root. Inside the container the script will live at `/opt/autoqsar/portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py`, making `root = /opt/autoqsar`.

---

## 1.3 Hardcoded Paths, Seeds, and Device Assumptions

| Location | File:Line | Current value | Required action |
|---|---|---|---|
| `root` resolution | `run_autoqsar_ga_benchmarks.py:9713` | `Path(__file__).resolve().parents[1]` | Correct as-is — resolves to `/opt/autoqsar` in container |
| Feature matrix cache | `run_autoqsar_ga_benchmarks.py:1374` | `root / "model_cache" / "benchmark_feature_matrix_cache"` | Bind-mount `$SCRATCH/autoqsar_in` to `/opt/autoqsar/model_cache` **or** pass `--shared-feature-matrix-cache-path` |
| Feature store | `run_autoqsar_ga_benchmarks.py:9714-9715` | `root / "model_cache" / "feature_store_parquet"` | Same — override via `--persistent-feature-store-path` |
| MapLight GNN pretrained dir | `run_autoqsar_ga_benchmarks.py:5354` | `root / "model_cache" / "maplight_gnn_pretrained"` | Must be in bind-mounted input volume at container startup |
| TDC split cache | `run_autoqsar_ga_benchmarks.py:2220` | `data_root / "_autoqsar_cache" / "tdc_single_pred"` | `data_root` is the `data/` directory; bind-mount brings it in |
| `--random-seed` default | `run_autoqsar_ga_benchmarks.py:9281` | `13` | Parameterized via CLI; `run_one.py` will pass `--random-seed $SEED` |
| `--chemprop-random-seed` default | `run_autoqsar_ga_benchmarks.py:9556` | `42` | Parameterized; `run_one.py` passes `--chemprop-random-seed $SEED` |
| GPU detection | `run_autoqsar_ga_benchmarks.py:321,6052` | `torch.cuda.is_available()` | Correct — `run_one.py --device auto` reads this; no hardcoded `cuda:0` |
| Uni-Mol auto-gate | `run_autoqsar_ga_benchmarks.py:9706-9712` | Auto-enabled when GPU detected | Correct; `--no-run-unimol-v1` disables on CPU |
| `.env` load | `run_autoqsar_ga_benchmarks.py:94` | `Path.cwd() / ".env"` or `root / ".env"` | Do not bake `.env` into image; inject env vars at runtime |
| `sys.path.insert(0, root)` | `run_autoqsar_ga_benchmarks.py:244` | Adds repo root to path | Correct for container where source lives at `/opt/autoqsar` |
| No absolute `C:\Users\...` paths found | — | — | Confirmed: no Windows absolute paths in production code |

---

## 1.4 Git State and Action Items

**Current commit:** `46bcd3d reorg and draft manuscript`  
**Working tree:** clean (0 modified files as of audit date 2026-06-18)  

### Dependency gaps identified
1. **`lightgbm` missing from all requirements/environment files.** Imported at line 284 inside a try/except (silent fail), so LightGBM models are silently skipped if not installed. The container requirements must add `lightgbm` with a pinned version compatible with CUDA 12.1.  
   *Recommended:* `lightgbm==4.5.0` (CUDA 12 support, Python 3.11 wheel available).

2. **`tabpfn-client` and `tabpfn-common-utils` are unpinned** in `requirements-cuda.txt` (lines with `>=` constraints). These must be pinned for reproducible container builds.

3. **`pywin32`, `pywinpty`, `win32-setctime`** appear in the requirements files — these are Windows-only and must be excluded from the Linux container requirements.

4. **`torch==2.3.0`** in `requirements-cpu.txt` vs `torch==2.5.1+cu121` in `requirements-cuda.txt` — the container targets CUDA 12.1 so uses the CUDA req file as base.

5. **`numpy` version mismatch:** cpu req pins `numpy==1.26.4`; cuda req says `numpy>=2.0,<2.5`. The CUDA lockfile must resolve this to a single version; current torch 2.5.1 supports numpy 2.x, so pin `numpy==2.2.6` (latest stable in the 2.x series as of build date).

### Pre-build action required
Before building a release image, ensure the working tree is clean and tagged (enforced by the `make image` target added in Step 6). The git SHA will be embedded in all image labels.

---

## BUILD_PROVENANCE.md stub

This audit creates `BUILD_PROVENANCE.md` as a stub; it will be auto-populated by the `make image` target.

```
Audit-time commit: 46bcd3d
Audit-time branch: feat/containerize
Audit-time tree-clean: true
```

---

*End of audit report. Proceeding to file creation per the containerization brief.*
