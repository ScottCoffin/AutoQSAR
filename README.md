# AutoQSAR

AutoQSAR is a portable QSAR modeling and benchmarking workspace for molecular
property prediction from SMILES strings. It has two main entry points:

- `portable_colab_qsar_bundle/colab_qsar_tutorial.ipynb`: an interactive,
  widget-driven notebook for building QSAR models on built-in or user-supplied
  datasets.
- `portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py`: a command-line
  benchmark runner for comparing model families across curated ChemML, TDCommons,
  MoleculeNet, Polaris, PODUAM, and literature datasets.

The repository is designed to support both learning-scale use and full benchmark
runs. The notebook focuses on a guided workflow that can run in Google Colab or a
local Jupyter kernel. The benchmark runner focuses on repeatable model evaluation,
resume-safe long runs, leaderboard comparisons, and reusable output artifacts.

## What This Repository Does

AutoQSAR takes molecular tables with a SMILES column and a numeric target column,
then runs a complete modeling workflow:

1. Load an example dataset or user-provided CSV/XLSX file.
2. Select SMILES and target columns.
3. Assess missingness and apply a user-selected missing-value strategy.
4. Parse, standardize, and canonicalize molecules with RDKit.
5. Optionally collapse duplicate structures.
6. Generate molecular representations from RDKit and MapLight-style feature
   families.
7. Split data by random, scaffold, target-quartile, or predefined split logic.
8. Run train-only feature filtering and feature selection.
9. Train conventional ML, tuned ML, deep learning, graph, foundation-model, and
   ensemble workflows when their dependencies are available.
10. Compare train/test metrics and benchmark leaderboard references.
11. Save metrics, predictions, selected features, runtime diagnostics, and model
    comparison reports.
12. Predict new molecules and optionally apply UMAP and applicability-domain
    diagnostics in the notebook.

## Models This Repo Runs

AutoQSAR runs different model sets depending on whether the active dataset is a
continuous regression task or a strict binary 0/1 classification task. Optional
models are skipped when their packages, credentials, hardware, or dataset-size
guardrails are not satisfied.

### Conventional Regression Models

For continuous targets, the conventional model table can include:

- `ElasticNetCV`
- `SVR`
- `Random forest` (`RandomForestRegressor`)
- `Extra trees` (`ExtraTreesRegressor`)
- `HistGradientBoosting` (`HistGradientBoostingRegressor`)
- `Voting Regressor (KNN, SVM)` using `KNeighborsRegressor` and `SVR`
- `AdaBoost` (`AdaBoostRegressor`)
- `Tabular MLP` (`MLPRegressor`)
- `Tabular CNN` (`TabularCNNRegressor`, TensorFlow-backed when available)
- `XGBoost` (`XGBRegressor`, when `xgboost` is installed)
- `LightGBM` (`LGBMRegressor`, when `lightgbm` is installed)
- `CatBoost` (`CatBoostRegressor`, when `catboost` is installed)
- `MapLight CatBoost` or `MapLight CatBoost (Strict Parity)` on MapLight
  classic descriptors when CatBoost and MapLight-style features are available
- `TabPFNRegressor` when TabPFN is enabled and the dataset is within the
  configured train-row guardrail

### Conventional Classification Models

For strict binary 0/1 targets, the benchmark runner switches to classification
models and classification metrics. The conventional classification table can
include:

- `LogisticRegression`
- `SVC`
- `Random forest` (`RandomForestClassifier`)
- `Extra trees` (`ExtraTreesClassifier`)
- `HistGradientBoosting` (`HistGradientBoostingClassifier`)
- `Voting Classifier (KNN, SVM)` using `KNeighborsClassifier` and `SVC`
- `AdaBoost` (`AdaBoostClassifier`)
- `Tabular MLP` (`MLPClassifier`)
- `XGBoost` (`XGBClassifier`, when `xgboost` is installed)
- `LightGBM` (`LGBMClassifier`, when `lightgbm` is installed)
- `CatBoost` (`CatBoostClassifier`, when `catboost` is installed)
- `TabPFNClassifier` when TabPFN is enabled and the dataset is within the
  configured train-row guardrail

### GA-Tuned Models

The optional genetic-algorithm tuning stage currently tunes:

- `ElasticNet` for regression targets
- elastic-net-penalized `LogisticRegression` for binary classification targets
  under the `ElasticNet` tuned-model family
- `CatBoostRegressor` for regression targets
- `CatBoostClassifier` for binary classification targets

GA tuning is disabled by default unless `--ga-models` is set explicitly or
`--ga-models auto` finds prior evidence that a tuned family is worth rerunning.

### Deep, Graph, And Pretrained Molecular Models

The notebook and benchmark runner also support these optional model families:

- `ChemML MLP (PyTorch)`
- `ChemML MLP (TensorFlow)`
- `TabPFNRegressor` as a tabular foundation-model workflow in the notebook
- `MapLight + GNN (CatBoost)` and `MapLight + GNN (CatBoost, Strict Parity)`,
  using MapLight classic features plus pretrained GIN embeddings when the
  DGL/PyTorch stack is available
- `Chemprop v2 (D-MPNN)`
- `Chemprop v2 (CMPNN)`
- `Chemprop v2 (AttentiveFP)`
- Chemprop selected-descriptor variants, which add train-only selected tabular
  descriptors to the graph model
- optional Chemprop RDKit2D featurizer variants when `--run-chemprop-rdkit2d`
  is enabled
- `Uni-Mol V1` in the benchmark runner and notebook
- `Uni-Mol V2` in the notebook, with configurable model size and GPU-only
  execution in that workflow

### Fusion And Ensemble Models

After base models produce aligned train/test predictions, AutoQSAR can run:

- `CFA fusion` / `CFA combinatorial fusion`, using best-per-workflow model
  selection before bounded combinatorial fusion
- `Ensemble (OOF Stacking (RidgeCV))` for regression, with a logistic
  meta-model fallback for binary classification
- `Ensemble (Weighted average (inverse train RMSE))`; for classification this
  weights by the configured primary classification metric
- `Ensemble (Simple average)`

The notebook's applicability-domain section also fits internal diagnostic
models, including random-forest-based uncertainty/coverage helpers, but those
are guide-rail diagnostics rather than candidate QSAR predictors.

Some optional model families are expensive or dependency-sensitive. The workflow
is built so unavailable optional backends can be skipped while the rest of the
run continues.

## Repository Layout

| Path | Purpose |
|---|---|
| `portable_colab_qsar_bundle/colab_qsar_tutorial.ipynb` | Generated interactive notebook for Colab or local Jupyter. |
| `portable_colab_qsar_bundle/build_colab_qsar_tutorial.py` | Source of truth for the generated notebook. Edit this file, then regenerate the notebook. |
| `portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py` | CLI benchmark runner. |
| `portable_colab_qsar_bundle/qsar_workflow_core.py` | Shared feature, split, CFA, and QSAR helper code used by notebook and benchmarks. |
| `portable_colab_qsar_bundle/benchmark_registry.py` | Shared dataset registry for notebook examples and benchmark discovery. |
| `portable_colab_qsar_bundle/simple_applicability_domain.py` | Applicability-domain support used by notebook prediction workflows. |
| `portable_colab_qsar_bundle/example_prediction_smiles.csv` | Small bundled prediction input example. |
| `data/benchmark_dataset_catalog.csv` | Catalog of benchmark datasets, splits, metrics, local files, and leaderboard metadata. |
| `data/benchmark_leaderboards/` | Cached leaderboard reference tables. |
| `benchmark_results/` | Benchmark run outputs. |
| `model_cache/` | Persistent feature/model caches for expensive reusable artifacts. |
| `refs/` and `portable_colab_qsar_bundle/references/` | Reference papers and ChemML notebooks used to guide workflow design. |
| `test_data/` | Local development and example datasets. |
| `environment-*.yml`, `requirements-*.txt` | Conda and pip/uv environment definitions. |
| `setup_env.bat`, `setup_env.ps1` | Windows setup helpers. |

## Requirements

Minimum practical requirements:

- Windows, macOS, Linux, or Google Colab.
- Python 3.11.
- Conda or Miniconda for the local environment.
- At least 4 GB RAM for small examples; 32 GB or more is recommended for larger
  benchmark runs.
- An NVIDIA GPU is optional but recommended for some deep-learning backends.

The main workflow can run CPU-only. GPU availability mainly affects runtime and
whether some optional backends, such as local TabPFN and Uni-Mol, are practical.

## Installation

Miniconda is recommended over full Anaconda because it installs faster and uses
less disk space. On Windows, after installing Miniconda or Anaconda, open an
Anaconda PowerShell prompt and run:

```powershell
conda init powershell
```

If PowerShell blocks activation scripts, run this once in PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Automated Setup

From the repository root, run one of the setup scripts:

```powershell
.\setup_env.ps1
```

or:

```bat
setup_env.bat
```

The scripts prompt for CUDA, CPU-only, or custom PyTorch setup, create the conda
environment, install `uv`, and install the pinned Python packages.

### Manual CPU Setup

```powershell
conda env create -f environment-cpu.yml -n autoqsar-py311
conda run -n autoqsar-py311 pip install uv
conda run -n autoqsar-py311 uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cpu.txt
conda activate autoqsar-py311
```

### Manual CUDA Setup

Use this path only on systems with an NVIDIA GPU and compatible CUDA support:

```powershell
conda env create -f environment-cuda.yml -n autoqsar-py311
conda run -n autoqsar-py311 pip install uv
conda run -n autoqsar-py311 uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cuda.txt
conda activate autoqsar-py311
```

The `--index-strategy unsafe-best-match` flag lets `uv` resolve pinned packages
across PyPI and the PyTorch wheel index. In this repository, the practical risk
is low because the package versions are pinned and the configured indexes are
trusted public package indexes.

### Verify The Environment

```powershell
conda activate autoqsar-py311
@'
import sys
print(sys.version)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch check failed:", exc)

import rdkit
print("rdkit imported")
'@ | python -
```

## Using The Interactive Notebook

The notebook is the easiest way to run AutoQSAR on your own data.

1. Activate the environment:

   ```powershell
   conda activate autoqsar-py311
   ```

2. Start Jupyter:

   ```powershell
   jupyter lab
   ```

3. Open:

   ```text
   portable_colab_qsar_bundle/colab_qsar_tutorial.ipynb
   ```

4. Use the `AutoQSAR (py311)` kernel if it is available.

If the kernel is missing, create it:

```powershell
conda activate autoqsar-py311
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name autoqsar-py311 --display-name "AutoQSAR (py311)"
```

### Notebook Workflow

Run the notebook from top to bottom. The major sections are:

- `0`: install packages and initialize runtime state.
- `1`: load a built-in example, upload a dataset, or read a file path.
- `1B`: choose SMILES and target columns.
- `1C`: assess missingness and preprocess selected columns.
- `2`: preview curated molecules and generate molecular features.
- `3`: build a PCA/t-SNE chemical similarity map.
- `4`: split train/test data, select features, train conventional models, and
  optionally run GA tuning.
- `5`: run selected deep-learning workflows.
- `6`: run optional Uni-Mol and Chemprop graph-model workflows.
- `7`: build ensembles from available trained models.
- `8`: explain selected model behavior.
- `9`: predict new molecules, generate UMAP views, and run applicability-domain
  checks.
- `10`: review next steps.

For your own data, provide at least:

- one SMILES column
- one numeric target column

The notebook can also load built-in examples from ChemML, TDCommons, MoleculeNet
PhysChem, Polaris ADME, and PODUAM.

### Regenerating The Notebook

Do not hand-edit the notebook JSON for durable workflow changes. Edit the builder:

```text
portable_colab_qsar_bundle/build_colab_qsar_tutorial.py
```

Then regenerate:

```powershell
conda activate autoqsar-py311
python portable_colab_qsar_bundle/build_colab_qsar_tutorial.py
```

This updates:

```text
portable_colab_qsar_bundle/colab_qsar_tutorial.ipynb
```

## Running Benchmarks

The benchmark runner evaluates the same core workflow across a curated dataset
catalog and writes machine-readable artifacts.

Basic run:

```powershell
conda activate autoqsar-py311
$out = "benchmark_results\autoqsar_benchmark_$(Get-Date -Format yyyyMMdd_HHmmss)"
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --output-dir $out `
  --resume
```

Preview the planned datasets and configuration without fitting models:

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py --dry-run
```

Run one named built-in dataset:

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --dataset-name tdc_caco2_wang `
  --output-dir benchmark_results\tdc_caco2_wang_test `
  --resume
```

Run a local CSV instead of the default benchmark set:

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --dataset path\to\your_dataset.csv `
  --output-dir benchmark_results\local_dataset_run `
  --resume
```

Include an extra local CSV in addition to the default benchmark set:

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --include-local-csv path\to\your_dataset.csv `
  --output-dir benchmark_results\with_local_dataset `
  --resume
```

Refresh leaderboard reference artifacts without model training:

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --refresh-leaderboards-only `
  --output-dir benchmark_results\leaderboard_refresh
```

### Benchmark Profiles

The runner has two profiles:

- `cost_optimized` (default): disables historically low-value expensive variants
  by default.
- `full`: restores the broader model set.

Example:

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --benchmark-profile full `
  --output-dir benchmark_results\full_profile `
  --resume
```

### Useful Benchmark Controls

| Option | Purpose |
|---|---|
| `--dataset-name NAME` | Filter built-in benchmark datasets by ID. Repeat for multiple datasets. |
| `--dataset PATH` | Run only one or more local CSV datasets. Repeat for multiple local files. |
| `--include-local-csv PATH` | Add a local CSV to the default benchmark set. |
| `--row-limit N` | Deterministic row cap for smoke tests. |
| `--minimum-rows N` | Skip datasets smaller than this threshold. |
| `--target-transform auto/raw/log10` | Control target transform policy. |
| `--split-strategy target_quartiles/random/scaffold/predefined` | Requested split strategy. Benchmark datasets may override this with catalog-specific split intent. |
| `--test-fraction N` | Test-set fraction for generated splits. |
| `--ga-models ElasticNet,CatBoost` | Enable GA tuning for listed model families. Empty default disables GA tuning. |
| `--ga-models auto` | Enable GA only for models with prior evidence of value. |
| `--run-tabpfn/--no-run-tabpfn` | Enable or disable TabPFNRegressor. |
| `--run-chemprop-mpnn/--no-run-chemprop-mpnn` | Enable or disable Chemprop MPNN. |
| `--run-chemprop-attentivefp/--no-run-chemprop-attentivefp` | Enable or disable Chemprop AttentiveFP. |
| `--run-unimol-v1/--no-run-unimol-v1` | Override Uni-Mol V1 auto behavior. |
| `--run-cfa/--no-run-cfa` | Enable or disable CFA fusion. |
| `--run-ensemble/--no-run-ensemble` | Enable or disable standard ensembles. |
| `--compare-run-dir PATH` | Generate run-vs-run attribution against a previous run. |
| `--resume/--no-resume` | Resume compatible incomplete runs. |

The runner is designed for long runs. Use `--resume` and a stable `--output-dir`
so completed datasets and compatible intermediate artifacts can be reused.

## Datasets And Benchmark Registry

The shared benchmark registry lives in:

```text
portable_colab_qsar_bundle/benchmark_registry.py
```

This file is the source of truth for the built-in notebook examples and the
benchmark runner's default dataset discovery. It includes:

- ChemML bundled examples.
- TDCommons ADME and Tox tasks.
- MoleculeNet PhysChem datasets.
- Polaris ADME benchmark mirrors.
- PODUAM POD datasets.
- Expanded FreeSolv literature benchmark metadata.

The broader dataset catalog is written to:

```text
data/benchmark_dataset_catalog.csv
```

That catalog records dataset source, local availability, SMILES/target columns,
recommended split, recommended metric, target type hints, leaderboard references,
and local benchmark result status.

## Outputs

Each benchmark run writes a run directory under `benchmark_results/`. Important
top-level outputs include:

| File | Meaning |
|---|---|
| `run_config.json` | Full benchmark configuration and resolved options. |
| `summary_metrics.csv` | Deduplicated model metrics across datasets. |
| `predictions.csv` | Train/test predictions from completed model stages. |
| `test_rmse_pivot.csv` | Dataset-by-model RMSE pivot when regression RMSE columns are available. |
| `leaderboard_top10_reference.csv` | Captured leaderboard reference rows for the run. |
| `leaderboard_top10_reference.json` | JSON form of leaderboard references. |
| `leaderboard_comparison_by_dataset.csv` | Best model per dataset compared with leaderboard references. |
| `step_runtime_summary.csv` | Runtime diagnostics collected across datasets. |
| `model_value_report.csv` | Model contribution/value summary across datasets. |
| `model_zero_value_candidates.csv` | Models that appear to add little value in the current run. |
| `run_vs_run_attribution_summary.json` | Optional current-vs-reference run comparison summary. |

Each dataset subdirectory can include:

- `metrics.csv`
- `predictions.csv`
- `selected_features.csv`
- `selector_coefficients.csv`
- `dropped_duplicate_features.csv`
- `ga_history.csv`
- `cfa_candidate_table.csv`
- `cfa_base_strengths.csv`
- `ensemble_results*.csv`
- `ensemble_weights*.csv`
- `step_runtime.csv`
- backend-specific outputs such as Chemprop train/test files and predictions

## Caching

AutoQSAR uses caches to avoid repeating expensive work:

- `model_cache/feature_store_parquet`: persistent molecular feature store keyed
  by canonical SMILES and feature representation.
- `model_cache/benchmark_feature_matrix_cache`: shared benchmark feature-matrix
  cache.
- `model_cache/tuned_conventional_ml`: tuned conventional model cache.
- backend-specific cache directories inside benchmark output folders.

If a code change should alter results but a run appears unchanged, check whether
you are reusing an existing output directory or model cache. For a clean
comparison, use a fresh `--output-dir` or clear only the specific stale cache you
intend to invalidate.

## Optional Authentication

### TabPFN

`TabPFNRegressor` is enabled by default in benchmark runs. Authentication depends
on the backend:

| Backend | When used | Authentication |
|---|---|---|
| `tabpfn` local backend | GPU available | Hugging Face token plus one-time Prior Labs license flow. |
| `tabpfn_client` API backend | CPU-only fallback | Prior Labs API key. |

For local TabPFN model download, set a Hugging Face token:

```powershell
huggingface-cli login
```

or:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
[System.Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_token_here", "User")
```

For the API backend:

```powershell
$env:PRIORLABS_API_KEY = "your_key_here"
[System.Environment]::SetEnvironmentVariable("PRIORLABS_API_KEY", "your_key_here", "User")
```

If credentials are missing in an interactive terminal, the benchmark runner
prompts once before datasets are processed. Press Enter without a key to skip
TabPFN and continue with other models.

### External Dataset And Model Downloads

Some loaders download public datasets, benchmark references, pretrained weights,
or package sources. The TDCommons path uses the repository's Windows-safe PyTDC
source-load fallback when needed.

## Common Workflows

### Quick Smoke Test

```powershell
conda activate autoqsar-py311
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --dataset-name chemml_organic_density `
  --row-limit 100 `
  --no-run-tabpfn `
  --no-run-chemprop-mpnn `
  --no-run-chemprop-attentivefp `
  --no-run-maplight-gnn `
  --no-run-cfa `
  --no-run-ensemble `
  --output-dir benchmark_results\smoke_test `
  --resume
```

### Resume A Long Run And Rebuild Ensembles

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --output-dir benchmark_results\all_benchmarks_run `
  --resume `
  --revisit-completed-datasets `
  --rebuild-ensemble
```

### Compare Two Runs

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --output-dir benchmark_results\new_run `
  --compare-run-dir benchmark_results\previous_run `
  --resume
```

This writes run-vs-run split, config, error, and leaderboard comparability files
when enough matching data are available.

## Development Notes

- For notebook behavior, edit `build_colab_qsar_tutorial.py` and regenerate
  `colab_qsar_tutorial.ipynb`.
- For shared features, split logic, persistent feature-store behavior, and CFA
  logic, edit `qsar_workflow_core.py`.
- For built-in dataset availability, edit `benchmark_registry.py`.
- For benchmark-runner behavior, edit `run_autoqsar_ga_benchmarks.py`.
- Keep benchmark split intent dataset-specific. The runner supports fallback
  logic, but benchmark datasets should preserve their intended split strategy
  where possible.
- Strict binary 0/1 benchmark datasets are treated as classification tasks and
  should report classification metrics rather than regression-only summaries.
- CFA uses best-per-workflow inputs before combinatorial fusion, with budget
  guardrails to avoid uncontrolled candidate expansion.
- Benchmark output is intentionally quiet by default. Enable verbose backend
  details only when diagnosing a specific backend.

## Troubleshooting

### Jupyter Uses The Wrong Python

Create or reselect the `AutoQSAR (py311)` kernel:

```powershell
conda activate autoqsar-py311
python -m ipykernel install --user --name autoqsar-py311 --display-name "AutoQSAR (py311)"
```

### CUDA Or Torch Import Fails

Use the CPU environment if CUDA compatibility is uncertain:

```powershell
conda env create -f environment-cpu.yml -n autoqsar-py311
```

### Benchmark Results Do Not Change After A Code Edit

Use a fresh output directory first. If the issue is cache-specific, clear only
the relevant cache under `model_cache/` or the affected dataset subdirectory in
`benchmark_results/`.

### Optional Backend Fails

Disable the failing backend and rerun:

```powershell
python portable_colab_qsar_bundle\run_autoqsar_ga_benchmarks.py `
  --no-run-tabpfn `
  --no-run-chemprop-mpnn `
  --no-run-chemprop-attentivefp `
  --output-dir benchmark_results\without_optional_backend `
  --resume
```

### Notebook Still Shows Old Behavior After Regeneration

Restart the notebook kernel or reload the notebook from disk. The generated
notebook file and the live in-memory notebook/kernel can diverge during active
editing.

## License

See `LICENSE`.
