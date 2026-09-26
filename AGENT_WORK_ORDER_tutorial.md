# QSARena Tutorial Usability Work Order

Step 0 handoff for `QSARena_IDE_Agent_Work_Order_Tutorial_Usability_Additional_File.docx`.
This file is the durable source of truth for the Additional file 2 tutorial/usability effort.

Branch prepared for Step 0: `docs/tutorial-usability-plan`, based on
`origin/integration/main-work-order-merge`.

Date: 2026-09-25

## Locked Decisions

- Additional file 2 is a standalone guided install and usage tutorial.
- Additional file 1 remains the existing supplementary Tables S1-S8 and is not repurposed.
- Every run and batch report is emitted in both HTML and Markdown.
- The tutorial must not document behavior that code lacks.
- Every command, flag, and expected output shown in the tutorial must be exercised by an
  automated executable-docs test on a tiny fixture.
- Step 0 is planning only. Do not begin Part A implementation until this file is committed.

## Current Capability Map

### Package And Entry Points

- `pyproject.toml`
  - Console scripts:
    - `qsarena-benchmark = portable_colab_qsar_bundle.run_qsarena_benchmarks:main`
    - `qsarena-applicability-domain = portable_colab_qsar_bundle.simple_applicability_domain:main`
  - Packages shipped: `qsarena`, `portable_colab_qsar_bundle`.
  - Optional extras already cover `boosting`, `deep`, `graph`, `foundation`, `benchmarks`,
    `tdc`, `chemml`, `notebook`, `dev`, and `all`.

### Workflow Core

- `portable_colab_qsar_bundle/qsar_workflow_core.py`
  - Shared molecular feature generation lives in `build_feature_matrix_from_smiles(...)`.
  - This module is used by both the benchmark runner and the notebook builder.
- `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`
  - Current bundled feature families are listed in `DEFAULT_BENCHMARK_FEATURE_FAMILIES`:
    `morgan`, `ecfp6`, `fcfp6`, `layered`, `atom_pair`, `topological_torsion`,
    `rdk_path`, `maccs`, `rdkit`, `maplight`.
  - Input inference candidates are `SMILES_CANDIDATES` and `TARGET_CANDIDATES`.
  - Dataset representation is `DatasetSpec`.
  - Dataset result representation is `DatasetRunResult`.
  - Cleanup and target transform are handled in `canonicalize_frame(...)`.
  - Splitting is handled in `split_data(...)`.
  - Train-only feature selection is handled in `select_features(...)`.
  - Classification/regression task detection is scattered through `current_dataset_task_type(...)`
    and dataset metadata.

### CLI Runner

- `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`
  - CLI parser: `parse_args()`.
  - Main lifecycle: `main()`.
  - Single local CSV discovery: `discover_local_datasets(...)`.
  - Built-in benchmark discovery: `discover_default_example_datasets(...)`.
  - PFAS workbook discovery: `discover_pfas_aux_workbook_datasets(...)`.
  - Per-dataset execution: `run_dataset(...)`.
  - Resume planning: `build_resume_execution_plan(...)` and `print_resume_execution_plan(...)`.
  - Config signature: `benchmark_config_signature(...)`.
  - Stage 2/3 cache signature and I/O:
    `stage23_resume_signature(...)`, `load_stage23_resume_cache(...)`,
    `write_stage23_resume_cache(...)`.
  - Per-dataset status: `write_dataset_status(...)`.
  - Run manifest: `run_config.json` written in `main()`.
  - Environment manifest: `qsarena.provenance.write_environment_manifest(...)` called in `main()`.
  - Runtime artifacts: per-dataset `step_runtime.csv`, `step_runtime.json`, plus run-level
    `run_timing.json` and `step_runtime_summary.csv`.
  - Summary outputs: `summary_metrics.csv`, `predictions.csv`, `leaderboard_comparison_by_dataset.csv`,
    `test_rmse_pivot.csv`, `model_value_report.csv`, `run_vs_run_attribution_summary.json`.
  - Existing `--dry-run` prints a plan, resource information, backend status, datasets, and some
    skip/budget information, then exits.
  - Existing `--resume/--no-resume` is present. There is no `--fresh` alias yet.
  - Existing parallel dataset support is `--parallel-datasets`, but this runs discovered curated/local
    datasets rather than an arbitrary user batch manifest.

No `run_qsarena_ga_benchmarks.py` source was found in this branch during Step 0. GA is integrated
into `run_qsarena_benchmarks.py`.

### Notebook

- Source of truth: `portable_colab_qsar_bundle/build_colab_qsar_tutorial.py`.
- Generated notebook: `portable_colab_qsar_bundle/colab_qsar_tutorial.ipynb`.
- Current generated notebook headings include:
  - setup and workflow map
  - data input and column selection
  - missingness and preprocessing
  - feature generation
  - split and train-only feature selection
  - conventional models
  - GA tuning
  - deep workflows
  - Uni-Mol and Chemprop
  - ensembles
  - model explanation
  - prediction
  - UMAP and applicability domain
- The notebook uses `# @param` controls. It already has many user-facing decisions, but they are
  not generated from a shared `RunConfig` schema and are not guaranteed to match the CLI.
- Prior usability guidance remains relevant: inspect the generated `.ipynb` cell order and headings,
  not only the builder, because Colab execution flow is the user-facing product.

### Applicability Domain

- CLI entry point: `portable_colab_qsar_bundle/simple_applicability_domain.py:main`.
  - Key flags include `--train_csv`, `--smiles_col`, `--target_col`, `--query_smiles`,
    `--radius`, `--n_bits`, `--no_descriptors`, `--no_fingerprint`, `--knn_neighbors`,
    `--knn_quantile`, `--mahalanobis_quantile`, `--low_concern_support_ratio`, `--use_mastml`,
    and `--install_mastml_if_missing`.
- Library functions: `qsarena/applicability_domain.py`.
  - `standardization_ad(...)`
  - `knn_similarity_ad(...)`
  - `tanimoto_distance_matrix(...)`
- Reliability-support modules:
  - `qsarena/uncertainty.py`
  - `qsarena/reliability_study.py`
  - `qsarena/reliability_tables.py`
  - `qsarena/qmrf.py`

### Dataset Registry

- `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`
  - Built-in dataset discovery and metadata attachment are in
    `discover_default_example_datasets(...)`, `load_benchmark_catalog_metadata(...)`,
    `attach_leaderboard_summary(...)`, and related helpers.
- `data/benchmark_dataset_catalog.csv`
  - Curated benchmark metadata.
- `data/benchmark_leaderboards/*.csv`
  - Cached leaderboard/literature comparison inputs.
- `benchmark_registry.py`
  - Registry/catalog source listed in `AGENTS.md`; inspect before editing registry behavior.

### Caching, Artifacts, And Provenance

- Feature caches:
  - Persistent per-SMILES feature store configured by `--enable-persistent-feature-store`,
    `--reuse-persistent-feature-store`, `--persistent-feature-store-path`.
  - Shared feature-matrix cache configured by `--enable-shared-feature-matrix-cache`,
    `--reuse-shared-feature-matrix-cache`, `--shared-feature-matrix-cache-path`.
- Stage cache:
  - `stage23_resume_cache.pkl`, keyed by `stage23_resume_signature(...)`.
- Dataset artifacts:
  - `metrics.csv`
  - `predictions.csv`
  - `ga_history.csv`
  - `selected_features.csv`
  - `selector_coefficients.csv`
  - feature deduplication reports
  - CFA candidate tables
  - ensemble weights/results
  - `run_status.json`
  - `step_runtime.csv`
  - `step_runtime.json`
- Run artifacts:
  - `run_config.json`
  - `environment_manifest.json`
  - `run_timing.json`
  - `run_complete.json`
  - `summary_metrics.csv`
  - `leaderboard_comparison_by_dataset.csv`
  - `step_runtime_summary.csv`
  - `model_value_report.csv`
  - `run_vs_run_attribution_summary.json`
- Provenance module:
  - `qsarena/provenance.py`
  - `write_environment_manifest(...)`
  - `environment_manifest(...)`
  - `split_hash(...)`

### Reporting

- Existing machine-readable and CSV reporting is strong.
- Existing Markdown reporting exists for QMRF-style reports in `qsarena/qmrf.py`.
- Existing reliability table Markdown/LaTeX support lives in `qsarena/reliability_tables.py`.
- Missing: a general self-contained run/batch `report.html` and `report.md`.
- Missing: structured `run.log` plus JSON event stream at selectable verbosity.

### Tests

- Existing tests include:
  - `tests/unit/test_provenance.py`
  - `tests/unit/test_applicability_domain.py`
  - `tests/unit/test_qmrf.py`
  - `tests/unit/test_uncertainty.py`
  - `tests/unit/test_tdc22_multiseed.py`
  - `tests/integration/test_reliability_pipeline.py`
  - `tests/test_subprocess_encoding.py`
  - `tests/test_model_filtering.py`
  - `tests/test_prepare_chemprop_repair_run.py`
- Existing tiny fixtures:
  - `tests/fixtures/tdc_tiny/caco2_wang/*.csv`
  - `tests/fixtures/tdc_tiny/herg/*.csv`
- Missing:
  - `tests/docs/`
  - executable-docs tests for tutorial command blocks
  - end-to-end single-dataset fixture test for the public `qsarena-benchmark` command
  - arbitrary user batch-mode tests
  - report HTML/Markdown tests

## Gap Analysis

### F1 - Unified Per-Decision Option Surface: Partial

Existing:

- Many CLI flags are already present in `parse_args()`.
- The notebook exposes many widget controls with `# @param`.
- `run_config.json` records resolved CLI args after profile/resource/backend defaults.
- `benchmark_config_signature(...)` signs most current CLI options.
- `apply_benchmark_profile_defaults(...)` provides `cost_optimized` and `full` defaults.

Gaps:

- No single `RunConfig` dataclass or pydantic model exists.
- No `--config run.yaml` support exists.
- No shared schema drives CLI flags, notebook widgets, validation, docs, and manifest serialization.
- No generated `docs/options_reference.md`.
- CLI precedence is currently command-line defaults plus post-parse mutation, not
  `CLI > config file > defaults`.
- Several notebook controls have richer usability behavior than the CLI, but the mapping is manual.

Decision-group classification:

1. Input and columns: Partial.
   - Exists: `DatasetSpec`, `discover_local_datasets(...)`, `infer_column(...)`, notebook sections 1A/1B.
   - Missing: ID column, multi-target, explicit classification threshold, unified task-type config.
2. Standardization: Partial.
   - Exists: `canonicalize_frame(...)` drops invalid SMILES, canonicalizes with RDKit, handles missing targets.
   - Missing: first-class keep/drop invalid policy, salt stripping, charge normalization, tautomer normalization,
     and unified dedup controls for CLI.
3. Featurization: Partial.
   - Exists: `DEFAULT_BENCHMARK_FEATURE_FAMILIES`, `build_feature_matrix_from_smiles(...)`, feature caches.
   - Missing: first-class per-family config loaded from `RunConfig`; MapLight composite exposed as a documented
     option group rather than an implementation detail.
4. Splitting: Partial.
   - Exists: `split_data(...)`, `--split-strategy`, `--test-fraction`, `--random-seed`, `--cv-folds`.
   - Missing: shared config model, per-dataset override path for arbitrary batches.
5. Feature selection: Partial.
   - Exists: `select_features(...)`, selector flags, train-only selection, RF fallback on timeout.
   - Missing: `rf_fallback` as an explicit top-level selector mode; threshold options are scattered.
6. Model library: Partial.
   - Exists: cost profiles and many per-backend booleans.
   - Missing: generated, validated per-family/per-model option registry.
7. GA tuning: Partial.
   - Exists: `--ga-models`, `auto`, generation/population/cv/mutation knobs.
   - Missing: explicit `off|on|auto` mode and time budget option.
8. Deep/pretrained backends: Partial.
   - Exists: Chemprop, ChemML, Uni-Mol, MapLight+GNN, TabPFN flags and resource defaults.
   - Missing: unified backend config and GPU override policy shared with the notebook.
9. Fusion: Partial.
   - Exists: CFA flags, score/rank controls.
   - Missing: schema-generated docs and config mapping.
10. Ensembles: Partial.
   - Exists: `--ensemble-methods`, OOF stacking, inverse-RMSE averaging, correlation filtering,
     `--ensemble-member-selection-split train|test`.
   - Gap: work order wants member-selection metric `cv|test` defaulting to `cv`; current CLI uses
     `train|test` defaulting to `train`. Work order wants `exclude_negative_test_r2_members`
     default false; current `--ensemble-exclude-negative-test-r2-members` default is true.
11. Applicability domain: Partial.
   - Exists: standalone AD CLI and library functions.
   - Missing: integrated run config/report option `standardization|confidence|both|off`.
12. Evaluation: Partial.
   - Exists: metrics are computed and primary metrics recorded; leaderboard metric logic exists.
   - Missing: explicit reported-metrics list and primary-metric override in shared config.
13. Model-selection protocol: Partial.
   - Exists: test-selected and CV-selected analyses in manuscript/notebook artifacts.
   - Missing: simple `test|cv` run option with report-both-by-default behavior.
14. Outputs: Partial.
   - Exists: `--output-dir`, resume, manifests, CSV/JSON artifacts.
   - Missing: locked `html+md` report format, verbosity, `--fresh`, and a unified overwrite policy.
15. Execution: Partial.
   - Exists: single CSV via repeated `--dataset`, curated dataset filters, PFAS workbook, TDC multiseed,
     `--parallel-datasets`.
   - Missing: arbitrary batch manifest/directory/list interface with per-dataset overrides and aggregate
     success/failure accounting.

### F2 - Single-Dataset Mode: Partial

Existing:

- `--dataset PATH` accepts one or more CSV paths and bypasses default benchmark discovery.
- `discover_local_datasets(...)` infers SMILES and target columns.
- `run_dataset(...)` can produce model metrics, predictions, and per-dataset artifacts.

Gaps:

- No dedicated fixture-backed integration test proves the public `qsarena-benchmark --dataset` path.
- No guaranteed `report.html` and `report.md`.
- No documented one-command quickstart verified by executable-docs tests.

### F3 - Arbitrary Batch Mode: Partial/Missing

Existing:

- Repeated `--dataset` can run multiple local CSVs.
- `--include-local-csv` can add local CSVs to curated examples.
- Built-in curated benchmark discovery runs many datasets.
- `--parallel-datasets` can execute discovered datasets concurrently.
- Per-dataset artifacts are isolated by slugified dataset subdirectories.

Gaps:

- No `--batch` manifest CSV mode.
- No directory-of-CSVs batch mode.
- No explicit list file mode.
- No per-dataset override schema for columns, task, split, target transform, or model options.
- Batch should continue past a malformed dataset and record failure. Current behavior may abort on
  uncaught errors in `run_dataset(...)`.
- No aggregate batch report and no explicit no-cap dataset-count contract.

### F4 - Resume At Dataset And Stage Granularity: Partial

Existing:

- Completed datasets are reused by `load_completed_dataset_result(...)` and `run_dataset(...)`.
- `build_resume_execution_plan(...)` reports pending datasets and missing model stages.
- Stage 2/3 cache is keyed by dataset content, split, selector settings, feature settings, and other inputs
  through `stage23_resume_signature(...)`.
- `benchmark_config_signature(...)` is recorded in `run_config.json`.
- `--resume/--no-resume` exists.

Gaps:

- No `--fresh` alias.
- Atomic temp-then-rename writes are not consistently used. For example, `write_dataset_status(...)`,
  `run_config.json`, `run_timing.json`, and stage runtime outputs write directly.
- Resume validation is not uniformly config-signature-aware for every artifact type.
- No kill-and-resume test exists.
- No test proves that a config change invalidates only affected stages.

### F5 - Intelligent Feedback And Reporting: Partial/Missing

Existing:

- `--dry-run` prints planned output directory, resources, feature families, selector guardrail,
  GA/CFA/MapLight/TDC multiseed status, backend status, leaderboard references, and datasets.
- Progress messages include per-dataset and per-stage counts, elapsed time, and ETA.
- `run_timing.json`, `step_runtime.csv`, and `step_runtime.json` provide machine-readable progress traces.
- Backend skips/failures are surfaced in console output and metrics rows.

Gaps:

- `--dry-run` does not compute SMILES parse rate, dropped count, duplicate rate, class balance,
  inferred task type, or all requested guardrail warnings before long work.
- No structured `run.log` and JSON event stream at selectable verbosity.
- No general in-run warning collector.
- No self-contained `report.html` and `report.md`.
- No report plots for won-by-family, cost-vs-gap, or AD coverage.
- No "what to do next" report section.

### D1 - Additional File 2 Tutorial: Missing

Existing:

- README has installation, usage, artifacts, and domain sections.
- Submission package currently includes Additional file 1 only.
- `submission/README.md` documents the current supplementary build process.

Gaps:

- No `docs/tutorial.md`.
- No `submission/additional_file_2_qsarena_tutorial.tex` or PDF build path.
- Manuscript Availability and Additional-file list do not yet reference Additional file 2.
- Additional file 2 flowchart and command excerpts do not exist.

### D2 - Executable Docs: Missing

Existing:

- Pytest is configured in `pyproject.toml`.
- Tiny TDC fixtures exist under `tests/fixtures/tdc_tiny`.
- Unit/integration tests cover provenance, AD, reliability, subprocess encoding, and some runner helpers.

Gaps:

- No `tests/docs/test_tutorial_runs.py`.
- No command-block extractor for `docs/tutorial.md`.
- No test checking tutorial flags against `qsarena-benchmark --help`.
- No pipeline for regenerating tutorial log/plot excerpts from fixture runs.

## Implementation Plan

Follow the work-order order exactly: Step 0 -> F1 -> F2 -> F4 -> F3 -> F5 -> D1 -> D2.

### Step 0 - Planning File

Target files:

- `docs/AGENT_WORK_ORDER_tutorial.md`

Validation:

- `git diff --check`

Completion criteria:

- This file is committed on `docs/tutorial-usability-plan`.

### F1 - Unified RunConfig

Target files:

- `qsarena/config.py`
- `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`
- `portable_colab_qsar_bundle/build_colab_qsar_tutorial.py`
- `configs/run.example.yaml`
- `docs/options_reference.md`
- `tests/unit/test_run_config.py`
- `tests/integration/test_config_precedence.py`

Proposed public functions/classes:

- `@dataclass class RunConfig`
- `RunConfig.defaults() -> RunConfig`
- `RunConfig.from_yaml(path: str | Path) -> RunConfig`
- `RunConfig.from_cli_args(args: argparse.Namespace, cli_tokens: Sequence[str]) -> RunConfig`
- `RunConfig.to_manifest_dict() -> dict[str, Any]`
- `RunConfig.config_signature() -> str`
- `RunConfig.validate() -> None`
- `add_run_config_arguments(parser: argparse.ArgumentParser) -> None`
- `apply_run_config_to_namespace(config: RunConfig, args: argparse.Namespace) -> argparse.Namespace`
- `render_options_reference(config_cls: type[RunConfig]) -> str`

Notes:

- Prefer stdlib `dataclasses` plus a small YAML loader helper. Avoid adding pydantic unless the project
  explicitly accepts the dependency.
- Add `PyYAML` only if a dependency is acceptable. Otherwise support JSON and a tiny YAML subset is a risk;
  decide before implementation.
- Keep `benchmark_config_signature(...)` as a compatibility wrapper until all callers move to `RunConfig`.
- The notebook widgets should be generated or validated against the same schema. At minimum, add a schema
  consistency test that compares widget keys, CLI flags, config keys, and options docs.
- Resolve ensemble semantics explicitly:
  - config key should expose the work-order language, likely `member_selection_metric: cv|test`.
  - map `cv` to current `ensemble_member_selection_split=train` or rename carefully.
  - change `exclude_negative_test_r2_members` default to false only if existing manuscript/repro commands are
    kept reproducible through an explicit legacy setting.

Tests:

- `test_defaults_validate`
- `test_run_yaml_round_trips`
- `test_cli_overrides_yaml_over_defaults`
- `test_invalid_option_reports_field_and_allowed_values`
- `test_manifest_contains_resolved_config_and_signature`
- `test_options_reference_lists_all_15_groups`
- `test_notebook_widget_keys_match_run_config_schema`

### F2 - Single Dataset Verification

Target files:

- `tests/fixtures/tutorial/single_dataset.csv`
- `tests/integration/test_single_dataset_cli.py`
- `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`
- Later: `docs/tutorial.md`

Proposed tests:

- `test_qsarena_benchmark_single_csv_writes_model_metrics_predictions_and_reports`
- `test_single_dataset_command_uses_public_console_script`

Implementation notes:

- Fixture should be at most 50 molecules.
- Use cheap settings in test command:
  - `--dataset`
  - `--output-dir`
  - `--benchmark-profile cost_optimized`
  - `--row-limit` if needed
  - disable expensive deep/GPU/model families explicitly through `RunConfig`.
- Assert:
  - at least one successful model metric row
  - `metrics.csv`
  - `predictions.csv`
  - `run_config.json`
  - `environment_manifest.json`
  - `report.html`
  - `report.md`

### F4 - Resume Hardening

Target files:

- `qsarena/artifacts.py`
- `qsarena/resume.py`
- `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`
- `tests/integration/test_resume_cli.py`
- `tests/unit/test_atomic_artifacts.py`

Proposed functions:

- `atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None`
- `atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None`
- `atomic_write_csv(path: Path, frame: pd.DataFrame, **kwargs: Any) -> None`
- `class ResumeDecision`
- `resume_decision_for_artifact(path: Path, expected_signature: str, stage: str) -> ResumeDecision`
- `fresh_output_dir(output_dir: Path) -> None`

Tests:

- `test_resume_skips_completed_dataset`
- `test_resume_reuses_stage23_cache_when_signature_matches`
- `test_config_change_invalidates_stage23_cache`
- `test_fresh_ignores_existing_artifacts`
- `test_atomic_write_leaves_no_partial_file_on_exception`

Implementation notes:

- Add `--fresh` as a friendly alias for `--no-resume` plus explicit overwrite/rebuild behavior.
- Convert direct writes for manifests/status/runtime to atomic helpers.
- Keep current reproduction behavior available for legacy canonical runs.

### F3 - Arbitrary Batch Mode

Target files:

- `qsarena/batch.py`
- `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`
- `tests/fixtures/tutorial/batch_manifest.csv`
- `tests/fixtures/tutorial/batch_dir/*.csv`
- `tests/integration/test_batch_cli.py`

Proposed functions/classes:

- `@dataclass class BatchDatasetSpec`
- `load_batch_manifest(path: Path) -> list[BatchDatasetSpec]`
- `discover_batch_directory(path: Path, defaults: RunConfig) -> list[BatchDatasetSpec]`
- `batch_specs_to_dataset_specs(batch_specs: Sequence[BatchDatasetSpec]) -> list[DatasetSpec]`
- `write_batch_summary(output_dir: Path, results: Sequence[DatasetRunResult]) -> Path`

CLI:

- `--batch PATH`
- `--batch-mode manifest|directory|auto`
- `--batch-continue-on-error/--no-batch-continue-on-error`

Tests:

- `test_batch_manifest_three_datasets_one_malformed_records_failure`
- `test_batch_directory_runs_all_csvs`
- `test_batch_per_dataset_overrides_columns_and_split`
- `test_batch_has_no_artificial_dataset_count_limit`

Implementation notes:

- Batch manifest columns should include:
  `dataset_name`, `path`, `smiles_col`, `target_col`, `id_col`, `task`, `split`,
  `predefined_split_col`, `target_transform`, plus optional per-dataset config override columns.
- Continue-past-failure should wrap per-dataset discovery and execution, not just model training.

### F5 - Feedback, Logs, And Reports

Target files:

- `qsarena/preflight.py`
- `qsarena/reporting.py`
- `qsarena/run_events.py`
- `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`
- `tests/unit/test_preflight.py`
- `tests/unit/test_reporting.py`
- `tests/integration/test_run_reports.py`

Proposed functions/classes:

- `preflight_dataset(frame: pd.DataFrame, config: RunConfig) -> PreflightResult`
- `preflight_run(datasets: Sequence[DatasetSpec], config: RunConfig) -> RunPreflight`
- `write_run_log_event(output_dir: Path, event: Mapping[str, Any]) -> None`
- `collect_run_report_data(output_dir: Path) -> RunReportData`
- `render_markdown_report(data: RunReportData) -> str`
- `render_html_report(data: RunReportData) -> str`
- `write_run_reports(output_dir: Path) -> tuple[Path, Path]`

Tests:

- `test_preflight_smiles_parse_rate_and_drop_count`
- `test_preflight_class_balance_and_inferred_task`
- `test_preflight_duplicate_rate`
- `test_dry_run_writes_plan_and_runs_no_models`
- `test_report_writes_html_md_and_contains_config_summary`
- `test_json_events_are_valid_json_lines`

Implementation notes:

- Keep reporting dependency-light. Prefer stdlib HTML rendering unless a dependency is already accepted.
- HTML must be self-contained.
- Markdown should be useful in plain text and suitable for tutorial excerpts.
- Plots requested by the work order:
  - won-by-family
  - cost-vs-gap
  - AD coverage
- If AD coverage is unavailable for a normal run, report "not run" with a what-next action rather than
  silently omitting the section.

### D1 - Additional File 2 Tutorial

Target files:

- `docs/tutorial.md`
- `submission/additional_file_2_qsarena_tutorial.tex`
- `submission/additional_file_2_qsarena_tutorial.pdf`
- `submission/README.md`
- `submission/body.tex`
- `manuscript.md`
- `CHANGELOG.md`

Required sections:

1. Overview and entry-point decision guide.
2. Installation.
3. Quickstart single dataset.
4. Decision reference for all 15 option groups.
5. Batch mode.
6. Resume.
7. Understanding reports.
8. Applicability domain and reliability.
9. Reproducibility.
10. Troubleshooting and FAQ.

Build notes:

- Do not disturb `submission/additional_file_1.tex` or `submission/additional_file_1.pdf`.
- Decide whether PDF is generated through pandoc, LaTeX, or a small Python renderer before writing
  tutorial-specific build code.
- Tutorial command examples must be copied from or generated by the D2 executable-docs fixture run.

Tests/checks:

- `test_tutorial_sections_present`
- supplementary PDF build command documented in `submission/README.md`
- grep `manuscript.md` and `submission/body.tex` to confirm Additional file 2 references are in sync

### D2 - Executable Docs

Target files:

- `tests/docs/test_tutorial_runs.py`
- `tests/docs/tutorial_command_runner.py` or helper inside the test file
- `docs/tutorial.md`
- `tests/fixtures/tutorial/*`

Proposed functions:

- `extract_command_blocks(markdown_path: Path) -> list[CommandBlock]`
- `run_tutorial_command(block: CommandBlock, tmp_path: Path) -> CompletedProcess`
- `documented_flags(markdown_path: Path) -> set[str]`
- `help_flags(command: str) -> set[str]`

Tests:

- `test_all_tutorial_commands_run_on_fixture`
- `test_documented_qsarena_benchmark_flags_exist_in_help`
- `test_documented_applicability_domain_flags_exist_in_help`
- `test_tutorial_expected_output_excerpts_match_fixture_run`

Implementation notes:

- Mark truly slow/GPU commands as documented alternatives, not executed default commands.
- Command blocks intended for docs testing should carry a convention such as HTML comments or fenced info
  strings so setup snippets, shell comments, and non-executable examples are handled deliberately.

## Acceptance Checklist

- [x] `docs/AGENT_WORK_ORDER_tutorial.md` committed with gap analysis, per-task plan
  (files/functions/tests), locked decisions, and a status log (Step 0).
- [ ] `RunConfig` single source of truth; all 15 decision groups via CLI + notebook + `--config`;
  `docs/options_reference.md` generated; `run.yaml` loads (F1).
- [ ] Single-dataset run verified end-to-end (F2).
- [ ] Batch over manifest + directory for arbitrary N; isolates datasets; continues past failures;
  aggregate summary (F3).
- [ ] Resume verified at dataset + stage granularity, config-signature-aware, interrupt-safe;
  `--resume/--fresh` documented (F4).
- [ ] Preflight/`--dry-run`, progress+ETA, actionable warnings, and `report.html` and `report.md`
  present and tested (F5).
- [ ] Tutorial published as Additional file 2 (all 10 sections); Availability + Additional-file list
  updated; Additional file 1 intact (D1).
- [ ] Every tutorial command runs green in CI on the fixture; no documented flag missing from `--help` (D2).
- [ ] `pytest -q` green; `CHANGELOG.md` + `docs/AGENT_WORK_ORDER_tutorial.md` updated;
  supplementary builds cleanly.

## Running Status Log

- 2026-09-25: Step 0 started from
  `QSARena_IDE_Agent_Work_Order_Tutorial_Usability_Additional_File.docx`.
- 2026-09-25: Created isolated worktree at
  `C:\Users\SCOTT~1.COF\AppData\Local\Temp\QSARena_tutorial_plan` on branch
  `docs/tutorial-usability-plan`, based on `origin/integration/main-work-order-merge`.
- 2026-09-25: Examined package metadata, CLI runner, workflow core references, generated notebook
  headings, AD entry point, provenance/report modules, submission README, and current tests/fixtures.
- 2026-09-25: Wrote Step 0 gap analysis and implementation plan in this file. No Part A code was
  implemented in this commit.

## Open Coordination Notes

- The normal repo checkout is intentionally left on its existing branch because `main` is reported to be
  open/running in the Jetstream A100 IDE for Chemprop and Ensemble data work.
- This Step 0 branch is safe to review independently. Do not merge into `main` until the branch
  consolidation plan around `integration/main-work-order-merge` is settled.
- Before changing ensemble defaults, preserve a documented legacy mode for reproducing previously deposited
  runs.
- Before adding YAML support, decide whether `PyYAML` is acceptable as a package dependency or only a dev/doc
  dependency.
