"""
qsarena.config — ``RunConfig``, the single source of truth for every workflow decision.

Every user-facing decision QSARena makes is one field of a section dataclass below. The field's
metadata (``_opt``) records its decision group (1-15), help text, allowed values, the
``qsarena-benchmark`` flag and argparse destination it maps to, and whether it may be overridden
per dataset in batch mode. Everything else is derived from that one declaration:

  * validation with actionable errors                     (``RunConfig.from_dict`` / ``validate``)
  * ``--config run.yaml`` loading, precedence CLI > file > defaults
                                                           (``RunConfig.to_arg_values``)
  * the resolved config written into the run manifest     (``RunConfig.from_namespace``)
  * ``docs/options_reference.md``                         (``render_options_reference``)
  * ``configs/run.example.yaml``                          (``render_example_yaml``)
  * the notebook widget mapping                           (``NOTEBOOK_WIDGETS``)

Defaults reproduce the benchmark runner's long-standing behaviour exactly; options that did not
exist before (salt stripping, deduplication, ...) default to "off". ``None`` means "decided by the
cost profile or by hardware detection" and is resolved to a concrete value in the manifest.

This module is stdlib-only (PyYAML is used when installed, JSON otherwise) so the Colab notebook can
load it without installing the full package.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "ConfigError",
    "GROUPS",
    "RunConfig",
    "OptionSpec",
    "iter_options",
    "option_by_key",
    "load_run_config",
    "add_run_config_arguments",
    "regroup_parser_help",
    "cli_provided_dests",
    "model_family",
    "MODEL_FAMILIES",
    "FEATURE_FAMILIES",
    "NOTEBOOK_WIDGETS",
    "notebook_widget_values",
    "run_config_from_notebook_values",
    "render_options_reference",
    "render_example_yaml",
    "per_dataset_arg_overrides",
]

#: The fifteen decision groups of the tutorial work order, in order.
GROUPS: dict[int, str] = {
    1: "Input & columns",
    2: "Standardization",
    3: "Featurization",
    4: "Splitting",
    5: "Feature selection",
    6: "Model library",
    7: "GA tuning",
    8: "Deep / pretrained backends",
    9: "Fusion",
    10: "Ensembles",
    11: "Applicability domain",
    12: "Evaluation",
    13: "Model-selection protocol",
    14: "Outputs, reporting & caching",
    15: "Execution",
}

#: The ten molecular feature families implemented in ``qsar_workflow_core``. ``maplight`` (the
#: MapLight-classic composite) is controlled separately by ``features.maplight_classic``.
FEATURE_FAMILIES: tuple[str, ...] = (
    "morgan",
    "ecfp6",
    "fcfp6",
    "layered",
    "atom_pair",
    "topological_torsion",
    "rdk_path",
    "maccs",
    "rdkit",
)

#: Model families used by ``models.enable_families`` and by the report's won-by-family plot.
MODEL_FAMILIES: dict[str, str] = {
    "conventional_ml": "scikit-learn models: ElasticNetCV/LogisticRegression, SVR/SVC, random forest, extra trees, "
    "HistGradientBoosting, KNN+SVM voting, AdaBoost, tabular MLP (and their GA-tuned versions)",
    "gradient_boosting": "XGBoost, LightGBM, CatBoost and MapLight CatBoost (needs qsarena[boosting])",
    "deep_tabular": "ChemML MLP (PyTorch/TensorFlow), tabular CNN (TensorFlow) and TabPFN (needs qsarena[deep] / "
    "qsarena[foundation])",
    "graph_nn": "Chemprop v2 graph variants (needs qsarena[graph])",
    "pretrained_3d": "Uni-Mol V1/V2 pretrained 3D models (needs qsarena[foundation]; V2 needs a GPU)",
    "maplight_gnn": "MapLight + GNN (CatBoost on MapLight features plus GIN embeddings; needs dgl/dgllife)",
    "fusion": "CFA combinatorial fusion",
    "ensemble": "OOF stacking / weighted / simple-average ensembles",
}

_CONVENTIONAL_MODEL_PREFIXES = (
    "elasticnet",
    "logisticregression",
    "svr",
    "svc",
    "random forest",
    "extra trees",
    "histgradientboosting",
    "voting",
    "adaboost",
    "tabular mlp",
)


def model_family(model_name: Any) -> str:
    """Map a model label as written to ``metrics.csv`` onto one of :data:`MODEL_FAMILIES`."""
    text = str(model_name or "").strip()
    lower = text.lower()
    if lower == "ensemble" or lower.startswith("ensemble ("):
        return "ensemble"
    if lower.startswith("cfa"):
        return "fusion"
    if lower.startswith("chemprop"):
        return "graph_nn"
    if lower.startswith("uni-mol"):
        return "pretrained_3d"
    if lower.startswith("maplight + gnn"):
        return "maplight_gnn"
    if lower.startswith(("chemml", "tabular cnn", "tabpfn")):
        return "deep_tabular"
    if lower.startswith(("xgboost", "lightgbm", "catboost", "maplight catboost")):
        return "gradient_boosting"
    return "conventional_ml"


class ConfigError(ValueError):
    """An invalid configuration value. The message names the key, the value and what is allowed."""


# ---------------------------------------------------------------------------------------------
# Field declaration helper
# ---------------------------------------------------------------------------------------------

_MISSING = dataclasses.MISSING


def _opt(
    default: Any = _MISSING,
    *,
    group: int,
    help: str,
    kind: str,
    choices: Sequence[Any] | None = None,
    aliases: Mapping[str, Any] | None = None,
    cli: str | None = None,
    dest: str | None = None,
    nullable: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    per_dataset: bool = False,
    signature: bool = True,
    null_means: str = "",
    default_factory: Callable[[], Any] | None = None,
) -> Any:
    """Declare one option. ``kind`` is bool|int|float|str|choice|tristate|list|str_or_list|int_or_list|float_pair."""
    metadata = {
        "qsarena_option": True,
        "group": int(group),
        "help": str(help),
        "kind": str(kind),
        "choices": tuple(choices) if choices is not None else None,
        "aliases": dict(aliases or {}),
        "cli": cli,
        "dest": dest,
        "nullable": bool(nullable),
        "minimum": minimum,
        "maximum": maximum,
        "per_dataset": bool(per_dataset),
        "signature": bool(signature),
        "null_means": str(null_means),
    }
    if default_factory is not None:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


# ---------------------------------------------------------------------------------------------
# Sections (Appendix A layout)
# ---------------------------------------------------------------------------------------------


@dataclass
class RunSection:
    mode: str = _opt(
        "auto", group=15, kind="choice", choices=("auto", "single", "batch", "benchmark"), cli="--run-mode",
        dest="run_mode",
        help="single = one or more CSVs given by input.path; batch = batch.source; benchmark = the curated "
        "benchmark collection; auto picks batch if batch.source is set, else single if input.path is set, "
        "else benchmark.",
    )
    output_dir: str | None = _opt(
        None, group=14, kind="str", nullable=True, cli="--output-dir", dest="output_dir", signature=False,
        null_means="benchmark_results/qsarena_benchmark_<timestamp>",
        help="Directory that receives every artifact and the report.",
    )
    resume: bool = _opt(
        True, group=14, kind="bool", cli="--resume / --no-resume", dest="resume", signature=False,
        help="Reuse completed datasets and stages already in output_dir when their config signature matches.",
    )
    overwrite: bool = _opt(
        False, group=14, kind="bool", cli="--fresh", dest="fresh", signature=False,
        help="Ignore everything in output_dir and recompute. The old directory is renamed to "
        "<name>_superseded_<timestamp>, never deleted.",
    )
    verbosity: str = _opt(
        "normal", group=14, kind="choice", choices=("quiet", "normal", "verbose", "debug"), cli="--verbosity",
        dest="verbosity", signature=False,
        help="Console detail and the minimum level written to events.jsonl. run.log always keeps the full "
        "console transcript.",
    )
    dry_run: bool = _opt(
        False, group=15, kind="bool", cli="--dry-run", dest="dry_run", signature=False,
        help="Run preflight checks, print the resolved plan and a cost estimate, write dry_run_plan.{json,md}, "
        "then exit without fitting any model.",
    )
    n_jobs: int = _opt(
        0, group=15, kind="int", cli="--n-jobs", dest="n_jobs", signature=False,
        help="CPU workers for parallel-capable estimators. 0 or negative uses every detected core.",
    )
    parallel_datasets: int = _opt(
        1, group=15, kind="int", minimum=1, cli="--parallel-datasets", dest="parallel_datasets", signature=False,
        help="Datasets processed concurrently (process pool). n_jobs is divided between workers.",
    )
    dataset_names: list[str] = _opt(
        group=15, kind="list", cli="--dataset-name", dest="dataset_name", default_factory=list,
        help="Benchmark mode only: restrict the curated collection to these dataset names "
        "(list them with --dry-run).",
    )


@dataclass
class InputSection:
    path: Any = _opt(
        None, group=1, kind="str_or_list", nullable=True, cli="--dataset", dest="dataset",
        help="CSV file (or list of CSV files) for single mode. Each file becomes one dataset named after it.",
    )
    smiles_col: str | None = _opt(
        None, group=1, kind="str", nullable=True, cli="--smiles-col", dest="smiles_column", per_dataset=True,
        null_means="auto-detect (QSAR_READY_SMILES, canonical_smiles, SMILES, smiles, Smiles)",
        help="Column holding the SMILES strings.",
    )
    target_col: Any = _opt(
        None, group=1, kind="str_or_list", nullable=True, cli="--target-col", dest="target_columns",
        per_dataset=True, null_means="auto-detect (TARGET, target, Target, ...)",
        help="Target column. A list runs one dataset per target (named <file>__<target>).",
    )
    id_col: str | None = _opt(
        None, group=1, kind="str", nullable=True, cli="--id-col", dest="id_column", per_dataset=True,
        help="Optional identifier column, copied into the per-dataset predictions.csv.",
    )
    task: str = _opt(
        "auto", group=1, kind="choice", choices=("auto", "regression", "classification"), cli="--task",
        dest="task_type", per_dataset=True,
        help="auto treats a target with exactly two values as classification and anything else as regression.",
    )
    classification_threshold: float | None = _opt(
        None, group=1, kind="float", nullable=True, cli="--classification-threshold",
        dest="classification_threshold", per_dataset=True,
        help="Binarize a continuous target: value >= threshold becomes 1, otherwise 0. Implies task: "
        "classification.",
    )
    target_transform: str = _opt(
        "auto", group=1, kind="choice", choices=("auto", "raw", "log10"), cli="--target-transform",
        dest="target_transform", per_dataset=True,
        help="auto keeps curated benchmark datasets on their raw scale and log10-transforms a user regression "
        "target when every value is positive. Classification targets are never transformed.",
    )
    minimum_rows: int = _opt(
        20, group=1, kind="int", minimum=2, cli="--minimum-rows", dest="minimum_rows", per_dataset=True,
        help="Skip a dataset with fewer valid rows than this after cleanup (small-dataset guardrail).",
    )
    row_limit: int = _opt(
        0, group=1, kind="int", minimum=0, cli="--row-limit", dest="row_limit", per_dataset=True,
        help="Deterministic random subsample of this many rows (0 = use every row). Meant for smoke tests.",
    )


@dataclass
class StandardizeSection:
    drop_unparseable: bool = _opt(
        True, group=2, kind="bool", cli="--drop-unparseable / --no-drop-unparseable",
        dest="drop_unparseable_smiles", per_dataset=True,
        help="true drops rows RDKit cannot parse and reports how many. false stops the dataset with an error "
        "listing the unparseable rows, so nothing is dropped silently.",
    )
    strip_salts: bool = _opt(
        False, group=2, kind="bool", cli="--strip-salts / --no-strip-salts", dest="strip_salts", per_dataset=True,
        help="Keep only the largest organic fragment (RDKit LargestFragmentChooser).",
    )
    normalize_charges: bool = _opt(
        False, group=2, kind="bool", cli="--normalize-charges / --no-normalize-charges", dest="normalize_charges",
        per_dataset=True, help="Neutralize charges where possible (RDKit Uncharger).",
    )
    normalize_tautomers: bool = _opt(
        False, group=2, kind="bool", cli="--normalize-tautomers / --no-normalize-tautomers",
        dest="normalize_tautomers", per_dataset=True,
        help="Replace each molecule with RDKit's canonical tautomer. Slower, and can change the drawn structure.",
    )
    deduplicate: str = _opt(
        "none", group=2, kind="choice", choices=("none", "exact", "canonical_smiles"), cli="--deduplicate",
        dest="deduplicate", per_dataset=True,
        help="none keeps every row; exact drops rows repeating both the input SMILES and the target; "
        "canonical_smiles merges rows with the same canonical structure (mean target for regression, "
        "majority label for classification).",
    )


@dataclass
class FeaturesSection:
    families: list[str] = _opt(
        group=3, kind="list", choices=FEATURE_FAMILIES + ("maplight",), aliases={"avalon": "maplight", "erg": "maplight"},
        cli="--feature-families", dest="feature_families", per_dataset=True,
        default_factory=lambda: list(FEATURE_FAMILIES),
        help="Molecular feature families concatenated into the model matrix. 'maplight' (the MapLight-classic "
        "composite) is normally switched by features.maplight_classic; 'avalon' and 'erg' exist only inside "
        "that composite and are accepted as aliases for it.",
    )
    maplight_classic: bool = _opt(
        True, group=3, kind="bool", cli="--maplight-classic / --no-maplight-classic", dest="maplight_classic",
        per_dataset=True,
        help="Add the MapLight-classic composite (Morgan counts + Avalon counts + ErG + RDKit descriptor panel).",
    )
    fingerprint_bits: int = _opt(
        1024, group=3, kind="int", minimum=64, cli="--fingerprint-bits", dest="fingerprint_bits", per_dataset=True,
        help="Bit length of the hashed fingerprint families.",
    )
    cache: bool = _opt(
        True, group=3, kind="bool", cli="--enable-shared-feature-matrix-cache / --reuse-shared-feature-matrix-cache",
        dest=None, signature=False,
        help="Cache whole feature matrices (content-addressed by SMILES, families and bits) and reuse them "
        "across datasets and runs.",
    )
    persistent_store: bool = _opt(
        True, group=3, kind="bool", cli="--enable-persistent-feature-store / --reuse-persistent-feature-store",
        dest=None, signature=False,
        help="Per-SMILES Parquet feature store so only new molecules are featurized.",
    )


@dataclass
class SplitSection:
    strategy: str = _opt(
        "target_quartile", group=4, kind="choice", choices=("target_quartile", "random", "scaffold", "predefined"),
        aliases={"target_quartiles": "target_quartile", "stratified": "target_quartile"}, cli="--split-strategy",
        dest="split_strategy", per_dataset=True,
        help="target_quartile stratifies the random split on target quartiles (falls back to random when that "
        "is impossible); scaffold keeps Bemis-Murcko scaffolds together; predefined reads "
        "split.predefined_split_col. Curated benchmark datasets keep their published split.",
    )
    test_fraction: float = _opt(
        0.2, group=4, kind="float", minimum=0.05, maximum=0.5, cli="--test-fraction", dest="test_fraction",
        per_dataset=True, help="Held-out test fraction.",
    )
    seed: int = _opt(
        13, group=4, kind="int", cli="--random-seed", dest="random_seed", per_dataset=True,
        help="Seed for the split, CV folds and every seeded model.",
    )
    cv_folds: int = _opt(
        5, group=4, kind="int", minimum=2, cli="--cv-folds", dest="cv_folds", per_dataset=True,
        help="Cross-validation folds on the training split (CV metrics, OOF stacking, CV selection).",
    )
    predefined_split_col: str | None = _opt(
        None, group=4, kind="str", nullable=True, cli="--predefined-split-col", dest="predefined_split_column",
        per_dataset=True,
        help="Column whose values are train/test (or training/holdout); used when strategy is predefined.",
    )


@dataclass
class FeatureSelectionSection:
    method: str = _opt(
        "elasticnetcv", group=5, kind="choice", choices=("elasticnetcv", "rf_fallback", "none"),
        aliases={"elasticnet_cv": "elasticnetcv", "rf_importance": "rf_fallback", "random_forest": "rf_fallback"},
        cli="--selector-method", dest="selector_method", per_dataset=True,
        help="Train-only selection. elasticnetcv falls back to random-forest importance on timeout; rf_fallback "
        "uses random-forest importance directly; none keeps every column.",
    )
    variance_threshold: float = _opt(
        1e-8, group=5, kind="float", minimum=0.0, cli="--variance-threshold", dest="dedup_variance_threshold",
        per_dataset=True, help="Drop columns whose training variance is at or below this value.",
    )
    binary_prevalence_range: list[float] = _opt(
        group=5, kind="float_pair", cli="--binary-prevalence-range", dest="binary_prevalence_range",
        per_dataset=True, default_factory=lambda: [0.005, 0.995],
        help="Keep a 0/1 column only if its training prevalence lies inside [low, high].",
    )
    drop_duplicate_columns: bool = _opt(
        True, group=5, kind="bool", cli="--drop-duplicate-columns / --no-drop-duplicate-columns",
        dest="drop_duplicate_feature_columns", per_dataset=True,
        help="Drop feature columns identical to an earlier column on the training rows.",
    )
    max_selected_features: int = _opt(
        0, group=5, kind="int", minimum=0, cli="--max-selected-features", dest="max_selected_features",
        per_dataset=True, help="0 caps the selection at ceil(10% of training rows); a positive value overrides.",
    )
    auto_rf_by_dataset_size: bool | None = _opt(
        None, group=5, kind="bool", nullable=True,
        cli="--selector-auto-rf-by-dataset-size / --no-selector-auto-rf-by-dataset-size",
        dest="selector_auto_rf_by_dataset_size", per_dataset=True,
        null_means="profile default (on for cost_optimized/quick, off for full)",
        help="Switch to random-forest importance up front when ElasticNetCV is predicted to exceed its time limit.",
    )
    elasticnet_timeout_seconds: float = _opt(
        7200.0, group=5, kind="float", minimum=1.0, cli="--selector-elasticnet-timeout-seconds",
        dest="selector_elasticnet_timeout_seconds", per_dataset=True,
        help="Wall-clock limit for the ElasticNetCV selector before falling back to random forest.",
    )


def _enable_families_default() -> dict[str, bool]:
    return {name: True for name in MODEL_FAMILIES}


@dataclass
class ModelsSection:
    profile: str = _opt(
        "cost_optimized", group=6, kind="choice", choices=("cost_optimized", "full", "quick"),
        cli="--benchmark-profile", dest="benchmark_profile",
        help="cost_optimized (default) drops historically low-value expensive variants; full restores them; "
        "quick keeps only the scikit-learn families plus fusion and ensembles (minutes on a laptop).",
    )
    enable_families: dict[str, bool] = _opt(
        group=6, kind="family_map", cli="--disable-model-families", dest="disabled_model_families",
        per_dataset=True, default_factory=_enable_families_default,
        help="Master switch per model family: " + "; ".join(f"{k} = {v}" for k, v in MODEL_FAMILIES.items()) + ".",
    )
    disable_models: list[str] = _opt(
        group=6, kind="list", cli="--disable-model", dest="disable_models", per_dataset=True, default_factory=list,
        help='Model labels to skip, exactly as they appear in metrics.csv (e.g. "Tabular CNN").',
    )
    only_models: list[str] = _opt(
        group=6, kind="list", cli="--only-model-names", dest="only_model_names", per_dataset=True,
        default_factory=list,
        help="If non-empty, run only these model labels (plus the ensemble when 'Ensemble' is listed).",
    )


@dataclass
class GATuningSection:
    mode: str = _opt(
        "off", group=7, kind="choice", choices=("off", "on", "auto"), cli="--ga-models off|on|auto", dest=None,
        per_dataset=True,
        help="off skips GA tuning; on tunes every estimator listed; auto tunes only the estimators that won "
        "or improved in the most recent comparable run.",
    )
    estimators: list[str] = _opt(
        group=7, kind="list", choices=("elastic_net", "catboost"), aliases={"elasticnet": "elastic_net"},
        cli="--ga-estimators", dest="ga_estimators", per_dataset=True, default_factory=lambda: ["elastic_net", "catboost"],
        help="Estimators tuned when mode is on (classification tunes elastic-net logistic regression).",
    )
    generations: int = _opt(
        12, group=7, kind="int", minimum=1, cli="--ga-generations", dest="ga_generations", per_dataset=True,
        help="GA generations.",
    )
    population_size: int = _opt(
        16, group=7, kind="int", minimum=2, cli="--ga-population-size", dest="ga_population_size", per_dataset=True,
        help="Individuals per generation.",
    )
    time_budget_min: float | None = _opt(
        None, group=7, kind="float", nullable=True, minimum=0.0, cli="--ga-time-budget-minutes",
        dest="ga_time_budget_minutes", per_dataset=True, null_means="no limit",
        help="Stop starting new generations once this many minutes have elapsed for one estimator.",
    )
    max_configs: int | None = _opt(
        None, group=7, kind="int", nullable=True, minimum=1, cli="--ga-max-configs", dest="ga_max_configs",
        per_dataset=True, null_means="no limit",
        help="Stop after this many distinct hyperparameter configurations have been cross-validated.",
    )


@dataclass
class ChempropSection:
    variants: list[str] | None = _opt(
        None, group=8, kind="list", nullable=True,
        choices=("dmpnn", "dmpnn_rdkit2d", "selected_features", "cmpnn", "attentivefp"),
        aliases={"dmpnn_selected": "selected_features", "mpnn": "dmpnn"}, cli="--run-chemprop-*", dest=None,
        per_dataset=True,
        null_means="profile default (cost_optimized: attentivefp + selected_features; full: all five)",
        help="Chemprop v2 variants. An empty list switches Chemprop off.",
    )
    epochs: int | None = _opt(
        None, group=8, kind="int", nullable=True, minimum=1, cli="--chemprop-epochs", dest="chemprop_epochs",
        per_dataset=True, null_means="profile default (15; full: 40)", help="Training epochs.",
    )
    ensemble_size: int | None = _opt(
        None, group=8, kind="int", nullable=True, minimum=1, cli="--chemprop-ensemble-size",
        dest="chemprop_ensemble_size", per_dataset=True, null_means="profile default (1; full: 3)",
        help="Independently initialised Chemprop models averaged per variant.",
    )
    batch_size: int = _opt(
        32, group=8, kind="int", minimum=1, cli="--chemprop-batch-size", dest="chemprop_batch_size",
        per_dataset=True, help="Mini-batch size.",
    )
    seed: int = _opt(
        42, group=8, kind="int", cli="--chemprop-random-seed", dest="chemprop_random_seed", per_dataset=True,
        help="Chemprop seed.",
    )


@dataclass
class UniMolSection:
    v1: str = _opt(
        "auto", group=8, kind="tristate", cli="--run-unimol-v1 / --no-run-unimol-v1", dest="run_unimol_v1",
        per_dataset=True, help="auto runs Uni-Mol V1 only when a GPU is detected.",
    )
    v2: str = _opt(
        "auto", group=8, kind="tristate", cli="--run-unimol-v2 / --no-run-unimol-v2", dest="run_unimol_v2",
        per_dataset=True, help="auto runs Uni-Mol V2 only when a GPU is detected (V2 always needs a GPU).",
    )
    v2_size: str = _opt(
        "84m", group=8, kind="choice", choices=("84m", "164m", "310m"), cli="--unimol-model-size",
        dest="unimol_model_size", per_dataset=True, help="Uni-Mol V2 checkpoint size.",
    )
    epochs: int | None = _opt(
        None, group=8, kind="int", nullable=True, minimum=1, cli="--unimol-epochs", dest="unimol_epochs",
        per_dataset=True, null_means="10 on CPU, 20 when a GPU is detected", help="Fine-tuning epochs.",
    )
    lr: float = _opt(
        1e-4, group=8, kind="float", minimum=0.0, cli="--unimol-learning-rate", dest="unimol_learning_rate",
        per_dataset=True, help="Learning rate.",
    )
    batch_size: int | None = _opt(
        None, group=8, kind="int", nullable=True, minimum=1, cli="--unimol-batch-size", dest="unimol_batch_size",
        per_dataset=True, null_means="32 on CPU; 32/64/128 by detected GPU memory", help="Mini-batch size.",
    )
    early_stopping_patience: int = _opt(
        5, group=8, kind="int", minimum=1, cli="--unimol-early-stopping", dest="unimol_early_stopping",
        per_dataset=True, help="Epochs without improvement before stopping.",
    )


@dataclass
class ChemMLSection:
    pytorch: bool = _opt(
        True, group=8, kind="bool", cli="--run-chemml-pytorch / --no-run-chemml-pytorch", dest="run_chemml_pytorch",
        per_dataset=True, help="ChemML-style dense MLP on the selected descriptors (PyTorch).",
    )
    tensorflow: bool | None = _opt(
        None, group=8, kind="bool", nullable=True, cli="--run-chemml-tensorflow / --no-run-chemml-tensorflow",
        dest="run_chemml_tensorflow", per_dataset=True, null_means="profile default (off; full: on)",
        help="The same MLP in TensorFlow.",
    )
    epochs: int = _opt(
        80, group=8, kind="int", minimum=1, cli="--chemml-training-epochs", dest="chemml_training_epochs",
        per_dataset=True, help="Training epochs.",
    )


@dataclass
class DeepSection:
    use_gpu: str = _opt(
        "auto", group=8, kind="tristate", cli="--use-gpu", dest="gpu_policy", signature=False,
        help="auto detects a CUDA GPU; false hides any GPU (CPU only); true treats a GPU as present and warns "
        "if none is detected.",
    )
    chemprop: ChempropSection = field(default_factory=ChempropSection)
    unimol: UniMolSection = field(default_factory=UniMolSection)
    chemml: ChemMLSection = field(default_factory=ChemMLSection)
    tabpfn: bool | None = _opt(
        None, group=8, kind="bool", nullable=True, cli="--run-tabpfn / --no-run-tabpfn", dest="run_tabpfn",
        per_dataset=True, null_means="profile default (on; quick: off)",
        help="TabPFN tabular foundation model (local package on GPU, else the Prior Labs API client).",
    )
    cnn: bool | None = _opt(
        None, group=8, kind="bool", nullable=True, cli="--run-cnn / --no-run-cnn", dest="run_cnn",
        per_dataset=True, null_means="profile default (on; quick: off)",
        help="1-D CNN on the selected descriptors (needs TensorFlow).",
    )


@dataclass
class FusionSection:
    cfa_score: bool = _opt(
        True, group=9, kind="bool", cli="--run-cfa / --no-run-cfa", dest="run_cfa", per_dataset=True,
        help="Run CFA combinatorial fusion (score combinations) over the best model of each workflow.",
    )
    cfa_rank: bool = _opt(
        True, group=9, kind="bool", cli="--cfa-include-rank-combinations / --no-cfa-include-rank-combinations",
        dest="cfa_include_rank_combinations", per_dataset=True,
        help="Also try CFA rank combinations (AC/WCP/WCDS). Has no effect when cfa_score is false.",
    )
    optimize_metric: str = _opt(
        "mae", group=9, kind="choice",
        choices=("mae", "rmse", "roc_auc", "auprc", "balanced_accuracy", "mcc", "accuracy"),
        cli="--cfa-optimize-metric", dest="cfa_optimize_metric", per_dataset=True,
        help="Training-side metric used to choose the CFA candidate (classification uses the primary metric "
        "when a regression metric is given).",
    )


@dataclass
class EnsembleSection:
    oof_stacking: bool = _opt(
        True, group=10, kind="bool", cli="--ensemble-methods", dest=None, per_dataset=True,
        help="RidgeCV stacker fitted on the members' out-of-fold training predictions.",
    )
    inverse_rmse_average: bool = _opt(
        True, group=10, kind="bool", cli="--ensemble-methods", dest=None, per_dataset=True,
        help="Average weighted by inverse member error (out-of-fold under member_selection_metric oof).",
    )
    simple_average: bool = _opt(
        False, group=10, kind="bool", cli="--ensemble-methods", dest=None, per_dataset=True,
        help="Unweighted average of the members.",
    )
    member_selection_metric: str = _opt(
        "oof", group=10, kind="choice", choices=("oof", "cv", "test"), aliases={"train": "cv"},
        cli="--ensemble-member-selection-split", dest="ensemble_member_selection_split", per_dataset=True,
        help="Which predictions drive ensemble membership, weights and the stacking meta-model. oof refits "
        "each member on K folds of the training split and uses its out-of-fold predictions: leakage-free "
        "and not biased toward models that memorise the training set. cv (alias train) uses in-sample "
        "training predictions and favours overfit members; it only reproduces the "
        "qsarena_benchmark_chemprop_fixed run. test reproduces the originally deposited benchmark run "
        "and is optimistically biased.",
    )
    oof_folds: int = _opt(
        5, group=10, kind="int", minimum=2, cli="--ensemble-oof-folds", dest="ensemble_oof_folds",
        per_dataset=True,
        help="Folds for the out-of-fold member predictions (member_selection_metric oof). Same fold "
        "geometry as cross-validation.",
    )
    oof_scope: str = _opt(
        "all", group=10, kind="choice", choices=("all", "cpu"), cli="--ensemble-oof-scope",
        dest="ensemble_oof_scope", per_dataset=True,
        help="Which members may be refitted per fold for out-of-fold predictions. Uni-Mol reuses its saved "
        "internal-fold predictions (cv.data) and is not refitted. all also refits Chemprop per fold on the GPU; "
        "cpu refits only CPU models, and Chemprop is then left out of the ensemble.",
    )
    oof_allow_api_refits: bool = _opt(
        False, group=10, kind="bool",
        cli="--ensemble-oof-allow-api-refits / --no-ensemble-oof-allow-api-refits",
        dest="ensemble_oof_allow_api_refits", per_dataset=True,
        help="Allow out-of-fold refits of members that call a metered remote API (TabPFN via the Prior Labs "
        "client; K extra fits per dataset, billed as credits). Off: such members are left out of the "
        "ensemble unless they already have out-of-fold predictions.",
    )
    oof_source_run: str | None = _opt(
        None, group=10, kind="str", nullable=True, cli="--ensemble-oof-source-run",
        dest="ensemble_oof_source_run", signature=False, null_means="look only in this run",
        help="Earlier run directory searched for saved Uni-Mol model folders (cv.data) when building "
        "out-of-fold predictions.",
    )
    exclude_negative_test_r2_members: bool = _opt(
        True, group=10, kind="bool",
        cli="--ensemble-exclude-negative-test-r2-members / --no-ensemble-exclude-negative-test-r2-members",
        dest="ensemble_exclude_negative_test_r2_members", per_dataset=True,
        help="Drop members with negative R2 on the member-selection split (train when member_selection_metric "
        "is cv, so the default is leakage-free).",
    )
    drop_correlated_members: bool = _opt(
        True, group=10, kind="bool",
        cli="--ensemble-drop-highly-correlated-members / --no-ensemble-drop-highly-correlated-members",
        dest="ensemble_drop_highly_correlated_members", per_dataset=True,
        help="Drop one member of each pair whose training predictions are nearly identical.",
    )
    max_member_correlation: float = _opt(
        0.995, group=10, kind="float", minimum=0.0, maximum=1.0, cli="--ensemble-max-train-correlation",
        dest="ensemble_max_train_correlation", per_dataset=True,
        help="Correlation above which two members count as redundant.",
    )
    stacking_cv_folds: int = _opt(
        5, group=10, kind="int", minimum=2, cli="--ensemble-stacking-cv-folds", dest="ensemble_stacking_cv_folds",
        per_dataset=True, help="Folds for the out-of-fold stacking predictions.",
    )


@dataclass
class ApplicabilityDomainSection:
    method: str = _opt(
        "both", group=11, kind="choice", choices=("standardization", "confidence", "both", "off"),
        cli="--ad-method", dest="ad_method", per_dataset=True,
        help="Per-test-molecule domain flags written to applicability_domain.csv. standardization = Roy et al. "
        "(2015) descriptor-range rule on the selected features; confidence = predicted-probability "
        "confidence |2p-1| of the selected model for classification, and k-nearest-neighbour Tanimoto "
        "similarity to the training set for regression. Diagnostics only: no metric changes.",
    )
    confidence_threshold: float = _opt(
        0.5, group=11, kind="float", minimum=0.0, maximum=1.0, cli="--ad-confidence-threshold",
        dest="ad_confidence_threshold", per_dataset=True,
        help="Classification: a prediction is in-domain when |2p-1| >= this value (0.5 means p <= 0.25 or "
        "p >= 0.75).",
    )
    knn_quantile: float = _opt(
        0.95, group=11, kind="float", minimum=0.5, maximum=0.999, cli="--ad-knn-quantile",
        dest="ad_knn_quantile", per_dataset=True,
        help="Regression: in-domain when the mean Tanimoto distance to the 5 nearest training molecules is at "
        "most this quantile of the training set's own leave-one-out distances.",
    )


_REGRESSION_METRICS = ("rmse", "mae", "r2", "spearman")
_CLASSIFICATION_METRICS = ("roc_auc", "auprc", "balanced_accuracy", "mcc")
_PRIMARY_METRICS = (
    "auto", "rmse", "mae", "r2", "spearman", "pearson",
    "roc_auc", "auprc", "balanced_accuracy", "mcc", "accuracy",
)


@dataclass
class EvaluationSection:
    regression_metrics: list[str] = _opt(
        group=12, kind="list", choices=_REGRESSION_METRICS, cli="--report-regression-metrics",
        dest="report_regression_metrics", signature=False, default_factory=lambda: list(_REGRESSION_METRICS),
        help="Metrics shown in the report tables for regression datasets. metrics.csv always contains all of them.",
    )
    classification_metrics: list[str] = _opt(
        group=12, kind="list", choices=_CLASSIFICATION_METRICS, aliases={"auroc": "roc_auc"},
        cli="--report-classification-metrics", dest="report_classification_metrics", signature=False,
        default_factory=lambda: list(_CLASSIFICATION_METRICS),
        help="Metrics shown in the report tables for classification datasets.",
    )
    primary_metric: str = _opt(
        "auto", group=12, kind="choice", choices=_PRIMARY_METRICS, aliases={"auroc": "roc_auc"},
        cli="--primary-metric", dest="primary_metric_override", per_dataset=True,
        help="Metric used to rank models, drive GA tuning and pick the best model. auto = the benchmark's "
        "leaderboard metric, else rmse (regression) or roc_auc (classification).",
    )


@dataclass
class SelectionSection:
    protocol: str = _opt(
        "both", group=13, kind="choice", choices=("test", "cv", "both"), cli="--selection-protocol",
        dest="selection_protocol", signature=False,
        help="How the report names each dataset's best model. test = best held-out score (optimistic: the "
        "test set chooses); cv = best cross-validated training score (honest; only models with CV metrics "
        "are eligible); both = report both side by side.",
    )


@dataclass
class ReportSection:
    formats: list[str] = _opt(
        group=14, kind="list", choices=("html", "md"), cli="(fixed)", dest="report_formats", signature=False,
        default_factory=lambda: ["html", "md"],
        help="Locked: every run and batch writes both report.html and report.md.",
    )
    include_plots: bool = _opt(
        True, group=14, kind="bool", cli="--report-plots / --no-report-plots", dest="report_include_plots",
        signature=False, help="Won-by-family, cost-vs-gap and AD-coverage plots (SVG, no plotting library needed).",
    )
    what_next: bool = _opt(
        True, group=14, kind="bool", cli="--report-what-next / --no-report-what-next", dest="report_what_next",
        signature=False, help='Actionable "what to do next" section.',
    )
    machine_readable_manifest: bool = _opt(
        True, group=14, kind="bool", cli="--report-manifest / --no-report-manifest", dest="report_manifest",
        signature=False, help="Write report_data.json alongside the reports.",
    )


@dataclass
class BatchSection:
    source: Any = _opt(
        None, group=15, kind="str_or_list", nullable=True, cli="--batch", dest="batch",
        help="A manifest CSV (one row per dataset), a directory of CSVs, a .txt file listing CSV paths, or a "
        "list of CSV paths.",
    )
    mode: str = _opt(
        "auto", group=15, kind="choice", choices=("auto", "manifest", "directory", "list"), cli="--batch-mode",
        dest="batch_mode", signature=False,
        help="How to read batch.source. auto: directory -> directory; .txt -> list; CSV with a 'path' column -> "
        "manifest; several paths -> list.",
    )
    continue_on_error: bool = _opt(
        True, group=15, kind="bool", cli="--continue-on-error / --no-continue-on-error", dest="continue_on_error",
        signature=False,
        help="Record a failing dataset (status failed + error + remedy) and continue with the next one.",
    )
    aggregate_summary: bool = _opt(
        True, group=15, kind="bool", cli="--aggregate-summary / --no-aggregate-summary",
        dest="batch_aggregate_summary", signature=False,
        help="Write dataset_summary.csv with one row per dataset (status, best model, metric, error).",
    )


@dataclass
class MultiseedSection:
    enabled: bool = _opt(
        True, group=15, kind="bool", cli="--run-tdc22-multiseed-best / --no-run-tdc22-multiseed-best",
        dest="run_tdc22_multiseed_best", signature=False,
        help="After the run, re-evaluate the selected model of each official TDC ADMET Benchmark Group dataset "
        "over several seeds. Datasets outside that group are skipped.",
    )
    seeds: Any = _opt(
        5, group=15, kind="int_or_list", cli="--tdc22-multiseed-seeds", dest="tdc22_multiseed_seeds",
        signature=False, help="Number of seeds (n means seeds 1..n) or an explicit list of seeds.",
    )


@dataclass
class CachingSection:
    granularity: list[str] = _opt(
        group=14, kind="list", choices=("dataset", "stage"), cli="--resume-granularity", dest="resume_granularity",
        signature=False, default_factory=lambda: ["dataset", "stage"],
        help="dataset reuses fully completed datasets; stage also reuses the split/feature-selection cache and "
        "completed model stages inside an interrupted dataset.",
    )
    validate_against_config_signature: bool = _opt(
        True, group=14, kind="bool", cli="--validate-resume-signature / --no-validate-resume-signature",
        dest="resume_validate_signature", signature=False,
        help="Recompute any cached model stage whose recorded config signature differs from the current one.",
    )
    atomic_writes: bool = _opt(
        True, group=14, kind="bool", cli="--atomic-writes / --no-atomic-writes", dest="atomic_writes",
        signature=False,
        help="Write every artifact to a temporary file and rename it into place, so an interrupted run never "
        "leaves a half-written file.",
    )


@dataclass
class RunConfig:
    """Every decision of a QSARena run. Build it with :meth:`defaults`, :meth:`from_dict`,
    :meth:`from_yaml` or :meth:`from_namespace`; never mutate ``_explicit`` by hand."""

    run: RunSection = field(default_factory=RunSection)
    input: InputSection = field(default_factory=InputSection)
    standardize: StandardizeSection = field(default_factory=StandardizeSection)
    features: FeaturesSection = field(default_factory=FeaturesSection)
    split: SplitSection = field(default_factory=SplitSection)
    feature_selection: FeatureSelectionSection = field(default_factory=FeatureSelectionSection)
    models: ModelsSection = field(default_factory=ModelsSection)
    ga_tuning: GATuningSection = field(default_factory=GATuningSection)
    deep: DeepSection = field(default_factory=DeepSection)
    fusion: FusionSection = field(default_factory=FusionSection)
    ensemble: EnsembleSection = field(default_factory=EnsembleSection)
    applicability_domain: ApplicabilityDomainSection = field(default_factory=ApplicabilityDomainSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    selection: SelectionSection = field(default_factory=SelectionSection)
    report: ReportSection = field(default_factory=ReportSection)
    batch: BatchSection = field(default_factory=BatchSection)
    multiseed: MultiseedSection = field(default_factory=MultiseedSection)
    caching: CachingSection = field(default_factory=CachingSection)
    _explicit: set = field(default_factory=set, repr=False, compare=False)

    # -- construction ---------------------------------------------------------------------------

    @classmethod
    def defaults(cls) -> "RunConfig":
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, source: str = "config") -> "RunConfig":
        """Build from a nested mapping (a parsed run.yaml). Unknown keys and invalid values raise
        :class:`ConfigError` naming the key, the value and the allowed values."""
        cfg = cls()
        errors: list[str] = []
        flat = _flatten_mapping(data or {}, errors=errors, source=source)
        for key, value in flat.items():
            key, value = _apply_key_aliases(key, value)
            if key is None:
                continue
            spec = option_by_key(key)
            if spec is None:
                errors.append(_unknown_key_message(key, source))
                continue
            try:
                cfg.set(key, value)
            except ConfigError as exc:
                errors.append(str(exc))
        if errors:
            raise ConfigError(f"{len(errors)} problem(s) in {source}:\n  - " + "\n  - ".join(errors))
        cfg.validate(resolved=False)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        return load_run_config(path)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "RunConfig":
        """The fully resolved config for a parsed (and default-resolved) runner namespace."""
        cfg = cls()
        ns = vars(args)
        for spec in iter_options():
            if spec.dest is None or spec.key in _COMPOUND_KEYS:
                continue
            if spec.dest not in ns:
                continue
            value = _value_from_arg(spec, ns[spec.dest])
            _set_path(cfg, spec.key, value)
        for key, reader in _COMPOUND_READERS.items():
            try:
                _set_path(cfg, key, reader(ns))
            except KeyError:
                continue
        cfg._explicit = set()
        return cfg

    # -- access ---------------------------------------------------------------------------------

    def get(self, key: str) -> Any:
        return _get_path(self, key)

    def set(self, key: str, value: Any) -> None:
        spec = option_by_key(key)
        if spec is None:
            raise ConfigError(_unknown_key_message(key, "config"))
        normalized = spec.normalize(value)
        _set_path(self, key, normalized)
        self._explicit.add(key)

    @property
    def explicit_keys(self) -> set[str]:
        return set(self._explicit)

    # -- validation -----------------------------------------------------------------------------

    def validate(self, *, resolved: bool = True) -> None:
        """Cross-field checks that single-field validation cannot see.

        ``resolved=False`` (a config file on its own) skips checks the command line can still
        satisfy, such as ``run.mode: single`` whose ``input.path`` arrives via ``--dataset``.
        """
        errors: list[str] = []
        for spec in iter_options():
            try:
                spec.normalize(self.get(spec.key))
            except ConfigError as exc:
                errors.append(str(exc))
        if sorted(self.report.formats) != ["html", "md"]:
            errors.append("report.formats: is locked to [html, md]; both reports are always written.")
        low, high = self.feature_selection.binary_prevalence_range
        if not (0.0 <= low < high <= 1.0):
            errors.append(
                f"feature_selection.binary_prevalence_range: {low}, {high} must satisfy 0 <= low < high <= 1."
            )
        if resolved and self.run.mode == "single" and not self.input.path:
            errors.append("run.mode: 'single' needs input.path (or --dataset PATH).")
        if resolved and self.run.mode == "batch" and not self.batch.source:
            errors.append("run.mode: 'batch' needs batch.source (or --batch PATH).")
        if (
            resolved
            and self.split.strategy == "predefined"
            and self.input.path
            and not self.split.predefined_split_col
        ):
            errors.append(
                "split.strategy: 'predefined' on a user CSV needs split.predefined_split_col "
                "(or --predefined-split-col COLUMN)."
            )
        if self.ga_tuning.mode == "on" and not self.ga_tuning.estimators:
            errors.append("ga_tuning.estimators: must list at least one estimator when ga_tuning.mode is 'on'.")
        if "maplight" in self.features.families and not self.features.maplight_classic:
            errors.append(
                "features.families: lists maplight (or its avalon/erg aliases) but features.maplight_classic is "
                "false; set one of them consistently."
            )
        if not self.features.families and not self.features.maplight_classic:
            errors.append("features.families: select at least one family (or set features.maplight_classic).")
        if self.input.classification_threshold is not None and self.input.task == "regression":
            errors.append("input.classification_threshold: cannot be combined with input.task: regression.")
        if errors:
            raise ConfigError(f"{len(errors)} invalid setting(s):\n  - " + "\n  - ".join(errors))

    # -- serialization --------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for spec in iter_options():
            _set_nested(out, spec.key, copy.deepcopy(self.get(spec.key)))
        return out

    def explicit_dict(self) -> dict[str, Any]:
        """Only the keys that were set explicitly (a file, a manifest row, notebook widgets)."""
        out: dict[str, Any] = {}
        for key in sorted(self._explicit):
            _set_nested(out, key, copy.deepcopy(self.get(key)))
        return out

    def to_explicit_yaml(self) -> str:
        """YAML with only the explicitly set keys, in schema order (for notebook export)."""
        lines: list[str] = []
        order = [spec.key for spec in iter_options() if spec.key in self._explicit]
        current: list[str] = []
        for key in order:
            parts = key.split(".")
            common = 0
            while common < min(len(current), len(parts) - 1) and current[common] == parts[common]:
                common += 1
            current = current[:common]
            for depth in range(common, len(parts) - 1):
                lines.append("  " * depth + f"{parts[depth]}:")
                current.append(parts[depth])
            indent = "  " * (len(parts) - 1)
            value = self.get(key)
            if isinstance(value, dict):
                lines.append(f"{indent}{parts[-1]}:")
                lines += [f"{indent}  {k}: {_yaml_scalar(v)}" for k, v in value.items()]
            else:
                lines.append(f"{indent}{parts[-1]}: {_yaml_scalar(value)}")
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False, default=str)

    def to_yaml(self) -> str:
        return render_example_yaml(self, comments=False)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {"config": self.to_dict(), "config_signature": self.config_signature()}

    def config_signature(self) -> str:
        """sha256 over every option that can change a model result (paths, verbosity, reporting and
        resume switches are excluded)."""
        payload = {spec.key: self.get(spec.key) for spec in iter_options() if spec.signature}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    # -- argparse bridge ------------------------------------------------------------------------

    def to_arg_values(self, *, explicit_only: bool = True) -> dict[str, Any]:
        """``{argparse dest: value}`` for the runner. With ``explicit_only`` only keys that were
        present in the file are returned, so unspecified options keep their CLI/profile defaults."""
        keys = self._explicit if explicit_only else {spec.key for spec in iter_options()}
        values: dict[str, Any] = {}
        for key in sorted(keys):
            spec = option_by_key(key)
            if spec is None or key in _COMPOUND_KEYS:
                continue
            if spec.dest is None:
                continue
            values[spec.dest] = _value_to_arg(spec, self.get(key))
        for group_keys, writer in _COMPOUND_WRITERS:
            if any(key in keys for key in group_keys):
                values.update(writer(self))
        return values


# ---------------------------------------------------------------------------------------------
# Option registry (derived from the dataclasses)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionSpec:
    key: str
    group: int
    help: str
    kind: str
    default: Any
    choices: tuple | None
    aliases: dict
    cli: str | None
    dest: str | None
    nullable: bool
    minimum: float | None
    maximum: float | None
    per_dataset: bool
    signature: bool
    null_means: str

    @property
    def group_name(self) -> str:
        return GROUPS[self.group]

    def normalize(self, value: Any) -> Any:
        """Validate and canonicalize one value for this option (raises :class:`ConfigError`)."""
        return _normalize_value(self, value)


_SECTION_ORDER = [f.name for f in fields(RunConfig) if not f.name.startswith("_")]
_OPTIONS_CACHE: list[OptionSpec] | None = None


def _walk(cls: type, prefix: str) -> Iterable[OptionSpec]:
    instance = cls()
    for f in fields(cls):
        if f.name.startswith("_"):
            continue
        meta = f.metadata
        key = f"{prefix}{f.name}"
        if meta.get("qsarena_option"):
            yield OptionSpec(
                key=key,
                group=meta["group"],
                help=meta["help"],
                kind=meta["kind"],
                default=copy.deepcopy(getattr(instance, f.name)),
                choices=meta["choices"],
                aliases=meta["aliases"],
                cli=meta["cli"],
                dest=meta["dest"],
                nullable=meta["nullable"],
                minimum=meta["minimum"],
                maximum=meta["maximum"],
                per_dataset=meta["per_dataset"],
                signature=meta["signature"],
                null_means=meta["null_means"],
            )
        elif dataclasses.is_dataclass(f.type if isinstance(f.type, type) else getattr(instance, f.name)):
            yield from _walk(type(getattr(instance, f.name)), key + ".")


def iter_options() -> list[OptionSpec]:
    """Every option in declaration order (the order of Appendix A)."""
    global _OPTIONS_CACHE
    if _OPTIONS_CACHE is None:
        _OPTIONS_CACHE = list(_walk(RunConfig, ""))
    return list(_OPTIONS_CACHE)


def option_by_key(key: str) -> OptionSpec | None:
    for spec in iter_options():
        if spec.key == key:
            return spec
    return None


# ---------------------------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------------------------

_TRUE = {"true", "yes", "on", "1", "y"}
_FALSE = {"false", "no", "off", "0", "n"}


def _as_bool(spec: OptionSpec, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{spec.key}: {value!r} is not a boolean (use true or false).")


def _as_number(spec: OptionSpec, value: Any, integer: bool) -> float | int:
    if isinstance(value, bool):
        raise ConfigError(f"{spec.key}: {value!r} is not a number.")
    try:
        number = int(value) if integer and float(value) == int(float(value)) else float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{spec.key}: {value!r} is not {'an integer' if integer else 'a number'}.") from None
    if integer and not isinstance(number, int):
        raise ConfigError(f"{spec.key}: {value!r} is not an integer.")
    if spec.minimum is not None and number < spec.minimum:
        raise ConfigError(f"{spec.key}: {value!r} is below the minimum {spec.minimum}.")
    if spec.maximum is not None and number > spec.maximum:
        raise ConfigError(f"{spec.key}: {value!r} is above the maximum {spec.maximum}.")
    return number


def _choice(spec: OptionSpec, value: Any) -> str:
    if isinstance(value, bool) and spec.choices:
        # YAML 1.1 reads bare on/off/yes/no as booleans; map them back onto the choice names.
        for true_name, false_name in (("on", "off"), ("true", "false"), ("yes", "no")):
            if true_name in spec.choices and false_name in spec.choices:
                return true_name if value else false_name
    text = str(value).strip()
    lowered = text.lower()
    lowered = str(spec.aliases.get(lowered, lowered))
    if spec.choices and lowered not in spec.choices:
        suggestion = difflib.get_close_matches(lowered, [str(c) for c in spec.choices], n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ConfigError(
            f"{spec.key}: {value!r} is not one of {', '.join(map(str, spec.choices))}.{hint}"
        )
    return lowered


def _split_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_value(spec: OptionSpec, value: Any) -> Any:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"null", "none", "~"} and spec.nullable):
        if spec.nullable:
            return None
        if spec.kind == "list":
            return []
        raise ConfigError(f"{spec.key}: must not be null.")
    kind = spec.kind
    if kind == "bool":
        return _as_bool(spec, value)
    if kind == "int":
        return int(_as_number(spec, value, integer=True))
    if kind == "float":
        return float(_as_number(spec, value, integer=False))
    if kind == "str":
        text = str(value).strip()
        return text if text else (None if spec.nullable else text)
    if kind == "choice":
        return _choice(spec, value)
    if kind == "tristate":
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value).strip().lower()
        if text == "auto":
            return "auto"
        return "true" if _as_bool(spec, text) else "false"
    if kind == "list":
        items = _split_list(value)
        if spec.choices is None:
            return [str(item).strip() for item in items if str(item).strip()]
        out: list[str] = []
        for item in items:
            canonical = _choice(spec, item)
            if canonical not in out:
                out.append(canonical)
        return out
    if kind == "str_or_list":
        if isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value if str(item).strip()]
            if not items:
                return None if spec.nullable else []
            return items[0] if len(items) == 1 else items
        text = str(value).strip()
        return text or None
    if kind == "int_or_list":
        if isinstance(value, (list, tuple)) or (isinstance(value, str) and "," in value):
            seeds = [int(_as_number(spec, item, integer=True)) for item in _split_list(value)]
            if not seeds:
                raise ConfigError(f"{spec.key}: needs at least one seed.")
            return seeds
        count = int(_as_number(spec, value, integer=True))
        if count < 1:
            raise ConfigError(f"{spec.key}: {value!r} must be at least 1.")
        return count
    if kind == "float_pair":
        items = _split_list(value)
        if len(items) != 2:
            raise ConfigError(f"{spec.key}: {value!r} must be two numbers [low, high].")
        return [float(_as_number(spec, items[0], integer=False)), float(_as_number(spec, items[1], integer=False))]
    if kind == "family_map":
        if isinstance(value, Mapping):
            out_map = _enable_families_default()
            for name, enabled in value.items():
                if name not in MODEL_FAMILIES:
                    suggestion = difflib.get_close_matches(str(name), list(MODEL_FAMILIES), n=1)
                    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                    raise ConfigError(
                        f"{spec.key}.{name}: unknown model family. Known: {', '.join(MODEL_FAMILIES)}.{hint}"
                    )
                out_map[str(name)] = _as_bool(spec, enabled)
            return out_map
        raise ConfigError(f"{spec.key}: must be a mapping of family -> true/false.")
    raise ConfigError(f"{spec.key}: unsupported option kind {kind!r}.")  # pragma: no cover


# ---------------------------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------------------------


def _get_path(obj: Any, key: str) -> Any:
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def _set_path(obj: Any, key: str, value: Any) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _set_nested(target: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _flatten_mapping(data: Mapping[str, Any], *, errors: list[str], source: str, prefix: str = "") -> dict[str, Any]:
    """Flatten nested sections to dotted keys, stopping at option leaves (family maps stay mappings)."""
    flat: dict[str, Any] = {}
    if not isinstance(data, Mapping):
        errors.append(f"{source}: the top level must be a mapping of sections (run:, input:, ...).")
        return flat
    for raw_key, value in data.items():
        key = f"{prefix}{raw_key}"
        spec = option_by_key(key)
        if spec is not None:
            flat[key] = value
            continue
        if isinstance(value, Mapping) and _is_section_prefix(key):
            flat.update(_flatten_mapping(value, errors=errors, source=source, prefix=key + "."))
            continue
        flat[key] = value
    return flat


def _is_section_prefix(key: str) -> bool:
    return any(spec.key.startswith(key + ".") for spec in iter_options())


def _unknown_key_message(key: str, source: str) -> str:
    known = [spec.key for spec in iter_options()]
    suggestion = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
    return f"{key}: unknown option in {source}.{hint} See docs/options_reference.md for every key."


def _apply_key_aliases(key: str, value: Any) -> tuple[str | None, Any]:
    """Appendix A spellings that differ from the schema."""
    if key == "deep.unimol.v2_84m":
        return ("deep.unimol.v2", value)
    return key, value


# ---------------------------------------------------------------------------------------------
# argparse <-> config conversions
# ---------------------------------------------------------------------------------------------

_SPLIT_TO_ARG = {"target_quartile": "target_quartiles"}
_SELECTOR_TO_ARG = {"elasticnetcv": "elasticnet_cv", "rf_fallback": "rf_importance", "none": "none"}
_SELECTOR_FROM_ARG = {v: k for k, v in _SELECTOR_TO_ARG.items()}
_GA_NAME_TO_ARG = {"elastic_net": "ElasticNet", "catboost": "CatBoost"}
_GA_NAME_FROM_ARG = {v.lower(): k for k, v in _GA_NAME_TO_ARG.items()}
_ENSEMBLE_METHOD_LABELS = {
    "oof_stacking": "OOF Stacking (RidgeCV)",
    "inverse_rmse_average": "Weighted average (inverse train RMSE)",
    "simple_average": "Simple average",
}
_CHEMPROP_VARIANT_DESTS = {
    "dmpnn": "run_chemprop_dmpnn",
    "dmpnn_rdkit2d": "run_chemprop_rdkit2d",
    "selected_features": "run_chemprop_selected_features",
    "cmpnn": "run_chemprop_cmpnn",
    "attentivefp": "run_chemprop_attentivefp",
}


def _value_to_arg(spec: OptionSpec, value: Any) -> Any:
    key = spec.key
    if key == "split.strategy":
        return _SPLIT_TO_ARG.get(value, value)
    if key == "feature_selection.method":
        return _SELECTOR_TO_ARG[value]
    if key == "ensemble.member_selection_metric":
        return {"oof": "oof", "cv": "train"}.get(value, "test")
    if key == "evaluation.primary_metric":
        return None if value == "auto" else value
    if key == "deep.use_gpu":
        return value
    if spec.kind == "tristate":
        return None if value == "auto" else value == "true"
    if key == "input.path":
        if value is None:
            return None
        return list(value) if isinstance(value, list) else [value]
    if key in {"input.target_col", "batch.source"}:
        if value is None:
            return None
        return list(value) if isinstance(value, list) else [value]
    if key == "multiseed.seeds":
        seeds = value if isinstance(value, list) else list(range(1, int(value) + 1))
        return ",".join(str(seed) for seed in seeds)
    return copy.deepcopy(value)


def _value_from_arg(spec: OptionSpec, value: Any) -> Any:
    key = spec.key
    if key == "split.strategy":
        return "target_quartile" if value == "target_quartiles" else value
    if key == "feature_selection.method":
        return _SELECTOR_FROM_ARG.get(value, value)
    if key == "ensemble.member_selection_metric":
        return {"oof": "oof", "test": "test"}.get(str(value), "cv")
    if key == "evaluation.primary_metric":
        return "auto" if value in (None, "", "auto") else value
    if spec.kind == "tristate" and key != "deep.use_gpu":
        return "auto" if value is None else ("true" if bool(value) else "false")
    if key in {"input.path", "input.target_col", "batch.source"}:
        if not value:
            return None
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        items = [str(item) for item in items]
        return items[0] if len(items) == 1 else items
    if key == "multiseed.seeds":
        seeds = [int(s) for s in str(value).split(",") if str(s).strip()]
        return len(seeds) if seeds == list(range(1, len(seeds) + 1)) else seeds
    if spec.kind == "list" and value is None:
        return []
    try:
        return _normalize_value(spec, value)
    except ConfigError:
        return value


def _write_ga(cfg: RunConfig) -> dict[str, Any]:
    mode = cfg.ga_tuning.mode
    out: dict[str, Any] = {"ga_estimators": list(cfg.ga_tuning.estimators)}
    out["ga_models"] = {"off": "", "auto": "auto", "on": "on"}[mode]
    return out


def _read_ga_mode(ns: Mapping[str, Any]) -> str:
    text = str(ns["ga_models"] or "").strip().lower()
    if not text or text == "off":
        return "off"
    return "auto" if text == "auto" else "on"


def _read_ga_estimators(ns: Mapping[str, Any]) -> list[str]:
    text = str(ns["ga_models"] or "").strip()
    if text and text.lower() not in {"auto", "on", "off"}:
        # legacy explicit model names (--ga-models ElasticNet,CatBoost)
        return [_GA_NAME_FROM_ARG.get(n.strip().lower(), n.strip().lower()) for n in text.split(",") if n.strip()]
    return list(ns.get("ga_estimators") or ["elastic_net", "catboost"])


def _write_ensemble(cfg: RunConfig) -> dict[str, Any]:
    methods = [label for name, label in _ENSEMBLE_METHOD_LABELS.items() if getattr(cfg.ensemble, name)]
    out: dict[str, Any] = {"ensemble_methods": ",".join(methods)}
    if not methods:
        out["run_ensemble"] = False
    return out


def _read_ensemble_flag(name: str) -> Callable[[Mapping[str, Any]], bool]:
    def reader(ns: Mapping[str, Any]) -> bool:
        text = str(ns["ensemble_methods"] or "")
        label = _ENSEMBLE_METHOD_LABELS[name]
        return label.lower() in [item.strip().lower() for item in text.split(",")]

    return reader


def _write_chemprop(cfg: RunConfig) -> dict[str, Any]:
    variants = cfg.deep.chemprop.variants
    if variants is None:
        return {}
    out: dict[str, Any] = {dest: (name in variants) for name, dest in _CHEMPROP_VARIANT_DESTS.items()}
    out["run_chemprop_mpnn"] = bool(variants)
    return out


def _read_chemprop(ns: Mapping[str, Any]) -> list[str]:
    if not bool(ns["run_chemprop_mpnn"]):
        return []
    return [name for name, dest in _CHEMPROP_VARIANT_DESTS.items() if bool(ns.get(dest, False))]


def _write_families(cfg: RunConfig) -> dict[str, Any]:
    return {"disabled_model_families": [name for name, on in cfg.models.enable_families.items() if not on]}


def _read_families(ns: Mapping[str, Any]) -> dict[str, bool]:
    disabled = set(ns["disabled_model_families"] or [])
    out = {name: name not in disabled for name in MODEL_FAMILIES}
    # A family whose master run flag is off is reported as disabled in the resolved config.
    masters = {
        "graph_nn": "run_chemprop_mpnn",
        "maplight_gnn": "run_maplight_gnn",
        "fusion": "run_cfa",
        "ensemble": "run_ensemble",
    }
    for name, dest in masters.items():
        if dest in ns and not bool(ns[dest]):
            out[name] = False
    return out


def _write_cache(cfg: RunConfig) -> dict[str, Any]:
    on = bool(cfg.features.cache)
    return {"enable_shared_feature_matrix_cache": on, "reuse_shared_feature_matrix_cache": on}


def _write_store(cfg: RunConfig) -> dict[str, Any]:
    on = bool(cfg.features.persistent_store)
    return {"enable_persistent_feature_store": on, "reuse_persistent_feature_store": on}


_COMPOUND_WRITERS: list[tuple[tuple[str, ...], Callable[[RunConfig], dict[str, Any]]]] = [
    (("ga_tuning.mode", "ga_tuning.estimators"), _write_ga),
    (("ensemble.oof_stacking", "ensemble.inverse_rmse_average", "ensemble.simple_average"), _write_ensemble),
    (("deep.chemprop.variants",), _write_chemprop),
    (("models.enable_families",), _write_families),
    (("features.cache",), _write_cache),
    (("features.persistent_store",), _write_store),
]
_COMPOUND_KEYS = {key for keys, _writer in _COMPOUND_WRITERS for key in keys}
_COMPOUND_READERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "ga_tuning.mode": _read_ga_mode,
    "ga_tuning.estimators": _read_ga_estimators,
    "ensemble.oof_stacking": _read_ensemble_flag("oof_stacking"),
    "ensemble.inverse_rmse_average": _read_ensemble_flag("inverse_rmse_average"),
    "ensemble.simple_average": _read_ensemble_flag("simple_average"),
    "deep.chemprop.variants": _read_chemprop,
    "models.enable_families": _read_families,
    "features.cache": lambda ns: bool(ns["enable_shared_feature_matrix_cache"])
    and bool(ns["reuse_shared_feature_matrix_cache"]),
    "features.persistent_store": lambda ns: bool(ns["enable_persistent_feature_store"])
    and bool(ns["reuse_persistent_feature_store"]),
}


def _parse_family_list(text: str) -> list[str]:
    out = []
    for item in _split_list(text):
        name = str(item).strip().lower()
        if name not in MODEL_FAMILIES:
            raise argparse.ArgumentTypeError(
                f"unknown model family {name!r}; choose from {', '.join(MODEL_FAMILIES)}"
            )
        out.append(name)
    return out


def _parse_feature_families(text: str) -> list[str]:
    spec = option_by_key("features.families")
    try:
        return _normalize_value(spec, text)
    except ConfigError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parse_metric_list(key: str) -> Callable[[str], list[str]]:
    def parser(text: str) -> list[str]:
        try:
            return _normalize_value(option_by_key(key), text)
        except ConfigError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None

    return parser


def _parse_pair(text: str) -> list[float]:
    try:
        return _normalize_value(option_by_key("feature_selection.binary_prevalence_range"), text)
    except ConfigError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_run_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` and every RunConfig option that has no pre-existing runner flag."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML (or JSON) run configuration. Precedence: command-line flags > this file > built-in defaults. "
        "Template: configs/run.example.yaml; every key: docs/options_reference.md.",
    )
    bool_flag = argparse.BooleanOptionalAction
    add = parser.add_argument
    add("--run-mode", choices=["auto", "single", "batch", "benchmark"], default="auto",
        help=option_by_key("run.mode").help)
    add("--fresh", action="store_true", default=False, help=option_by_key("run.overwrite").help)
    add("--verbosity", choices=["quiet", "normal", "verbose", "debug"], default="normal",
        help=option_by_key("run.verbosity").help)
    add("--smiles-col", dest="smiles_column", default=None, help=option_by_key("input.smiles_col").help)
    add("--target-col", dest="target_columns", action="append", default=None,
        help=option_by_key("input.target_col").help + " Repeat the flag for several targets.")
    add("--id-col", dest="id_column", default=None, help=option_by_key("input.id_col").help)
    add("--task", dest="task_type", choices=["auto", "regression", "classification"], default="auto",
        help=option_by_key("input.task").help)
    add("--classification-threshold", type=float, default=None, help=option_by_key("input.classification_threshold").help)
    add("--drop-unparseable", dest="drop_unparseable_smiles", action=bool_flag, default=True,
        help=option_by_key("standardize.drop_unparseable").help)
    add("--strip-salts", action=bool_flag, default=False, help=option_by_key("standardize.strip_salts").help)
    add("--normalize-charges", action=bool_flag, default=False, help=option_by_key("standardize.normalize_charges").help)
    add("--normalize-tautomers", action=bool_flag, default=False,
        help=option_by_key("standardize.normalize_tautomers").help)
    add("--deduplicate", choices=["none", "exact", "canonical_smiles"], default="none",
        help=option_by_key("standardize.deduplicate").help)
    add("--feature-families", type=_parse_feature_families, default=list(FEATURE_FAMILIES),
        help="Comma-separated. " + option_by_key("features.families").help)
    add("--maplight-classic", action=bool_flag, default=True, help=option_by_key("features.maplight_classic").help)
    add("--predefined-split-col", dest="predefined_split_column", default=None,
        help=option_by_key("split.predefined_split_col").help)
    add("--variance-threshold", dest="dedup_variance_threshold", type=float, default=1e-8,
        help=option_by_key("feature_selection.variance_threshold").help)
    add("--binary-prevalence-range", type=_parse_pair, default=[0.005, 0.995],
        help="LOW,HIGH. " + option_by_key("feature_selection.binary_prevalence_range").help)
    add("--drop-duplicate-columns", dest="drop_duplicate_feature_columns", action=bool_flag, default=True,
        help=option_by_key("feature_selection.drop_duplicate_columns").help)
    add("--disable-model-families", dest="disabled_model_families", type=_parse_family_list, default=[],
        help="Comma-separated model families to switch off: " + ", ".join(MODEL_FAMILIES) + ".")
    add("--disable-model", dest="disable_models", action="append", default=[],
        help=option_by_key("models.disable_models").help + " Repeat for several models.")
    add("--ga-estimators", type=_parse_metric_list("ga_tuning.estimators"), default=["elastic_net", "catboost"],
        help="Comma-separated. " + option_by_key("ga_tuning.estimators").help + " Used with --ga-models on.")
    add("--ga-time-budget-minutes", type=float, default=None, help=option_by_key("ga_tuning.time_budget_min").help)
    add("--ga-max-configs", type=int, default=None, help=option_by_key("ga_tuning.max_configs").help)
    add("--use-gpu", dest="gpu_policy", choices=["auto", "true", "false"], default="auto",
        help=option_by_key("deep.use_gpu").help)
    add("--ad-method", choices=["standardization", "confidence", "both", "off"], default="both",
        help=option_by_key("applicability_domain.method").help)
    add("--ad-confidence-threshold", type=float, default=0.5,
        help=option_by_key("applicability_domain.confidence_threshold").help)
    add("--ad-knn-quantile", type=float, default=0.95, help=option_by_key("applicability_domain.knn_quantile").help)
    add("--report-regression-metrics", type=_parse_metric_list("evaluation.regression_metrics"),
        default=list(_REGRESSION_METRICS), help=option_by_key("evaluation.regression_metrics").help)
    add("--report-classification-metrics", type=_parse_metric_list("evaluation.classification_metrics"),
        default=list(_CLASSIFICATION_METRICS), help=option_by_key("evaluation.classification_metrics").help)
    add("--primary-metric", dest="primary_metric_override", choices=list(_PRIMARY_METRICS), default="auto",
        help=option_by_key("evaluation.primary_metric").help)
    add("--selection-protocol", choices=["test", "cv", "both"], default="both",
        help=option_by_key("selection.protocol").help)
    add("--report-plots", dest="report_include_plots", action=bool_flag, default=True,
        help=option_by_key("report.include_plots").help)
    add("--report-what-next", action=bool_flag, default=True, help=option_by_key("report.what_next").help)
    add("--report-manifest", action=bool_flag, default=True, help=option_by_key("report.machine_readable_manifest").help)
    add("--batch", action="append", default=None, help=option_by_key("batch.source").help + " Repeatable.")
    add("--batch-mode", choices=["auto", "manifest", "directory", "list"], default="auto",
        help=option_by_key("batch.mode").help)
    add("--continue-on-error", action=bool_flag, default=True, help=option_by_key("batch.continue_on_error").help)
    add("--aggregate-summary", dest="batch_aggregate_summary", action=bool_flag, default=True,
        help=option_by_key("batch.aggregate_summary").help)
    add("--resume-granularity", type=_parse_metric_list("caching.granularity"), default=["dataset", "stage"],
        help="Comma-separated. " + option_by_key("caching.granularity").help)
    add("--validate-resume-signature", dest="resume_validate_signature", action=bool_flag, default=True,
        help=option_by_key("caching.validate_against_config_signature").help)
    add("--atomic-writes", action=bool_flag, default=True, help=option_by_key("caching.atomic_writes").help)
    parser.set_defaults(report_formats=["html", "md"])


def cli_provided_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    """argparse destinations explicitly given on the command line (either polarity of a boolean)."""
    tokens = [str(token) for token in argv]
    provided: set[str] = set()
    for action in parser._actions:  # noqa: SLF001 - argparse has no public action iterator
        for option in action.option_strings:
            if any(token == option or token.startswith(option + "=") for token in tokens):
                provided.add(action.dest)
                break
    return provided


def regroup_parser_help(parser: argparse.ArgumentParser) -> None:
    """Show ``--help`` as the fifteen decision groups. Only the help layout changes."""
    by_dest: dict[str, int] = {}
    for spec in iter_options():
        if spec.dest:
            by_dest.setdefault(spec.dest, spec.group)
    extra = {
        "ga_models": 7, "ensemble_methods": 10, "run_chemprop_mpnn": 8, "run_chemprop_dmpnn": 8,
        "run_chemprop_cmpnn": 8, "run_chemprop_attentivefp": 8, "run_chemprop_selected_features": 8,
        "run_chemprop_rdkit2d": 8, "run_maplight_gnn": 6, "run_ensemble": 10,
        "enable_shared_feature_matrix_cache": 3, "reuse_shared_feature_matrix_cache": 3,
        "enable_persistent_feature_store": 3, "reuse_persistent_feature_store": 3, "config": 15,
        "include_local_csv": 15, "pfas_aux_workbook": 15, "pfas_aux_sheet": 15, "tdc22_multiseed": 15,
    }
    for dest, group in extra.items():
        by_dest.setdefault(dest, group)
    try:
        groups = {
            number: parser.add_argument_group(f"{number}. {name}")
            for number, name in GROUPS.items()
        }
        optionals = parser._optionals  # noqa: SLF001
        remaining = []
        for action in list(optionals._group_actions):  # noqa: SLF001
            group_number = by_dest.get(action.dest)
            if group_number is None:
                remaining.append(action)
                continue
            groups[group_number]._group_actions.append(action)  # noqa: SLF001
        optionals._group_actions[:] = remaining  # noqa: SLF001
        optionals.title = "other options (advanced and legacy flags)"
        parser._action_groups.remove(optionals)  # noqa: SLF001
        parser._action_groups.append(optionals)  # noqa: SLF001
    except Exception:  # pragma: no cover - help layout must never break the CLI
        return


def load_run_config(path: str | Path) -> RunConfig:
    """Load a YAML or JSON run configuration with actionable errors."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"--config: {path} does not exist.")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: invalid JSON ({exc}).") from None
    else:
        try:
            import yaml  # type: ignore
        except ImportError:  # pragma: no cover - PyYAML is a core dependency
            raise ConfigError(
                f"{path}: reading YAML needs PyYAML (pip install pyyaml), or pass a .json config instead."
            ) from None
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML ({exc}).") from None
    return RunConfig.from_dict(data, source=str(path))


def per_dataset_arg_overrides(overrides: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Convert ``{config key: value}`` (a batch-manifest row) into ``{argparse dest: value}``.

    Only options marked per-dataset are allowed; anything run-level (output_dir, verbosity, ...)
    raises :class:`ConfigError`.
    """
    cfg = RunConfig()
    errors: list[str] = []
    for key, value in overrides.items():
        spec = option_by_key(key)
        if spec is None:
            errors.append(_unknown_key_message(key, source))
            continue
        if not spec.per_dataset:
            errors.append(f"{key}: is a run-level option and cannot be overridden per dataset ({source}).")
            continue
        try:
            cfg.set(key, value)
        except ConfigError as exc:
            errors.append(f"{exc} ({source})")
    if errors:
        raise ConfigError("\n".join(errors))
    return cfg.to_arg_values(explicit_only=True)


# ---------------------------------------------------------------------------------------------
# Notebook widget mapping
# ---------------------------------------------------------------------------------------------

#: config key -> notebook ``# @param`` variable(s) implementing the same decision. The notebook runs
#: its own interactive pipeline, so only decisions it implements are listed; the options reference
#: marks the rest "CLI / run.yaml only".
NOTEBOOK_WIDGETS: dict[str, tuple[str, ...]] = {
    "input.smiles_col": ("smiles_column",),
    "input.target_col": ("target_column",),
    "standardize.deduplicate": ("collapse_duplicate_canonical_smiles",),
    "features.families": (
        "use_morgan_features", "use_ecfp6_features", "use_fcfp6_features", "use_layered_features",
        "use_atom_pair_features", "use_topological_torsion_features", "use_rdk_path_features", "use_maccs_keys",
        "use_rdkit_descriptors",
    ),
    "features.maplight_classic": ("use_maplight_classic",),
    "features.fingerprint_bits": ("fingerprint_bits",),
    "features.persistent_store": ("enable_persistent_feature_store", "reuse_persistent_feature_store"),
    "split.strategy": ("data_split_strategy",),
    "split.test_fraction": ("test_fraction",),
    "split.seed": ("model_random_seed",),
    "split.cv_folds": ("cv_folds",),
    "feature_selection.method": ("feature_selector_method",),
    "ga_tuning.generations": ("ga_generations",),
    "ga_tuning.population_size": ("ga_population_size",),
    "ga_tuning.estimators": ("tune_elasticnet", "tune_catboost"),
    "deep.chemml.pytorch": ("run_chemml_pytorch",),
    "deep.chemml.tensorflow": ("run_chemml_tensorflow",),
    "deep.chemml.epochs": ("chemml_training_epochs",),
    "models.enable_families": ("run_maplight_gnn",),
}

_FAMILY_WIDGETS = dict(zip(FEATURE_FAMILIES, NOTEBOOK_WIDGETS["features.families"]))
_NOTEBOOK_SELECTOR = {"elasticnetcv": "elasticnet_cv", "rf_fallback": "random_forest_importance", "none": "none"}


def notebook_widget_names() -> list[str]:
    """Every notebook ``# @param`` variable that has a RunConfig equivalent."""
    return sorted({name for names in NOTEBOOK_WIDGETS.values() for name in names})


def notebook_widget_values(cfg: RunConfig, *, explicit_only: bool = True) -> dict[str, Any]:
    """``{notebook widget variable: value}`` for the decisions a run.yaml sets."""
    keys = cfg.explicit_keys if explicit_only else set(NOTEBOOK_WIDGETS)
    out: dict[str, Any] = {}
    for key in NOTEBOOK_WIDGETS:
        if key not in keys:
            continue
        value = cfg.get(key)
        if key == "input.smiles_col":
            out["smiles_column"] = value or "AUTO"
        elif key == "input.target_col":
            first = value[0] if isinstance(value, list) else value
            out["target_column"] = first or "AUTO"
        elif key == "standardize.deduplicate":
            out["collapse_duplicate_canonical_smiles"] = value == "canonical_smiles"
        elif key == "features.families":
            for family, widget in _FAMILY_WIDGETS.items():
                out[widget] = family in value
        elif key == "features.maplight_classic":
            out["use_maplight_classic"] = bool(value)
        elif key == "features.fingerprint_bits":
            out["fingerprint_bits"] = str(int(value))
        elif key == "features.persistent_store":
            out["enable_persistent_feature_store"] = bool(value)
            out["reuse_persistent_feature_store"] = bool(value)
        elif key == "split.strategy":
            out["data_split_strategy"] = _SPLIT_TO_ARG.get(value, value)
        elif key == "feature_selection.method":
            out["feature_selector_method"] = _NOTEBOOK_SELECTOR[value]
        elif key == "ga_tuning.estimators":
            out["tune_elasticnet"] = "elastic_net" in value
            out["tune_catboost"] = "catboost" in value
        elif key == "models.enable_families":
            out["run_maplight_gnn"] = bool(value.get("maplight_gnn", True))
        else:
            (widget,) = NOTEBOOK_WIDGETS[key]
            out[widget] = value
    return out


def run_config_from_notebook_values(values: Mapping[str, Any]) -> RunConfig:
    """The RunConfig equivalent of a notebook session's widget values (for export to run.yaml)."""
    data: dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        _set_nested(data, key, value)

    def has(*names: str) -> bool:
        return all(name in values for name in names)

    if has("smiles_column"):
        put("input.smiles_col", None if str(values["smiles_column"]).upper() == "AUTO" else values["smiles_column"])
    if has("target_column"):
        put("input.target_col", None if str(values["target_column"]).upper() == "AUTO" else values["target_column"])
    if has("collapse_duplicate_canonical_smiles"):
        put("standardize.deduplicate", "canonical_smiles" if values["collapse_duplicate_canonical_smiles"] else "none")
    if all(widget in values for widget in _FAMILY_WIDGETS.values()):
        put("features.families", [family for family, widget in _FAMILY_WIDGETS.items() if values[widget]])
    if has("use_maplight_classic"):
        put("features.maplight_classic", bool(values["use_maplight_classic"]))
    if has("fingerprint_bits"):
        put("features.fingerprint_bits", int(values["fingerprint_bits"]))
    if has("enable_persistent_feature_store"):
        put("features.persistent_store", bool(values["enable_persistent_feature_store"]))
    if has("data_split_strategy"):
        put("split.strategy", values["data_split_strategy"])
    for key, widget in (
        ("split.test_fraction", "test_fraction"),
        ("split.seed", "model_random_seed"),
        ("split.cv_folds", "cv_folds"),
        ("ga_tuning.generations", "ga_generations"),
        ("ga_tuning.population_size", "ga_population_size"),
        ("deep.chemml.pytorch", "run_chemml_pytorch"),
        ("deep.chemml.tensorflow", "run_chemml_tensorflow"),
        ("deep.chemml.epochs", "chemml_training_epochs"),
    ):
        if has(widget):
            put(key, values[widget])
    if has("feature_selector_method"):
        reverse = {v: k for k, v in _NOTEBOOK_SELECTOR.items()}
        method = str(values["feature_selector_method"])
        if method in reverse:
            put("feature_selection.method", reverse[method])
    if has("tune_elasticnet", "tune_catboost"):
        estimators = [n for n, w in (("elastic_net", "tune_elasticnet"), ("catboost", "tune_catboost")) if values[w]]
        if estimators:
            put("ga_tuning.estimators", estimators)
    if has("run_maplight_gnn"):
        put("models.enable_families", {"maplight_gnn": bool(values["run_maplight_gnn"])})
    return RunConfig.from_dict(data, source="notebook widgets")


# ---------------------------------------------------------------------------------------------
# Generated documentation
# ---------------------------------------------------------------------------------------------


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = repr(value)
        return text if ("e" in text or "." in text) else text + ".0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
    text = str(value)
    needs_quote = (
        not text
        or text.strip() != text
        or text.lower() in {"true", "false", "yes", "no", "on", "off", "null", "none", "~"}
        or any(ch in text for ch in ":#,[]{}&*!|>'\"%@`")
        or text[0] in "-?"
    )
    if not needs_quote:
        try:
            float(text)
            needs_quote = True
        except ValueError:
            pass
    return json.dumps(text) if needs_quote else text


def _wrap_comment(text: str, indent: str, width: int = 90) -> list[str]:
    import textwrap

    return [f"{indent}# {line}" for line in textwrap.wrap(text, width=width - len(indent) - 2)]


def render_example_yaml(cfg: RunConfig | None = None, *, comments: bool = True) -> str:
    """A complete, commented run.yaml (``configs/run.example.yaml`` is exactly this with defaults)."""
    cfg = cfg or RunConfig()
    lines: list[str] = []
    if comments:
        lines += [
            "# ============================================================================================",
            "# QSARena run configuration (generated from qsarena/config.py; do not edit by hand)",
            "# Use:        qsarena-benchmark --config run.yaml",
            "# Precedence: command-line flags > this file > built-in defaults",
            "# Every value below is the built-in default. Delete the lines you do not want to change.",
            "# Full reference: docs/options_reference.md",
            "# ============================================================================================",
        ]
    by_section: dict[str, list[OptionSpec]] = {}
    for spec in iter_options():
        by_section.setdefault(spec.key.split(".")[0], []).append(spec)
    for section in _SECTION_ORDER:
        specs = by_section.get(section, [])
        if not specs:
            continue
        groups = sorted({spec.group for spec in specs})
        if comments:
            label = ", ".join(f"{g}. {GROUPS[g]}" for g in groups)
            lines += ["", f"# ---------- {label} ----------"]
        lines.append(f"{section}:")
        open_subsections: list[str] = []
        for spec in specs:
            parts = spec.key.split(".")[1:]
            parents = parts[:-1]
            # open nested subsections (deep.chemprop.*)
            common = 0
            while common < min(len(parents), len(open_subsections)) and parents[common] == open_subsections[common]:
                common += 1
            open_subsections = open_subsections[:common]
            for depth in range(common, len(parents)):
                lines.append("  " * (depth + 1) + f"{parents[depth]}:")
                open_subsections.append(parents[depth])
            indent = "  " * (len(parents) + 1)
            value = cfg.get(spec.key)
            if comments:
                help_text = spec.help
                if spec.choices and spec.kind in {"choice", "list"}:
                    help_text += f" Choices: {', '.join(map(str, spec.choices))}."
                if spec.nullable and spec.null_means:
                    help_text += f" null = {spec.null_means}."
                if spec.cli and spec.cli != "(fixed)":
                    help_text += f" CLI: {spec.cli}."
                lines += _wrap_comment(help_text, indent)
            if spec.kind == "family_map":
                lines.append(f"{indent}{parts[-1]}:")
                for family, enabled in value.items():
                    lines.append(f"{indent}  {family}: {_yaml_scalar(enabled)}")
            else:
                lines.append(f"{indent}{parts[-1]}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _option_cells(spec: OptionSpec, *, aliases: bool = True) -> dict[str, str]:
    """Markdown cell texts describing one option (shared by the reference and the tutorial)."""
    default = spec.default
    if default is None and spec.null_means:
        default_text = f"`null` ({spec.null_means})"
    elif spec.kind == "family_map":
        default_text = "all `true`"
    else:
        default_text = f"`{_yaml_scalar(default)}`"
    if spec.kind == "bool":
        allowed = "true / false"
    elif spec.kind == "tristate":
        allowed = "auto / true / false"
    elif spec.choices:
        allowed = ", ".join(f"`{c}`" for c in spec.choices)
        if spec.kind == "list":
            allowed = "list of " + allowed
    elif spec.kind == "family_map":
        allowed = "mapping of " + ", ".join(f"`{c}`" for c in MODEL_FAMILIES) + " to true/false"
    elif spec.kind in {"int", "float"}:
        bounds = []
        if spec.minimum is not None:
            bounds.append(f">= {spec.minimum}")
        if spec.maximum is not None:
            bounds.append(f"<= {spec.maximum}")
        allowed = spec.kind + (f" ({', '.join(bounds)})" if bounds else "")
    elif spec.kind == "float_pair":
        allowed = "[low, high]"
    elif spec.kind == "int_or_list":
        allowed = "int or list of int"
    elif spec.kind == "str_or_list":
        allowed = "path/name or list"
    else:
        allowed = "text"
    if spec.aliases and aliases:
        allowed += " (aliases: " + ", ".join(f"`{a}`->`{b}`" for a, b in spec.aliases.items()) + ")"
    widgets = NOTEBOOK_WIDGETS.get(spec.key)
    widget_text = ", ".join(f"`{w}`" for w in widgets) if widgets else "CLI / run.yaml only"
    cli = f"`{spec.cli}`" if spec.cli and spec.cli != "(fixed)" else "(fixed)"
    description = spec.help + (" *Per dataset.*" if spec.per_dataset else "")
    return {"key": f"`{spec.key}`", "default": default_text, "allowed": allowed, "cli": cli,
            "widget": widget_text, "description": description}


def render_group_table(group: int, *, descriptions: bool = True) -> str:
    """Markdown table of one decision group's options.

    ``descriptions=True`` is the full table of docs/options_reference.md. ``descriptions=False`` is the
    compact tutorial table: no descriptions or aliases, and dash counts in the separator row that
    give pandoc relative column widths for the PDF."""
    columns = ["key", "default", "allowed", "cli", "widget"] + (["description"] if descriptions else [])
    titles = {"key": "Key", "default": "Default", "allowed": "Allowed values", "cli": "CLI flag",
              "widget": "Notebook widget", "description": "Description"}
    if descriptions:
        separator = "|" + "---|" * len(columns)
    else:
        widths = {"key": 24, "default": 16, "allowed": 24, "cli": 20, "widget": 16}
        separator = "|" + "|".join("-" * widths[c] for c in columns) + "|"
    lines = ["| " + " | ".join(titles[c] for c in columns) + " |", separator]
    for spec in iter_options():
        if spec.group != group:
            continue
        cells = _option_cells(spec, aliases=descriptions)
        lines.append("| " + " | ".join(_md_cell(cells[c]) for c in columns) + " |")
    return "\n".join(lines)


_GENERATED_BLOCK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[\w.\- ]+) -->\n)(?P<body>.*?)(<!-- END GENERATED -->)", re.DOTALL
)


def refresh_generated_blocks(markdown: str) -> str:
    """Rewrite the ``<!-- BEGIN GENERATED: ... -->`` blocks of a document (docs/tutorial.md):
    ``options-group N`` -> that group's option table; ``run.example.yaml`` -> the commented template."""

    def replace(match: "re.Match[str]") -> str:
        name = match.group("name").strip()
        if name.startswith("options-group "):
            body = render_group_table(int(name.split()[-1]), descriptions=False)
        elif name == "run.example.yaml":
            body = "```yaml\n" + render_example_yaml().rstrip("\n") + "\n```"
        else:
            raise ValueError(f"unknown generated block {name!r}")
        return f"{match.group(1)}{body}\n{match.group(4)}"

    return _GENERATED_BLOCK.sub(replace, markdown)


def render_options_reference() -> str:
    """``docs/options_reference.md``: every option, grouped by the fifteen decision groups."""
    lines = [
        "# QSARena options reference",
        "",
        "<!-- Generated by `python -m qsarena.config --write-docs`; do not edit by hand. -->",
        "",
        "Every workflow decision is one key of `RunConfig` (`qsarena/config.py`). A key can be set in a",
        "`run.yaml` passed with `qsarena-benchmark --config run.yaml`, or with the listed command-line flag.",
        "Precedence is **command-line flag > config file > built-in default**. The resolved configuration and",
        "its signature are written to `run_config.json` in every run directory. Keys marked *per dataset*",
        "may also be overridden per row of a batch manifest.",
        "",
        "The notebook column lists the `# @param` widget of `colab_qsar_tutorial.ipynb` implementing the same",
        "decision; step 0B of the notebook loads a `run.yaml` into those widgets and the last step exports",
        "the widgets back to a `run.yaml`.",
        "",
    ]
    by_group: dict[int, list[OptionSpec]] = {}
    for spec in iter_options():
        by_group.setdefault(spec.group, []).append(spec)
    lines.append("## Contents")
    lines.append("")
    for number, name in GROUPS.items():
        anchor = f"{number}-{name}".lower()
        anchor = "".join(ch if ch.isalnum() or ch == "-" else ("-" if ch == " " else "") for ch in anchor)
        lines.append(f"{number}. [{name}](#{anchor})")
    lines.append("")
    for number, name in GROUPS.items():
        lines += [f"## {number}. {name}", ""]
        lines += render_group_table(number, descriptions=True).splitlines()
        lines.append("")
    lines += [
        "## Model families",
        "",
        "`models.enable_families` and the report's won-by-family plot use these families:",
        "",
    ]
    for name, description in MODEL_FAMILIES.items():
        lines.append(f"- `{name}`: {description}")
    lines.append("")
    return "\n".join(lines)


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Regenerate the RunConfig-derived documentation files.")
    parser.add_argument("--write-docs", action="store_true", help="Write docs/options_reference.md and configs/run.example.yaml.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if not args.write_docs:
        parser.print_help()
        return 0
    (args.root / "docs").mkdir(parents=True, exist_ok=True)
    (args.root / "configs").mkdir(parents=True, exist_ok=True)
    (args.root / "docs" / "options_reference.md").write_text(render_options_reference(), encoding="utf-8", newline="\n")
    (args.root / "configs" / "run.example.yaml").write_text(render_example_yaml(), encoding="utf-8", newline="\n")
    tutorial = args.root / "docs" / "tutorial.md"
    if tutorial.exists():
        text = tutorial.read_text(encoding="utf-8")
        tutorial.write_text(refresh_generated_blocks(text), encoding="utf-8", newline="\n")
    print("Wrote docs/options_reference.md and configs/run.example.yaml (and refreshed docs/tutorial.md)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
