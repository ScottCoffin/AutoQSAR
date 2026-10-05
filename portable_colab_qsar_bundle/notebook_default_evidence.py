"""Benchmark evidence for every notebook widget default.

``build_colab_qsar_tutorial.py`` appends a "Why these defaults?" table to the guidance cell above
each form. The *Default* column is read from the cell's own ``# @param`` line, so it cannot drift;
this module supplies the benchmark setting, a status and the reason. The builder fails if a form
option has no entry here (or an entry names an option that no longer exists).

Sources (keep these in sync when the evidence changes):
- Benchmark settings: ``benchmark_results/autoqsar_benchmark_20260623_153839/run_config.json``
  (canonical A100 ``full`` run, 44 datasets, one fixed configuration, no per-dataset tuning).
- Lighter-profile comparison: ``benchmark_name_date`` (RTX 4060 ``cost_optimized``) vs the A100
  run, ``manuscript_numbers.json["run_comparison_vs_rtx"]`` / Table S5.
- Feature families: Table S3; component ablation: Table S2; per-family cost: Table 6;
  family results: Table 3; fusion vs best single model: ``manuscript_numbers.json``.
- Per-model and per-Chemprop-variant median gaps: computed from
  ``qsarena_benchmark_chemprop_fixed/*/metrics.csv`` (the run with working Chemprop and TabPFN),
  single models only (ensembles/CFA excluded), each dataset scored on its Table S1 analysis metric,
  gap = relative distance to that dataset's best single model.
- Applicability domain: ``results/reliability_tdc22/summary.json`` (Section 3.13).
"""

from __future__ import annotations

BENCH = "Benchmark"   # same value the 44-dataset benchmark used
LIGHT = "Lighter"     # cheaper than the benchmark setting, deliberately, for a notebook runtime
EVID = "Evidence"     # chosen because of a specific benchmark result
DIFF = "Differs"      # differs from the benchmark, and the benchmark points the other way
CAVEAT = "Caveat"     # used by the benchmark too, but has a known limitation you should know
UNTESTED = "Untested" # affects results, but the benchmark never varied it
NOEFFECT = "No effect"  # display, caching, paths or seeds: does not change model quality

NA = "n/a"

#: Status -> meaning. Rendered into the notebook intro and into the 9F HTML report.
DEFAULTS_LEGEND: list[tuple[str, str]] = [
    (BENCH, "Same value the benchmark used on all 44 datasets. It is known to work well without tuning. It is not proven optimal: the benchmark compared models and feature families, but did not sweep most hyperparameters."),
    (EVID, "Chosen because of a specific benchmark result, cited in the row."),
    (LIGHT, "Deliberately cheaper than the benchmark setting so the notebook finishes on a free Colab runtime. The benchmark value is shown; use it when accuracy matters more than time."),
    (DIFF, "Differs from the benchmark, and the benchmark result points the other way. Read the row before keeping the default."),
    (CAVEAT, "Matches the benchmark, but has a limitation you should know about."),
    (UNTESTED, "Changes results, but the benchmark never varied it. The default is a common library or literature value."),
    (NOEFFECT, "Display, caching, file paths or random seeds. Does not change how good the models are."),
]

_LEGEND_TABLE = "\n".join(
    ["| Status | Meaning |", "| --- | --- |"] + [f"| **{status}** | {meaning} |" for status, meaning in DEFAULTS_LEGEND]
)

DEFAULTS_INTRO = """
## How the default settings were chosen

Every form below ends its "Before You Run" note with a **Why these defaults?** table. It compares each option's default with the setting used in the QSARena benchmark (44 datasets, 22 regression and 22 classification, from TDC, MoleculeNet, Polaris, ChemML and PODUAM, all run under **one fixed configuration with no per-dataset tuning**), and says whether that default is backed by the benchmark.

__LEGEND_TABLE__

Two numbers recur below. Per-family cost is the median time to fit one model in the benchmark (A100 GPU, 32 CPU cores; a free Colab runtime is several times slower): conventional ML about 7 s, ChemML MLP about 40 s, MapLight + GNN about 137 s, Chemprop about 269 s per variant, Uni-Mol about 374 s. A "median gap" is how far a model sat from the best single model on each dataset, as a relative difference, taken over the 22 regression or 22 classification datasets (lower is better).

A lighter profile costs little at the level of the *best* model. A second full benchmark run on an RTX 4060 used the same lighter deep-learning settings as this notebook (Chemprop 15 epochs and 1 ensemble member, Uni-Mol 10 epochs). On the 37 datasets with an identical split, its best model differed from the A100 run's best model by a median of 0.17%. That does not mean every individual model is unaffected.

At the end, step `9F` writes an HTML report that records every option each step actually ran with, next to its default and this evidence, so the run can be reproduced.
""".replace("__LEGEND_TABLE__", _LEGEND_TABLE)

# title -> option -> (benchmark setting, status, why)
EVIDENCE: dict[str, dict[str, tuple[str, str, str]]] = {
    "0. Install packages and initialize the tutorial": {
        "persist_outputs_to_google_drive": (NA, NOEFFECT, "Only decides where outputs are saved. Turn on if you want caches and results to survive a Colab reset."),
        "google_drive_output_root": (NA, NOEFFECT, "Output folder on Drive; used only when the option above is on."),
    },
    "0B. Optional: apply a QSARena run.yaml to the widgets": {
        "run_config_yaml_path": (NA, NOEFFECT, "Empty means keep the widget defaults documented here."),
    },
    "1A. Load a dataset": {
        "data_source": (NA, NOEFFECT, "Start with an example so every later step has known-good input."),
        "example_dataset": (NA, EVID, "MoleculeNet FreeSolv (642 molecules) is the smallest built-in regression set, so the whole notebook runs fastest on it. ChemML examples do not load in Colab (Open Babel)."),
        "dataset_file_path": (NA, NOEFFECT, "Used only with the File path source."),
        "default_target_transform": ("auto", BENCH, "The benchmark ran `auto` on every dataset; it kept the raw target on 43/44 and log10-transformed 1."),
        "preview_rows": (NA, NOEFFECT, "Display only."),
    },
    "1B. Choose the SMILES and target columns": {
        "smiles_column": (NA, NOEFFECT, "AUTO detects the column; set it explicitly if detection picks the wrong one."),
        "target_column": (NA, NOEFFECT, "AUTO detects the column; set it explicitly if detection picks the wrong one."),
        "use_only_smiles_target": ("SMILES only", BENCH, "Every benchmark model was built from SMILES-derived features alone."),
        "include_all_columns": ("off", BENCH, "Extra columns were not used in the benchmark. They can leak the answer if they are derived from the target."),
        "include_auxiliary_columns": ("off", BENCH, "Not used in the benchmark. Turn on only for measured covariates you will also have for new molecules."),
        "auxiliary_columns": (NA, NOEFFECT, "Used only when auxiliary columns are on."),
        "preview_selected_rows": (NA, NOEFFECT, "Display only."),
    },
    "1C. Assess missingness and preprocess the selected columns": {
        "missing_value_strategy": (NA, UNTESTED, "No default on purpose: you must choose. `ignore_row` is the safe choice for a missing target, because `zero` and `interpolate` invent measurements."),
        "custom_missing_tokens": (NA, UNTESTED, "Common spreadsheet spellings of a missing value. Add your lab's own codes."),
        "target_transform_strategy": ("auto", BENCH, "Same rule as the benchmark: log-transform only clearly skewed, positive targets."),
        "shifted_log10_epsilon": (NA, UNTESTED, "Used only by `shifted_log10`."),
        "collapse_duplicate_canonical_smiles": (NA, UNTESTED, "Not varied in the benchmark (which used curated sets). Keep it on: a molecule repeated in train and test inflates the test score."),
        "preview_rows_after_cleaning": (NA, NOEFFECT, "Display only."),
    },
    "2A. Preview curated molecules": {
        "preview_molecules": (NA, NOEFFECT, "Display only."),
        "preview_selection_mode": (NA, NOEFFECT, "Display only."),
    },
    "2B. Generate molecular features": {
        "use_morgan_features": ("on (all families built)", LIGHT, "The benchmark built all 11 families and let selection choose. Morgan (ECFP4) was the least enriched family among selected features (0.31x uniform) but is cheap and selected on 38/44 datasets."),
        "use_ecfp6_features": ("on", LIGHT, "Off to keep the matrix small. Enrichment among selected features 0.53x."),
        "use_fcfp6_features": ("on", LIGHT, "Off to keep the matrix small. Enrichment 0.73x."),
        "use_layered_features": ("on", LIGHT, "Off to keep the matrix small. Enrichment 0.48x."),
        "use_atom_pair_features": ("on", LIGHT, "Off to keep the matrix small. Enrichment 0.66x."),
        "use_topological_torsion_features": ("on", LIGHT, "Off to keep the matrix small. Enrichment 0.56x."),
        "use_rdk_path_features": ("on", LIGHT, "Off to keep the matrix small. Enrichment 0.62x."),
        "use_maccs_keys": ("on", EVID, "Small (167 bits) and slightly over-selected (1.18x)."),
        "use_rdkit_descriptors": ("on", EVID, "The most over-selected family in the benchmark (5.16x uniform; picked on 40/44 datasets)."),
        "use_maplight_classic": ("on", EVID, "On, as in the benchmark. Its Avalon and ErG parts were among the most over-selected families (2.5x, 2.9x uniform), it supplied 46% of selected features from 23% of those available, adding it improved the best model on 10/44 datasets, and it enables MapLight CatBoost, the single model with the lowest median regression gap (4.5%). It costs little: about 4.5 s for 642 molecules, the same as the RDKit descriptors."),
        "morgan_radius": ("2", BENCH, "Radius 2 (ECFP4), as in the benchmark."),
        "fingerprint_bits": ("1024", BENCH, "1024 bits, as in the benchmark."),
        "enable_persistent_feature_store": ("on", NOEFFECT, "Caching only: rebuilt features are identical."),
        "reuse_persistent_feature_store": ("on", NOEFFECT, "Caching only."),
        "persistent_feature_store_path": (NA, NOEFFECT, "Cache location."),
    },
    "3A. Build an interactive PCA -> t-SNE map": {
        "tsne_use_morgan_features": (NA, NOEFFECT, "Visual map only; it does not feed any model."),
        "tsne_use_rdkit_descriptors": (NA, NOEFFECT, "Visual map only. Descriptors plus ECFP6 give a map that mixes shape and substructure."),
        "tsne_use_ecfp6_features": (NA, NOEFFECT, "Visual map only."),
        "tsne_use_fcfp6_features": (NA, NOEFFECT, "Visual map only."),
        "tsne_use_layered_features": (NA, NOEFFECT, "Visual map only."),
        "tsne_use_atom_pair_features": (NA, NOEFFECT, "Visual map only."),
        "tsne_use_topological_torsion_features": (NA, NOEFFECT, "Visual map only."),
        "tsne_use_rdk_path_features": (NA, NOEFFECT, "Visual map only."),
        "tsne_use_maccs_keys": (NA, NOEFFECT, "Visual map only."),
        "tsne_use_all_descriptors": (NA, NOEFFECT, "Visual map only."),
        "tsne_morgan_radius": (NA, NOEFFECT, "Visual map only."),
        "tsne_fingerprint_bits": (NA, NOEFFECT, "Visual map only."),
        "perplexity": (NA, NOEFFECT, "Visual map only. 30 is the scikit-learn default."),
        "max_points_for_map": (NA, NOEFFECT, "Visual map only; caps runtime on large sets."),
        "embedding_random_seed": (NA, NOEFFECT, "Seed."),
        "enable_similarity_map_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_similarity_map_cache": (NA, NOEFFECT, "Caching only."),
        "similarity_map_cache_run_name": (NA, NOEFFECT, "Cache name."),
    },
    "4A. Split train/test data": {
        "data_split_strategy": ("official split where one exists", UNTESTED, "The benchmark used each suite's official split on 27/44 datasets, scaffold on 12, target quartiles on 4 and random on 1, so it never compared strategies. Target quartiles keep the full target range in the test set. Use `predefined` for an official split, or `scaffold` for a harder, more realistic test on new chemotypes."),
        "test_fraction": ("0.2", BENCH, "20% held out, as in the benchmark wherever it made its own split."),
        "model_random_seed": ("13", NOEFFECT, "Seed. The benchmark used 13; any fixed value gives a reproducible split."),
    },
    "4A.5. Configure train-only feature selection": {
        "feature_selector_method": ("elasticnet_cv", BENCH, "The benchmark's selector. Selection is fitted on the training split only."),
        "lasso_feature_selection_alpha": (NA, NOEFFECT, "Read only by `fixed_lasso`."),
        "selector_alpha_grid_min_log10": ("-5", BENCH, "Same alpha grid as the benchmark."),
        "selector_alpha_grid_max_log10": ("-1", BENCH, "Same alpha grid as the benchmark."),
        "selector_alpha_grid_size": ("12", BENCH, "Same alpha grid as the benchmark."),
        "elasticnet_l1_ratio_grid": ("0.3, 0.7", BENCH, "Same l1-ratio grid as the benchmark."),
        "selector_cv_folds": ("3", BENCH, "As in the benchmark. The selector took a median of 295 s per dataset with 3 folds on 32 cores; 5 or 10 folds cost proportionally more."),
        "lasso_coefficient_threshold": ("1e-10", BENCH, "As in the benchmark."),
        "lasso_max_iter": ("10000", BENCH, "As in the benchmark."),
        "lasso_coordinate_selection": (NA, UNTESTED, "scikit-learn default (cyclic)."),
        "estimate_selector_runtime": (NA, NOEFFECT, "Prints an estimate only."),
        "selector_auto_rf_by_dataset_size": ("off (A100); on (RTX run)", LIGHT, "Switches to a random-forest selector when ElasticNetCV is predicted to exceed the threshold. On for a laptop or Colab; the A100 run could afford ElasticNetCV everywhere."),
        "selector_auto_rf_threshold_seconds": ("7200", BENCH, "As in the benchmark."),
        "selector_auto_rf_log10_slope": ("1.225", BENCH, "Same timing model as the benchmark config. The A100 run's own timings fit a flatter curve (slope 0.77, intercept -0.30 on 32 cores), so this default overestimates time on a fast machine and falls back to random forest sooner, which is the safe direction on Colab."),
        "selector_auto_rf_log10_intercept": ("-0.658", BENCH, "See the slope row."),
        "max_selected_features": ("0 (10% of training rows)", BENCH, "0 applies the benchmark rule: keep at most 10% of the training molecules' count, ranked by coefficient size. In the benchmark this kept a median of 94 features (up to 1,076 on hERG-Karim). A fixed number such as 512 overrides the rule."),
        "random_forest_selector_trees": ("400 (fallback)", UNTESTED, "Used only by the random-forest selector. The benchmark's fallback used 400 trees; the difference was not tested."),
        "show_lasso_coefficient_diagnostics": (NA, NOEFFECT, "Display only."),
        "top_lasso_coefficients_to_show": (NA, NOEFFECT, "Display only."),
        "enable_feature_selector_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_feature_selector_cache": (NA, NOEFFECT, "Caching only."),
        "selector_cache_run_name": (NA, NOEFFECT, "Cache name."),
    },
    "4C. Train conventional ML models and show an interactive metrics table": {
        "use_cross_validation": ("on", BENCH, "5-fold CV on the training split, as in the benchmark. CV scores are what you should choose a model by; the test set is for the final check."),
        "nested_selection_cv": ("on", EVID, "Refits the feature selector inside every CV fold. Selecting features once on all training rows and then cross-validating overstates CV scores (by 22.7 points for ElasticNetCV and 6.1 for random forest in a controlled test on 9 datasets); test scores are unaffected. Off is faster on large feature matrices."),
        "cv_folds": ("5", BENCH, "As in the benchmark."),
        "model_random_seed": ("13", NOEFFECT, "Seed."),
        "enable_conventional_model_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_conventional_cached_models": (NA, NOEFFECT, "Caching only."),
        "conventional_cache_run_name": (NA, NOEFFECT, "Cache name."),
        "run_elasticnet_cv": ("on", BENCH, "A linear baseline. Weak in the benchmark (median gap 22.7% regression) but fast and interpretable."),
        "run_svr": ("on", BENCH, "Median gap 17.4% regression; SVC 11.4% classification."),
        "run_random_forest": ("on", BENCH, "Strong for classification (median gap 5.9%, 3 wins); 16.0% for regression."),
        "run_extra_trees": ("on", BENCH, "Median gap 17.4% regression, 7.4% classification."),
        "run_hist_gradient_boosting": ("on", BENCH, "Median gap 11.1% regression, 8.2% classification."),
        "run_voting_knn_svr": ("on", BENCH, "Median gap 19.1% regression, 8.7% classification."),
        "run_adaboost": ("on", BENCH, "Among the weakest (29.1% regression, 9.9% classification), but conventional models cost about 7 s each, and keeping the whole library is what wins: 7 of the 10 top-10 leaderboard placements lost under honest selection came from narrowing the model library."),
        "run_tabular_cnn": ("on", BENCH, "The weakest regression model (median gap 47.4%). Kept for completeness; turn it off to save time."),
        "cnn_training_epochs": ("35", CAVEAT, "35, as in the benchmark. When a GPU is found, 4C raises it to at least 60 (and the batch to 128); that GPU setting was not benchmarked."),
        "cnn_batch_size": ("64", CAVEAT, "64, as in the benchmark; raised to 128 when a GPU is found (not benchmarked)."),
        "run_xgboost": ("on", BENCH, "The best conventional regression model (median gap 10.1%); 5.8% classification."),
        "run_lightgbm": ("not run", UNTESTED, "LightGBM was not in the benchmark model set."),
        "run_catboost": ("on", BENCH, "Best conventional classification model (median gap 5.2%); 10.8% regression."),
        "elasticnet_model_alpha_grid_min_log10": ("-4", BENCH, "As in the benchmark."),
        "elasticnet_model_alpha_grid_max_log10": ("0", BENCH, "As in the benchmark."),
        "elasticnet_model_alpha_grid_size": ("12", BENCH, "As in the benchmark."),
        "elasticnet_model_l1_ratio_grid": ("0.4, 0.8", BENCH, "As in the benchmark."),
        "elasticnet_model_cv_folds": ("3", BENCH, "As in the benchmark."),
        "elasticnet_model_max_iter": ("15000", BENCH, "As in the benchmark."),
    },
    "4E. Run genetic-algorithm tuning for selected conventional models": {
        "ga_cv_folds": ("GA off", UNTESTED, "Used only if you tick a tune_* box."),
        "ga_objective": ("not run", UNTESTED, "RMSE matches how regression winners are ranked elsewhere in this notebook."),
        "ga_generations": ("not run", UNTESTED, "Small search so it finishes in a notebook session."),
        "ga_population_size": ("not run", UNTESTED, "Small search so it finishes in a notebook session."),
        "ga_crossover_size": ("not run", UNTESTED, "Small search so it finishes in a notebook session."),
        "ga_mutation_size": ("not run", UNTESTED, "Small search so it finishes in a notebook session."),
        "ga_mutation_probability": ("not run", UNTESTED, "Not tested."),
        "ga_early_stopping": ("not run", UNTESTED, "Stops after 4 generations without improvement."),
        "enable_tuned_model_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_tuned_cached_models": (NA, NOEFFECT, "Caching only."),
        "tuned_cache_run_name": (NA, NOEFFECT, "Cache name."),
        "previous_ga_run_source": (NA, NOEFFECT, "Reuses an earlier GA run."),
        "upload_previous_ga_run_source": (NA, NOEFFECT, "Reuses an earlier GA run."),
        "tune_elasticnet": ("off (GA disabled)", BENCH, "Off: GA tuning was disabled for all 44 benchmark datasets, and every published result uses untuned models. It multiplies training cost by roughly generations x population x folds (8 x 16 x 5 = 640 fits per model here) for small, unmeasured gains. 4E, 4F and 4G skip when no box is ticked."),
        "tune_svr": ("off (GA disabled)", BENCH, "Off: GA tuning was disabled for all 44 benchmark datasets, and every published result uses untuned models. It multiplies training cost by roughly generations x population x folds (8 x 16 x 5 = 640 fits per model here) for small, unmeasured gains. 4E, 4F and 4G skip when no box is ticked."),
        "tune_random_forest": ("off (GA disabled)", BENCH, "Off: GA tuning was disabled for all 44 benchmark datasets, and every published result uses untuned models. It multiplies training cost by roughly generations x population x folds (8 x 16 x 5 = 640 fits per model here) for small, unmeasured gains. 4E, 4F and 4G skip when no box is ticked."),
        "tune_xgboost": ("off (GA disabled)", BENCH, "Off: GA tuning was disabled for all 44 benchmark datasets, and every published result uses untuned models. It multiplies training cost by roughly generations x population x folds (8 x 16 x 5 = 640 fits per model here) for small, unmeasured gains. 4E, 4F and 4G skip when no box is ticked. If you do tune one model, XGBoost (the best conventional regression model) is the most likely to benefit."),
        "tune_catboost": ("off (GA disabled)", BENCH, "Off: GA tuning was disabled for all 44 benchmark datasets, and every published result uses untuned models. It multiplies training cost by roughly generations x population x folds (8 x 16 x 5 = 640 fits per model here) for small, unmeasured gains. 4E, 4F and 4G skip when no box is ticked."),
    },
    "4G. Plot GA convergence history for tuned conventional models": {
        "show_ga_history_elasticnet": (NA, NOEFFECT, "Display only."),
        "show_ga_history_svr": (NA, NOEFFECT, "Display only."),
        "show_ga_history_random_forest": (NA, NOEFFECT, "Display only."),
        "show_ga_history_xgboost": (NA, NOEFFECT, "Display only."),
        "show_ga_history_catboost": (NA, NOEFFECT, "Display only."),
    },
    "5B. Train selected deep-learning models": {
        "run_chemml_pytorch": ("on", BENCH, "Included as the tutorial's simple neural network (about 40 s). ChemML MLPs were the weakest family in the benchmark (median gap 20.7% regression, 12.4% classification)."),
        "run_chemml_tensorflow": ("on", LIGHT, "Off to save time. It was the better ChemML backend for classification (7.5% vs 12.4%); consider it there."),
        "run_maplight_gnn": ("on", LIGHT, "Off because DGL often fails to install in Colab. When it runs it is strong for regression (4 wins, median gap 8.0%) but weak for classification (11.5%); about 137 s per dataset."),
        "run_tabpfn_deep": ("off (A100); on (later run)", LIGHT, "Off because it needs a Prior Labs API key. Where it ran, TabPFN was the second-best single regression model (6 wins, median gap 5.2%); 7.3% classification. Worth enabling for regression."),
        "run_tabpfn_local": (NA, UNTESTED, "Runs TabPFN locally instead of through the API; needs the model weights and ideally a GPU."),
        "tabpfn_max_train_rows": ("1000", BENCH, "As in the canonical run; the later run used 11,000. Larger sets are subsampled to this size."),
        "chemml_hidden_layers": ("2", BENCH, "As in the benchmark."),
        "chemml_hidden_width": ("256", LIGHT, "Half the benchmark width, for CPU runtimes."),
        "chemml_training_epochs": ("80", LIGHT, "Fewer than the benchmark's 80, for CPU runtimes."),
        "chemml_batch_size": ("64", BENCH, "As in the benchmark."),
        "chemml_learning_rate": ("0.001", BENCH, "As in the benchmark."),
        "chemml_use_cross_validation": ("on", BENCH, "As in the benchmark."),
        "chemml_cv_folds": ("5", BENCH, "As in the benchmark."),
        "deep_random_seed": (NA, NOEFFECT, "Seed."),
        "enable_deep_model_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_deep_cached_models": (NA, NOEFFECT, "Caching only."),
        "deep_cache_run_name": (NA, NOEFFECT, "Cache name."),
    },
    "5C. Compare conventional and deep-learning performance": {
        "comparison_scatter_top_n": (NA, NOEFFECT, "Display only."),
        "include_tuned_models_in_comparison": (NA, NOEFFECT, "Display only."),
    },
    "6A. Install Uni-Mol packages (optional; may restart the runtime)": {
        "force_reinstall_unimol": (NA, NOEFFECT, "Only for repairing a broken install."),
    },
    "6B. Prepare Uni-Mol train and test files from the QSAR split": {
        "enable_unimol_model_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_unimol_cached_models": (NA, NOEFFECT, "Caching only."),
        "unimol_run_label": (NA, NOEFFECT, "Cache name."),
        "unimol_train_subset_size": ("full training set", BENCH, "0 uses every training molecule, as the benchmark did. A subset is faster but changes the model."),
        "unimol_prepare_random_seed": (NA, NOEFFECT, "Seed."),
    },
    "6C. Train and evaluate Uni-Mol V1": {
        "unimol_internal_split": ("random", BENCH, "As in the benchmark."),
        "unimol_epochs": ("20", LIGHT, "Half the A100 setting; the RTX run used 10 (see the note at the top). Uni-Mol V1 was the most consistent family: median gap 6.8% regression and 1.8% classification, results on all 44 datasets."),
        "unimol_learning_rate": ("1e-4", BENCH, "As in the benchmark."),
        "unimol_batch_size": ("32", BENCH, "As in the A100 run (the RTX run used 16). Lower it if the GPU runs out of memory."),
        "unimol_early_stopping": ("5", BENCH, "As in the benchmark."),
        "unimol_num_workers": ("8", NOEFFECT, "Data-loading workers only. 0 avoids worker crashes in Colab and on Windows."),
    },
    "6D. Train and evaluate Uni-Mol V2": {
        "filter_unimolv2_incompatible": (NA, UNTESTED, "Drops molecules Uni-Mol V2 cannot embed instead of failing; this changes the dataset, so it is off."),
        "unimol_internal_split": ("random", BENCH, "As in the benchmark."),
        "unimol_epochs": ("20", LIGHT, "Half the benchmark setting, for notebook runtimes."),
        "unimol_learning_rate": ("1e-4", BENCH, "As in the benchmark."),
        "unimol_batch_size": ("32", BENCH, "As in the benchmark."),
        "unimol_early_stopping": ("5", BENCH, "As in the benchmark."),
        "unimol_model_size": ("84m", BENCH, "The only size benchmarked. Best single classification model where it ran (median gap 0.9%, 7 datasets), but 13.9% for regression."),
        "unimol_max_atoms": ("64", UNTESTED, "Larger than the benchmark's 64, so fewer large molecules are truncated, at the cost of GPU memory. Use 64 if memory is tight."),
        "unimol_use_amp": ("on", BENCH, "Mixed precision, as in the benchmark."),
        "unimol_num_workers": ("8", NOEFFECT, "Data-loading workers only. 0 avoids worker crashes in Colab and on Windows."),
    },
    "6E. Install Chemprop v2 packages (optional; may restart the runtime)": {
        "force_reinstall_chemprop": (NA, NOEFFECT, "Only for repairing a broken install."),
    },
    "6F. Train and evaluate Chemprop v2 graph variants": {
        "enable_chemprop_model_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_chemprop_cached_models": (NA, NOEFFECT, "Caching only."),
        "chemprop_run_label": (NA, NOEFFECT, "Cache name."),
        "run_chemprop_dmpnn": ("on", EVID, "The best Chemprop variant for regression (4 wins, median gap 9.0%); mid-table for classification (5.5%). On by default because this notebook's examples are regression tasks."),
        "run_chemprop_cmpnn": ("on", LIGHT, "No wins in either task (10.8% regression, 7.0% classification)."),
        "run_chemprop_attentivefp": ("on", EVID, "The best Chemprop variant for classification (4 wins, median gap 2.9%); 11.6% for regression."),
        "run_chemprop_rdkit2d_extra": ("on", LIGHT, "Second-best Chemprop variant for classification (3.6%)."),
        "run_chemprop_selected_features": ("on", EVID, "Off: the weakest Chemprop variant in both tasks (median gap 21.2% regression vs 9.0% for plain D-MPNN; 5.0% classification), with no wins on any of 44 datasets. Adding the selected descriptors to the graph model made it worse, not better, so it is not worth the extra ~4.5 min per run. Turn it on only to study descriptor-augmented graphs."),
        "chemprop_epochs": ("40", LIGHT, "The RTX run used 15, like this default. That comparison does not isolate Chemprop, so use 40 when Chemprop is the model you care about."),
        "chemprop_batch_size": ("32", BENCH, "As in the benchmark."),
        "chemprop_num_workers": ("8", NOEFFECT, "Data-loading workers only. Must stay 0 on Windows, where each worker reloads CUDA and can exhaust memory."),
        "chemprop_ensemble_size": ("3", LIGHT, "The benchmark averaged 3 models per variant; 1 is three times faster."),
        "chemprop_random_seed": ("42", BENCH, "As in the benchmark."),
    },
    "6G. Plot observed vs predicted values for selected Uni-Mol models": {
        "show_unimolv1_plot": (NA, NOEFFECT, "Display only."),
        "show_unimolv2_plot": (NA, NOEFFECT, "Display only."),
    },
    "7A. Build an optional ensemble from trained models": {
        "build_ensemble": ("on", CAVEAT, "Fusion helped classification (beat the best single model on 10/22 datasets, by a median 1.8% when it did) but rarely regression (1/22). The benchmark chose ensemble members using test scores, so these figures are provisional until its out-of-fold rebuild; this notebook already uses out-of-fold predictions."),
        "run_oof_stacking_ensemble": ("on", BENCH, "The benchmark's main ensemble; 8 classification wins (provisional, see above)."),
        "run_weighted_inverse_rmse_ensemble": ("on", BENCH, "As in the benchmark."),
        "run_cfa_ensemble": ("on", BENCH, "Combinatorial fusion improved the best model on 3/44 datasets: 2 classification wins, none for regression. Nearly free once base models exist."),
        "stacking_cv_folds": ("5", BENCH, "As in the benchmark."),
        "stacking_random_seed": (NA, NOEFFECT, "Seed."),
        "ensemble_oof_folds": ("5", EVID, "Members are weighted on out-of-fold predictions. The benchmark showed that weighting on training predictions gives almost all weight to memorising models such as extra trees; out-of-fold predictions prevent that."),
        "exclude_negative_oof_r2_members": ("on (test R2)", EVID, "The benchmark dropped members with negative *test* R2, which leaks the test set. This applies the same filter to out-of-fold R2."),
        "drop_highly_correlated_members": ("on", BENCH, "As in the benchmark."),
        "max_train_prediction_correlation": ("0.995", UNTESTED, "Stricter than the benchmark's 0.995, so near-duplicate members are dropped more often. Not compared."),
        "include_conventional": ("on", BENCH, "The benchmark pooled all workflows."),
        "include_tuned_conventional": ("on", BENCH, "Used only if 4E ran."),
        "include_unimol": ("on", BENCH, "Uni-Mol members use its own saved internal-fold predictions."),
        "cfa_best_per_workflow_only": ("on", BENCH, "As in the benchmark."),
        "cfa_optimize_metric": ("mae", BENCH, "MAE, as in the benchmark. It sets both the per-workflow member choice and the metric the CFA search minimises."),
        "cfa_max_models": ("0", BENCH, "No limit, as in the benchmark."),
        "cfa_max_candidate_subsets": ("250000", LIGHT, "Smaller search than the benchmark, for notebook runtimes."),
    },
    "8A. Configure explanation settings for the best conventional model": {
        "top_features_to_show": (NA, NOEFFECT, "Display only."),
        "explanation_sample_size": (NA, NOEFFECT, "Explanation runtime vs stability; does not change the model."),
        "explanation_random_seed": (NA, NOEFFECT, "Seed."),
        "try_shap_for_tree_models": (NA, NOEFFECT, "Off because SHAP is slow on large tree ensembles; permutation importance is used instead."),
        "permutation_repeats": (NA, NOEFFECT, "More repeats give steadier importances; does not change the model."),
    },
    "8C. Explain the ChemML deep model": {
        "explanation_family": (NA, NOEFFECT, "Explanation method only."),
        "explanation_scope": (NA, NOEFFECT, "Explanation method only."),
        "lrp_strategy": (NA, NOEFFECT, "Used only by LRP."),
        "test_instance_index": (NA, NOEFFECT, "Which molecule to explain."),
        "background_reference_size": (NA, NOEFFECT, "Explanation runtime vs stability."),
        "global_explanation_samples": (NA, NOEFFECT, "Explanation runtime vs stability."),
        "lime_instances_to_show": (NA, NOEFFECT, "Display only."),
        "top_deep_features_to_show": (NA, NOEFFECT, "Display only."),
    },
    "9A. Predict from a trained model using a new SMILES table": {
        "prediction_workflow": ("test-best (benchmark)", EVID, "Chooses the model with the lowest CV RMSE (OOF RMSE for ensembles) on the training split, never the test set. Choosing by test score makes that score optimistic: in the benchmark the CV-chosen model sat a median 8.7-16.1% behind the test-best one, which is the honest cost of not peeking. Models without a CV score (Uni-Mol, Chemprop) are listed and can be named below. Test RMSE is used only if no model has a CV score."),
        "prediction_model_name": (NA, NOEFFECT, "Empty uses the best model of the chosen workflow."),
        "prediction_input_source": (NA, NOEFFECT, "Five built-in molecules so 9A runs without a file. They are a smoke test, not a meaningful prediction set."),
        "prediction_input_path": (NA, NOEFFECT, "Used only with the File path source."),
        "prediction_smiles_column": (NA, NOEFFECT, "Column name in the input file."),
        "save_prediction_output": (NA, NOEFFECT, "Output only."),
        "prediction_output_path": (NA, NOEFFECT, "Output only."),
    },
    "9B. Build an interactive UMAP for train, test, and new molecules": {
        "umap_neighbors": (NA, NOEFFECT, "Visual map only."),
        "umap_min_dist": (NA, NOEFFECT, "Visual map only; 0.1 is the UMAP default."),
        "umap_metric": (NA, NOEFFECT, "Visual map only."),
        "umap_random_seed": (NA, NOEFFECT, "Seed."),
        "enable_prediction_umap_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_prediction_umap_cache": (NA, NOEFFECT, "Caching only."),
        "prediction_umap_cache_run_name": (NA, NOEFFECT, "Cache name."),
    },
    "9D. Fit and apply the MAST-ML / MADML applicability-domain workflow": {
        "install_mastml_if_missing": (NA, NOEFFECT, "Installation only."),
        "mastml_random_forest_estimators": (NA, UNTESTED, "The MAST-ML domain model was not part of the benchmark."),
        "mastml_n_repeats": (NA, UNTESTED, "More repeats are slower but steadier."),
        "mastml_bins": (NA, UNTESTED, "Not tested."),
        "mastml_kernel": (NA, UNTESTED, "Not tested."),
        "mastml_use_custom_bandwidth": (NA, UNTESTED, "Off lets MAST-ML pick the bandwidth."),
        "mastml_bandwidth": (NA, UNTESTED, "Used only with a custom bandwidth."),
        "ad_enable_knn_distance": (NA, CAVEAT, "In the paper's reliability study (reference random forest, 22 TDC datasets) structural domain flags were weak: molecules flagged by kNN Tanimoto distance had only 1.15x the error of in-domain ones, while uncertainty-based flags had 3.30x. Treat a structural out-of-domain flag as a warning, not proof of a bad prediction."),
        "ad_knn_neighbors": (NA, UNTESTED, "Common choice; not tuned."),
        "ad_knn_in_domain_quantile": (NA, UNTESTED, "The 5% of training molecules farthest from their neighbours define the boundary."),
        "ad_enable_mahalanobis": (NA, CAVEAT, "Descriptor-space distance; same caveat as kNN."),
        "ad_mahalanobis_in_domain_quantile": (NA, UNTESTED, "Not tuned."),
        "ad_consensus_low_support_ratio": (NA, UNTESTED, "Not tuned."),
        "enable_mastml_ad_cache": (NA, NOEFFECT, "Caching only."),
        "reuse_mastml_ad_cache": (NA, NOEFFECT, "Caching only."),
        "mastml_ad_cache_run_name": (NA, NOEFFECT, "Cache name."),
        "save_mastml_ad_output": (NA, NOEFFECT, "Output only."),
        "mastml_ad_output_path": (NA, NOEFFECT, "Output only."),
    },
    "9E. Export the widget choices as run.yaml": {
        "export_run_yaml_path": (NA, NOEFFECT, "Output only."),
    },
    "9F. Export notebook results to an HTML report": {
        "html_report_path": (NA, NOEFFECT, "Output only."),
        "html_report_title": (NA, NOEFFECT, "Output only."),
        "include_model_result_tables": (NA, NOEFFECT, "Report content only."),
        "include_prediction_outputs": (NA, NOEFFECT, "Report content only."),
        "include_applicability_domain_outputs": (NA, NOEFFECT, "Report content only."),
        "max_html_report_table_rows": (NA, NOEFFECT, "Report size only."),
        "download_html_report_in_colab": (NA, NOEFFECT, "Download only."),
    },
}


def _format_default(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        return f"`{value}`" if value else "(empty)"
    return f"`{value}`"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def default_evidence_records(title: str, schema: list[dict]) -> dict[str, dict]:
    """option -> {default, benchmark, status, why} for one form, in form order (for the 9F report)."""
    entries = EVIDENCE.get(title, {})
    records: dict[str, dict] = {}
    for entry in schema:
        name = entry.get("name")
        if not name:
            continue
        bench, status, why = entries.get(name, (NA, "", ""))
        records[name] = {
            "default": entry.get("default"),
            "benchmark": "" if bench == NA else bench,
            "status": status,
            "why": why,
        }
    return records


def render_default_evidence(title: str, schema: list[dict]) -> str | None:
    """Markdown 'Why these defaults?' table for a form, or None if it has no options.

    Raises ValueError when the form and EVIDENCE disagree, so the builder cannot ship a form
    option without a documented default.
    """
    params = [entry for entry in schema if entry.get("name")]
    if not params:
        return None
    entries = EVIDENCE.get(title)
    if entries is None:
        raise ValueError(f"notebook_default_evidence: no EVIDENCE entry for form {title!r}")
    names = [entry["name"] for entry in params]
    missing = [name for name in names if name not in entries]
    extra = [name for name in entries if name not in names]
    if missing or extra:
        raise ValueError(
            f"notebook_default_evidence: form {title!r} is out of sync; "
            f"undocumented options={missing}, stale entries={extra}"
        )
    rows = [
        "#### Why these defaults?",
        "",
        "| Option | Default | Benchmark | Status | Why |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in params:
        bench, status, why = entries[entry["name"]]
        bench_text = "" if bench == NA else _cell(bench)
        rows.append(
            f"| `{entry['name']}` | {_cell(_format_default(entry.get('default')))} | {bench_text} "
            f"| **{status}** | {_cell(why)} |"
        )
    return "\n".join(rows)
