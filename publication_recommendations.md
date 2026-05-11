# AutoQSAR: Publication Recommendations

## Executive Assessment

AutoQSAR is publishable as a methods/tool paper. The contribution is strongest when framed around the **practical finding** that a well-engineered conventional ML pipeline with ensemble fusion and rich molecular features achieves competitive-to-SOTA performance across 37 benchmark datasets, delivered through an accessible open-source tool requiring no deep learning expertise or GPU infrastructure.

However, two claims require correction before submission: the ESOL and MoleculeNet Lipophilicity "beat SOTA" findings are compared against outdated 2017 MoleculeNet baselines (GCN RMSE 0.885 for ESOL; GCN RMSE 0.781 for Lipophilicity), not current state of the art. Modern models achieve ESOL RMSE ~0.56 (PrismNet, *Adv Sci* 2026) and Lipophilicity RMSE ~0.55 (PrismNet). AutoQSAR's scores of 0.621 and 0.595 are strong but rank approximately 5th–6th against current published results rather than 1st. See the companion ESOL/Lipophilicity benchmark CSV for full details.

---

## Strengths

### 1. Breadth of Validation

37 datasets across 5 benchmark suites (TDC ADMET, MoleculeNet, Polaris, PODUAM, ChemML). Most published papers benchmark on a single suite. For context:

- ADMETboost (Tian et al. 2022, *J Mol Model* 28:408): 22 TDC tasks only
- ADMET-AI (Swanson et al. 2024, *Bioinformatics* 40:btae416): 41 TDC tasks
- MolGPS (Sypetkowski et al. 2024, NeurIPS 2024): 38 tasks across TDC + Polaris + MoleculeNet
- MolE (Méndez-Lucio et al. 2024, *Nat Commun* 15:10021): 22 TDC tasks

AutoQSAR's cross-suite coverage is comparable to MolGPS and exceeds most other entries.

### 2. Consistent Competitive Performance

Top-10 on 35/37 datasets. Confirmed new best on 5 datasets where the reference benchmarks are on comparable scaffold splits:

| Dataset | AutoQSAR | Previous Best | Delta |
|---------|----------|---------------|-------|
| tdc_skin_reaction | 0.775 AUROC | 0.741 (FATE-Tox MTL) | +0.034 |
| tdc_bioavailability_ma | 0.777 AUROC | 0.748 (MaxQsaring) | +0.029 |
| tdc_carcinogens_lagunin | 0.863 AUROC | 0.848 (FATE-Tox MTL) | +0.015 |
| tdc_toxcast | 0.791 AUROC | 0.714 (PrismNet, MoleculeNet split) | +0.077 |
| tdc_hydrationfreeenergy_freesolv | 0.645 RMSE | 0.654 (PrismNet, MoleculeNet split) | -0.009 |

Note: the tdc_toxcast and tdc_hydrationfreeenergy_freesolv comparisons involve different split protocols (TDC scaffold vs. MoleculeNet scaffold) — this should be acknowledged.

### 3. Conventional ML Matching Foundation Models

The best model per dataset is frequently a conventional model (CatBoost, Random Forest, AdaBoost) or an ensemble (CFA, OOF stacking), not a deep graph neural network. This directly engages a live debate:

- Kamuntavicius et al. (2025, *J Cheminform* 17:108) found rdkit_desc to be the superior standalone feature representation, with optimal model and feature choices being highly dataset-dependent.
- Koleiev et al. (2026, bioRxiv, DOI: 10.64898/2026.02.26.708193) found only 3 of the top TDC models passed all reproducibility checks: CaliciBoost, MapLight, and MapLight+GNN — all tree-based methods.
- MaxQsaring (Xu et al. 2025, *J Pharm Anal* 15:101411) achieved rank 1 on 19/22 TDC tasks using automatic feature combination with simple ML algorithms to ensure the intrinsic interpretability of the models.

AutoQSAR's results substantially expand the evidence base for this finding.

### 4. Practical Accessibility

A code-free Jupyter notebook runnable in Google Colab with no GPU, achieving competitive results, is a genuine contribution to democratizing ADMET modeling.

---

## Weaknesses To Address

### 1. No Single Novel Method

The winning model varies across datasets (CatBoost, TabPFN, MapLight+GNN, CFA, ensembles). Reviewers will ask: *"What is the specific methodological contribution?"* Frame this as a feature, not a bug: the contribution is the **empirical finding** that a well-designed AutoML pipeline with best-per-task model selection matches specialized architectures.

### 2. Outdated MoleculeNet References

The ESOL and MoleculeNet Lipophilicity SOTA claims must be corrected. The MoleculeNet leaderboard only contains 2017-era baselines (GCN, Random Forest). Against current published results:

- **ESOL**: AutoQSAR 0.621 ranks ~5th (current SOTA: PrismNet 0.558, HiGNN 0.570, DMPNN 0.575)
- **Lipophilicity**: AutoQSAR 0.595 ranks ~6th (current SOTA: PrismNet 0.549, GRAPHMSL 0.562, MV-Mol 0.566)

These are still strong results — top-10 in a competitive field — but they are not SOTA. Present them honestly.

### 3. Single-Run Results

The TDC standard is mean +/- standard deviation across 5 independent runs with different seeds. Submitting single-run results will draw criticism. Add multi-seed evaluation for at least the TDC-22 datasets.

### 4. Split Protocol Mismatches

Some "beat SOTA" claims compare TDC scaffold splits against MoleculeNet scaffold splits or literature-reported splits that may differ. For TDC-22 datasets, use the official `admet_group` benchmark splits to ensure comparability.

---

## Initial Recommendation Shortlist

The more detailed notebook crosswalk below supersedes this short list and separates completed, partially implemented, and missing analyses.

### Critical (Required)

1. **Multi-seed evaluation** (5 seeds) for all TDC-22 datasets using official `admet_group` splits
2. **Computational cost table** comparing wall-clock time and hardware for AutoQSAR vs. MolGPS/MolE/ADMET-AI

### Strongly Recommended

3. **Ablation study**: Run all 37 datasets with:
   - (a) Conventional ML only (no DL, no ensemble)
   - (b) + MapLight features
   - (c) + Deep learning backends (Chemprop, TabPFN, Uni-Mol)
   - (d) + CFA/ensemble fusion
   - This demonstrates which components drive the gains.

### Optional (Strengthens Paper)

4. **Applicability domain analysis**: Existing workflow components may produce AD diagnostics; add a cross-dataset uncertainty/error calibration summary.

5. **Comparison with ADMET-AI web server**: Since ADMET-AI is the most accessible current competitor, a head-to-head comparison on overlapping datasets would be compelling.

---

## Notebook Crosswalk and Remaining Analysis Gaps

The targeted recommendations were cross-referenced against `portable_colab_qsar_bundle/benchmark_results_summary.ipynb`. The summary notebook already covers a substantial part of the publication analysis layer, but several items remain incomplete or only partially implemented.

| Recommendation | Notebook status | Remaining publication gap |
|---|---|---|
| Multi-seed TDC-22 evaluation | **Not implemented** | Official TDC-22 train/test splits are already being used for the 22 admet_group datasets in the current workflow. The missing piece is full 5-seed workflow execution and mean +/- std aggregation across seeds 1-5, using TDC-compatible evaluation outputs. |
| Computational cost comparison | **Implemented** | Runner now emits inference-only timing and neural parameter-count fields for future benchmark artifacts; notebook builds a manuscript-oriented model-family cost table with current-run wall-clock time, hardware notes, parameter-count notes, inference-time fields, and MolGPS/MolE/ADMET-AI comparator rows. Existing benchmark artifacts still lack the newly added timing/parameter fields until rerun. |
| Updated leaderboard references | **Implemented from local curated references** | Notebook now combines run references, MaxQsaring-containing TDC ADMET references, and current ESOL/Lipophilicity literature references into publication comparison artifacts. |
| Ablation/component contribution | **Implemented from existing metrics** | Notebook now computes staged best-achievable performance from conventional ML through MapLight, deep backends, CFA, and ensemble/full-pipeline stages. |
| Per-dataset best-model breakdown | **Implemented** | Notebook now reports win counts by family, a pie chart, a family-by-dataset rank heatmap, and a winning-model table. |
| Dataset difficulty stratification | **Implemented from existing artifacts** | Notebook now builds dataset descriptors, class balance, optional RDKit scaffold diversity, rank/difficulty summaries, and descriptor/outcome correlations. |
| Feature-family importance | **Implemented with artifact fallback** | Notebook now reports feature-family importance shares and ranks by dataset. It uses `model_feature_importances.csv` when future runs provide it and falls back to selector coefficient/importances for existing artifacts. |
| Ensemble value-add deep dive | **Implemented from existing metrics** | Notebook now reports primary-metric-aware CFA/ensemble value-add for regression and classification, including top-3 rates, wins, base-model deltas, and competitive base-model counts. |
| Head-to-head with ADMET-AI | **Not implemented** | No ADMET-AI supplementary-result comparison is currently included. |
| Applicability-domain calibration | **Not implemented in summary notebook** | Existing workflow may produce AD diagnostics, but the summary notebook does not aggregate uncertainty/AD score vs. actual error. |
| Temporal robustness | **Not implemented** | No scaffold-vs-temporal comparison is currently included. |
| Reproducibility audit | **Implemented as manifest/checklist** | Notebook now creates dataset-level artifact checksums, split-hash coverage, repo-file checksums, and a Zenodo-ready archive file list. |
| Classification metric expansion | **Implemented** | The summary notebook now uses dataset-specific primary classification metrics: the designated TDC CYP tasks rank by `test_auprc`, while other classification datasets use declared primary metrics or AUROC fallback. AUROC, AUPRC, balanced accuracy, and MCC remain visible as secondary metrics. |

### Completed in `benchmark_results_summary.ipynb`

- **TDC-primary classification metrics**: Implemented row-specific primary metric selection, including AUPRC for CYP2C9_Veith, CYP2D6_Veith, CYP3A4_Veith, CYP2C9_Substrate, CYP2D6_Substrate, and CYP3A4_Substrate.
- **Benchmark artifact instrumentation**: Updated `run_autoqsar_ga_benchmarks.py` so future metrics include train/test inference-only timing, per-1000-molecule inference rates, and neural parameter-count fields where the backend exposes exact counts.
- **Publication-grade computational cost table**: Added a model-family cost comparison table with AutoQSAR current-run runtimes and hardware/parameter notes, plus MolGPS, MolE, and ADMET-AI comparator rows. Future reruns will populate the new inference and parameter-count fields.
- **Per-dataset best-model breakdown**: Added win counts by model family, a pie chart, rank-by-family heatmap, and winning-model table.
- **Feature-family importance analysis**: Added feature-family importance summaries and heatmaps. The notebook consumes future per-model importance artifacts when present and otherwise uses selector coefficient/importances as a fallback for the current run.
- **Ensemble value-add context**: Added primary-metric-aware CFA/ensemble comparisons against the best base model, including classification and regression summaries, top-3 rates, and per-dataset deltas.
- **Component ablation from existing metrics**: Added staged cumulative performance analysis for conventional ML, MapLight classic features, neural/deep backends, CFA, and full ensemble pipeline.
- **Dataset difficulty stratification**: Added dataset descriptor, class-balance, optional scaffold-diversity, winning-family, and descriptor/outcome correlation summaries.
- **Reproducibility audit and archive manifest**: Added artifact presence checks, SHA-256 checksums, split hashes, git metadata, repo-file manifest, and a data-availability statement.
- **Updated leaderboard reference audit**: Added publication reference tables combining run references, curated TDC ADMET references including MaxQsaring, and current ESOL/Lipophilicity literature references.

### Implemented No-Rerun Publication Analyses

These items are implemented from existing benchmark artifacts and notebook logic without rerunning model training. The notebook writes publication-ready CSV artifacts into the selected benchmark run directory when rerun.

1. **Ensemble value-add context**: Extend the current ensemble diagnostics beyond regression win counts.
   - Status: implemented in `benchmark_results_summary.ipynb`.
   - Output artifacts: `publication_ensemble_value_add_summary.csv`, `publication_ensemble_value_add_by_dataset.csv`.
   - Source artifacts: per-dataset `metrics.csv`, `predictions.csv`, and the notebook's existing best-model/rank tables.
   - Step 1: Identify ensemble and CFA rows using `workflow` and model-name patterns.
   - Step 2: For each dataset, compute the best base-model score, best ensemble/CFA score, delta vs. best base, and rank of each ensemble/CFA row.
   - Step 3: Split results by task type so classification and regression are summarized separately using each dataset's primary metric.
   - Step 4: Count how often ensembles/CFA win, how often they are top-3, and how often they underperform the best base model.
   - Step 5: Add a manuscript-ready table and compact figure in `benchmark_results_summary.ipynb`; update this file when complete.

2. **Component ablation from existing metrics**: Compute staged best-achievable performance per dataset from already-run model families.
   - Status: implemented in `benchmark_results_summary.ipynb`.
   - Output artifacts: `publication_component_ablation_summary.csv`, `publication_component_ablation_by_dataset.csv`.
   - Source artifacts: existing per-model metric rows and architecture-family mappings in `benchmark_results_summary.ipynb`.
   - Step 1: Define ordered stages: conventional ML only; conventional ML plus MapLight classic features; plus neural/deep backends; plus CFA/ensemble fusion; full pipeline.
   - Step 2: Map each model/workflow row to one or more stages based on `workflow`, `model`, and architecture-family labels.
   - Step 3: For each dataset and stage, calculate the best primary metric available up to that stage.
   - Step 4: Calculate marginal improvement from the previous stage using metric directionality.
   - Step 5: Add a waterfall or cumulative-improvement figure plus a table of stage win/improvement counts.

3. **Dataset difficulty stratification**: Relate performance patterns to dataset characteristics.
   - Status: implemented in `benchmark_results_summary.ipynb`.
   - Output artifacts: `publication_dataset_difficulty_stratification.csv`, `publication_dataset_difficulty_correlations.csv`, `publication_winning_family_difficulty_profile.csv`.
   - Source artifacts: dataset-level metrics, split sizes, task type, predictions, benchmark catalog metadata, and SMILES in prediction artifacts.
   - Step 1: Build a dataset descriptor table with `n_train`, `n_test`, task type, primary metric, best observed score, and leaderboard delta where available.
   - Step 2: For classification datasets, compute class balance from observed labels in train/test predictions or source labels.
   - Step 3: Compute optional scaffold diversity from test/train SMILES using RDKit Bemis-Murcko scaffolds; this is feature analysis only, not a benchmark rerun.
   - Step 4: Join descriptors to winning model family and rank/delta summaries.
   - Step 5: Add scatterplots and correlation tables, such as dataset size vs. leaderboard delta colored by winning family.

4. **Reproducibility audit and archive manifest**: Build the data-availability checklist from existing outputs.
   - Status: implemented in `benchmark_results_summary.ipynb`.
   - Output artifacts: `publication_reproducibility_manifest.csv`, `publication_reproducibility_repo_files.csv`, `publication_reproducibility_summary.csv`, `publication_zenodo_archive_dataset_file_list.csv`, `publication_data_availability_statement.txt`.
   - Source artifacts: `run_config.json`, per-dataset metrics/predictions/selected features, leaderboard references, environment files, and git metadata.
   - Step 1: Generate a manifest of required files and mark present/missing status for each dataset.
   - Step 2: Record git commit hash, run timestamp, configured seeds, split hashes, input file checksums, and benchmark artifact checksums.
   - Step 3: Add a notebook/report table summarizing reproducibility coverage and known gaps.
   - Step 4: Prepare a Zenodo-ready file list and data-availability statement.
   - Step 5: Archive upload and DOI minting can happen later; the manifest/checklist does not require rerunning benchmarks.

5. **Updated leaderboard references**: Improve comparison accuracy without rerunning AutoQSAR.
   - Status: implemented in `benchmark_results_summary.ipynb` using local curated reference tables.
   - Output artifacts: `publication_leaderboard_top10_reference.csv`, `publication_leaderboard_comparison_by_dataset.csv`, `publication_leaderboard_reference_audit.csv`.
   - Source artifacts: `data/benchmark_dataset_catalog.csv`, `leaderboard_top10_reference.json`, `leaderboard_top10_reference.csv`, and current literature/reference tables.
   - Step 1: Add MaxQsaring references across TDC-22 where available.
   - Step 2: Replace legacy MoleculeNet-only ESOL and Lipophilicity comparators with current published references.
   - Step 3: Regenerate `leaderboard_top10_reference.csv` and `leaderboard_comparison_by_dataset.csv`.
   - Step 4: Rerun only the summary notebook cells that consume reference tables.
   - Step 5: Update claims in this recommendations document if rank estimates change.

### Lower Priority No-Rerun or External-Curation Items

6. **Head-to-head with ADMET-AI**: This does not require rerunning AutoQSAR, but it does require importing and normalizing ADMET-AI supplementary metrics.
   - Step 1: Download or manually curate ADMET-AI results for overlapping TDC ADMET benchmark datasets.
   - Step 2: Normalize dataset names, metrics, split assumptions, and metric directionality.
   - Step 3: Join against AutoQSAR best-per-dataset results using the same primary metric.
   - Step 4: Report win/loss/tie counts and per-dataset deltas.

7. **Applicability-domain calibration**: This is only no-rerun if per-molecule uncertainty or AD scores are already present in artifacts.
   - Step 1: Search benchmark outputs for uncertainty, AD, UMAP, distance-to-training, or prediction-interval columns.
   - Step 2: If available, join those fields to held-out prediction errors.
   - Step 3: Compute Spearman correlation between uncertainty/AD score and absolute error.
   - Step 4: Report high-error flagging rate and calibration plots.
   - Step 5: If no AD/uncertainty artifacts exist, defer this until the workflow emits those diagnostics.

### Items That Require New Benchmark Runs

8. **Multi-seed evaluation for TDC-22 official splits**: Official TDC-22 train/test splits are already used where `admet_group` is available, but the current workflow is not a full 5-seed benchmark.
   - Required work: rerun all 22 TDC ADMET Benchmark Group datasets with seeds 1-5, aggregate mean +/- standard deviation, and optionally apply paired tests against published baselines.
   - Implementation note: add a workflow-level seed loop or a dedicated TDC-22 multi-seed driver. The existing `maplight_parity_seeds` setting is not a substitute because it only applies to the MapLight parity path.

9. **Temporal robustness check**: Requires new scaffold-vs-temporal split comparisons.
   - Required work: identify datasets with temporal metadata or predefined temporal splits, rerun the affected model evaluations on temporal splits, and compare performance degradation against the current split protocol.

---

## Recommended Framing

### Title (Option A — Tool Paper)

*"AutoQSAR: An Accessible AutoML Framework for Molecular Property Prediction Achieving Competitive Performance Across 37 Benchmark Datasets"*

### Title (Option B — Empirical Finding)

*"Conventional Machine Learning with Ensemble Fusion Matches Foundation Models for ADMET Prediction: Evidence from 37 Benchmark Datasets"*

### Key Claims (in order of strength)

1. AutoQSAR achieves top-10 performance on 35/37 ADMET/molecular property benchmarks using a unified pipeline that requires no deep learning expertise or GPU hardware.
2. Conventional ML with CFA fusion and MapLight-style features matches or exceeds billion-parameter foundation models (MolGPS, MolE) on the majority of benchmarks.
3. No single model architecture dominates across all ADMET tasks — optimal architecture is highly dataset-dependent.
4. The tool is openly available, code-free, and runnable in Google Colab, lowering the barrier for ADMET modeling.

---

## Target Journals

### Tier 1 — Best Fit

| Journal | IF | Why | How to Frame |
|---------|-----|-----|-------------|
| **Journal of Cheminformatics** | ~8 | Published ADMETboost, FATE-Tox, CaliciBoost, Kamuntavicius et al. Software/benchmark focus. Open access. | Tool paper + empirical benchmark. Most natural home. |
| **Digital Discovery (RSC)** | ~7 | Actively seeking ML for chemistry. Open access. Newer audience. | Tool paper emphasizing accessibility and democratization. |

### Tier 2 — Strong Alternatives

| Journal | IF | Why | How to Frame |
|---------|-----|-----|-------------|
| **JCIM** | ~6 | ACS flagship for QSAR/cheminformatics | Needs stronger methods novelty; emphasize CFA + MapLight integration |
| **Bioinformatics** | ~5 | Published ADMET-AI. Application Note format (shorter). | Application Note: accessible tool with benchmark validation |
| **Briefings in Bioinformatics** | ~9 | If framed as benchmark + review + tool. Higher impact. | Empirical study with tool delivery |

### Tier 3 — Backup

| Journal | IF | Why | How to Frame |
|---------|-----|-----|-------------|
| **Scientific Reports** | ~4 | Broad scope. Accepts practical tools. | Straightforward tool + benchmark paper |
| **Molecular Informatics** | ~3 | Wiley. QSAR-focused audience. | Traditional QSAR methods audience |

### Primary Recommendation

**Journal of Cheminformatics** with Framing Option B (empirical finding + tool delivery). The "conventional ML matches foundation models" narrative is timely, provocative, and directly engages the community's most active debate.

---

## Competitive Landscape Summary

Papers published 2022–2026 that benchmark on overlapping datasets:

| Paper | Year | Journal | Datasets | Top Finding |
|-------|------|---------|----------|-------------|
| ADMETboost (XGBoost) | 2022 | J Mol Model | TDC-22 | XGBoost #1 on 18/22 |
| MapLight + GNN | 2023 | arXiv | TDC-22 | CatBoost + fingerprint combo #1 on 11/22 |
| ADMET-AI | 2024 | Bioinformatics | 41 TDC | Best avg rank, Chemprop-RDKit GNN |
| MolE | 2024 | Nat Commun | TDC-22 | 100M-param transformer, #1 on 10/22 |
| MolGPS | 2024 | NeurIPS | 38 (TDC + Polaris + MolNet) | 3B-param GNN, #1 on 11/22 TDC |
| MiniMol | 2024 | ICML Workshop | TDC-22 | 10M params, beats MolE on 17/22 |
| CaliciBoost | 2025 | J Cheminform | Caco-2 only | AutoML, #1 on Caco-2 |
| MaxQsaring | 2025 | J Pharm Anal | TDC-22 | Auto feature combo, #1 on 19/22 |
| Kamuntavicius et al. | 2025 | J Cheminform | TDC + external | Feature selection study |
| FATE-Tox | 2025 | J Cheminform | TDC Tox (4 datasets) | 3D equivariant, #1 on skin_reaction |
| Koleiev et al. | 2026 | bioRxiv | TDC-22 | Reproducibility: only 3/top pass |
| PrismNet | 2026 | Adv Sci | MoleculeNet | Best on 8/11 MoleculeNet tasks |
| **AutoQSAR (this work)** | **2026** | **—** | **37 (TDC + MolNet + Polaris + PODUAM + ChemML)** | **Broadest coverage; conventional ML + ensemble matches/beats SOTA on 35/37** |

The key differentiator is breadth + accessibility. No other paper covers all 5 benchmark suites with a single accessible tool.
