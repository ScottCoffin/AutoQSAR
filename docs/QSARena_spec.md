QSARena Dataset-Property Meta-Analysis — Implementation Spec for IDE Agent
Author of spec: prepared for Scott Coffin (QSARena)Target repo: github.com/ScottCoffin/QSARenaDeliverable: a new analysis module + manuscript figures/tables + manuscript text edits that answer "which dataset properties predict which model family wins?" — built entirely on the existing single-seed benchmark run, reusing the existing pipeline.

0. Read this first — intent, constraints, non-goals
Intent. The paper's headline is that no single model family dominates across 44 benchmarks. This work turns that null-sounding result into a conditional one: identify the dataset meta-features (size, chemical diversity, train/test similarity, class imbalance) that predict which family is best, and add a small interpretable "family recommender." This is well-precedented — per-dataset algorithm selection from dataset meta-features beat the single best method by up to 13% in the Meta-QSAR study (Olier, Sadawi, Bickerton, Vanschoren, Groşan, Soldatova, King, Machine Learning 2017, which compared 18 regression methods across >2,700 QSAR problems and found meta-learning-based algorithm selection outperformed the single best method — random forests on fingerprints — by up to 13% on average) 1 — and it directly answers the observation that ADMET model choice is highly dataset-dependent, with no clear single best model or feature choice 2.
Hard constraints (do not violate):
1.	No multi-seed, no retraining in the core. Phases 1–6 must run using only the already-deposited single-seed run artifacts. The single-seed run consumed 112 hours of A100 time, and a five-seed replication (~560 hours) is out of budget 3; do not attempt it. The only phase permitted to train models is the optional Phase 7, which is off by default.
2.	Reuse the existing pipeline. Do not reimplement SMILES standardization, featurization, scaffold logic, or split loading. Import them from the workflow core. The repository already contains the workflow core (qsar_workflow_core.py), the per-dataset benchmark artifacts for the A100 run (benchmark_results/autoqsar_benchmark_20260623_153839/: metrics, selected features, selector coefficients, split-signature hashes, runtime records, fusion candidate tables and ensemble weights), the analysis notebook, and the scripts that regenerate every figure, table and quoted number (render_manuscript_assets.py, verify_manuscript_numbers.py) 3. Locate exact paths before coding; the names above are authoritative but confirm.
3.	Use the same train/test partitions the benchmark used. Reconstruct splits from the deposited split artifacts / split-signature hashes, not by re-splitting, so meta-features align exactly with the reported results.
4.	Model the stable target. Because results are single-seed, the primary modeling outcome is the continuous relative gap of each family's best model to the per-dataset best; the secondary is the robust binary "within-5%-of-best." Never model the raw per-dataset "winner" as a hard label — the paper itself warns that the per-dataset winner is the winner on this split, an unknown fraction of the 44 winners would change under a different seed, and win counts should be read as a coarse distribution rather than a precise ranking 3.
5.	Quantify uncertainty by bootstrapping over the 44 datasets, not over seeds. This is the sanctioned way to get confidence bands here.
6.	Handle Chemprop specially. The Chemprop v2 family produced valid results on only 6 of 44 datasets 3; treat its cells as missing, exclude it from family-level trend fits, and annotate it — mirroring the paper's stance that it is the one row that should not be compared with the others 3.
Non-goals. No new model families; no changes to the benchmark runner; no re-derivation of the existing Figs 1–6 (only new figures). Keep new third-party dependencies to zero if possible (RDKit, scikit-learn, scipy, pandas, numpy, matplotlib are already present); statsmodels is allowed only if LOESS is needed and not already vendored.
Definition of done (global).
•	pytest green for all new tests.
•	python portable_colab_qsar_bundle/render_manuscript_assets.py regenerates all existing assets plus the new figures/tables with no errors.
•	python portable_colab_qsar_bundle/verify_manuscript_numbers.py passes, including the new computed macros — this is the repo's existing guard, where every table, figure and number is regenerated from the deposited benchmark run with one command and verify_manuscript_numbers.py fails on any drift 4.
•	Repo linter clean (the journal expects code to conform to an external linter such as black for Python 5).

1. Repository orientation task (Phase 0 — do before writing code)
Produce docs/meta_analysis/INVENTORY.md capturing, with exact paths:
•	The per-dataset metrics artifact schema (columns: dataset, model, family, task, primary_metric, metric_value, is_valid, …) as actually stored under benchmark_results/autoqsar_benchmark_20260623_153839/.
•	Where the per-family-best gap to per-dataset-best is computed today (this underlies Fig 6). Reuse that exact function/source so the new analysis is consistent with the published Fig 6 gap matrix.
•	How splits are persisted and reloaded (split-signature hashes → train/test indices or SMILES).
•	The dataset registry entry format (size, task, split.strategy, primary_metric) and the four split protocols in use: 27 predefined, 12 scaffold, 4 target-quartile, 1 random 3.
•	The feature-selection output per dataset (selected feature families; underlies Fig 4).
•	The public entry points the runner already exposes, including the applicability-domain tool qsarena-applicabilitydomain, and the manuscript-macro mechanism used by verify_manuscript_numbers.py.
Acceptance test (Phase 0): tests/meta/test_inventory.py asserts each documented path exists and each named artifact loads into a DataFrame with the documented columns. Fail loudly if a path is wrong so downstream phases don't silently invent data.

2. Module layout
Create a self-contained package:
qsarena/meta_analysis/
  __init__.py
  io.py            # load metrics, splits, registry, feature-selection artifacts (reuse core)
  meta_features.py # per-dataset descriptors (size, diversity, SNN, imbalance, label stats)
  gap_matrix.py    # per-family-best gap to per-dataset-best (reuse Fig 6 source)
  stats.py         # dataset-bootstrap CIs, permutation tests, LODO recommender
  figures.py       # manuscript-ready figures (style-matched to render_manuscript_assets)
  tables.py        # supplementary tables (meta-feature catalog, effect sizes)
  pipeline.py      # orchestrates end-to-end; single `run_meta_analysis(out_dir)` entry
docs/meta_analysis/
  INVENTORY.md
  METHODS.md       # human-readable methods text mirrored into the manuscript
tests/meta/
  test_*.py
pipeline.run_meta_analysis() must be callable from render_manuscript_assets.py and must write all figures/tables and a machine-readable meta_numbers.json consumed by verify_manuscript_numbers.py.

3. Phase 1 — Meta-feature catalog (meta_features.py)
Objective. For each of the 44 datasets, compute a row of meta-features from the deposited train/test partition, reusing core standardization/fingerprint utilities.
Features to compute (function compute_meta_features(dataset_id) -> dict):
Size & task
•	n_train, n_test, n_total, task (regression/classification), split_strategy, primary_metric.
Class imbalance (classification only)
•	pos_prevalence = positives / n_train; imbalance_ratio = max(p,1−p)/min(p,1−p). Rationale: on heavily imbalanced classification datasets, conventional methods such as kernel SVM outperform learnable featurizations 6, so imbalance is a candidate family-selecting feature.
Label distribution (regression only)
•	target_range, target_std, target_skew, target_kurtosis on the train targets (after any configured transform — match what the models saw).
Chemical diversity (both) — compute on train molecules, reuse core fingerprinting (ECFP4 / Morgan r=2, 2048 bits by default):
•	n_bemis_murcko_scaffolds, scaffolds_per_molecule = n_scaffolds/n_train, singleton_scaffold_frac.
•	internal_diversity = 1 − mean pairwise Tanimoto. For n_train > 2,000, estimate from a fixed-seed random sample of 20,000 pairs (deterministic). Note for METHODS.md: Tanimoto on binary fingerprints is the complement of the Jaccard distance the user asked about — they are the same metric.
Train→test similarity / applicability-domain shift (both) — the single most important difficulty axis, since prediction accuracy is highly correlated with the similarity of the predicted molecule to its nearest training-set neighbour 7:
•	For each test molecule, SNN = max Tanimoto to any train molecule. Report mean_snn, median_snn, and ood_fraction = share of test molecules with SNN < 0.40.
•	Reuse qsarena-applicabilitydomain for the neighbour computation if it already produces train→test distances; only fall back to a local ECFP4 nearest-neighbour if it does not.
Output: tables/meta_feature_catalog.csv (44 rows) → becomes Supplementary Table S-meta.
Acceptance tests (test_meta_features.py):
1.	Hand-computed toy case: a 6-molecule train / 3-molecule test fixture with known scaffolds, known positive count, and hand-computed nearest-neighbour Tanimoto → assert exact values (tolerance 1e-6 for similarity).
2.	Determinism: two calls with the sampling path (n>2,000) give identical internal_diversity (fixed seed).
3.	Ranges: pos_prevalence ∈ ⟨0,1⟩, mean_snn ∈ ⟨0,1⟩, ood_fraction ∈ ⟨0,1⟩, all diversity counts ≥ 0.
4.	Partition provenance: the loaded train/test indices match the split-signature hash on record for ≥3 spot-checked datasets (guards constraint #3).
5.	Coverage: catalog has exactly 44 rows and no NaNs in size/task/split columns.

4. Phase 2 — Gap matrix loader (gap_matrix.py)
Objective. Produce the per-dataset × per-family matrix of relative gap of the family's best model to the per-dataset best primary metric, identical in definition to Fig 6, plus a within5 boolean.
Functions:
•	load_family_gap_matrix() -> DataFrame (index: dataset, columns: family; values: gap %). Reuse the Fig 6 source computation — do not re-derive independently.
•	within5_matrix() derived at the 5% threshold.
•	Chemprop cells for its 38 non-valid datasets = NaN; add a chemprop_valid mask.
Acceptance tests (test_gap_matrix.py):
1.	Consistency with Fig 6: for ≥5 (dataset, family) cells, the loaded gap equals the value used by the existing Fig 6 renderer (tolerance 1e-6).
2.	Winner sanity: each dataset has ≥1 family at gap 0 (the winner), consistent with the star markers in Fig 6.
3.	Chemprop masking: exactly 6 datasets have non-NaN Chemprop gaps (matches the reported 6/44).
4.	Within-5% counts reproduce the paper's family within-5% figures where stated (e.g., the top three families clustered near ensembles 75%, 3D pretrained 70%, conventional ML 66% within 5% of best 3) within rounding.

5. Phase 3 — Statistics (stats.py)
All inference must be exploratory / hypothesis-generating and uncertainty must come from dataset-level bootstrap. Implement:
•	bootstrap_ci(func, data, n=10_000, seed=0) — resample the 44 datasets with replacement; return point estimate + 95% percentile CI. Used for every aggregate relationship.
•	spearman_with_ci(x, y) — rank correlation of a meta-feature against family gap, with dataset-bootstrap CI. (Rank-based because n is small and relationships may be monotone-nonlinear.)
•	permutation_test(x, y, n=10_000, seed=0) — label-shuffle null for the correlation; return two-sided p.
•	crossover_estimate(size, gap_family_A, gap_family_B) — fit each family's gap vs log10(n_train) (LOESS or robust linear), return the interpolated size where the fitted curves cross, with a bootstrap CI. Pre-register the target contrast: conventional ML/ensembles vs Uni-Mol (3D pretrained). Literature expects a size-driven crossover — deep methods overtake trees at roughly 500–2,000 training points for D-MPNN vs random forest 8 — and, more broadly, that dataset size is essential for representation-learning models to excel 9; report where (or whether) it occurs under QSARena's single fixed configuration.
•	lodo_recommender(meta_features, target) — the capstone recommender:
◦	Target = family with min gap per dataset grouped into ≤4 classes (ensemble, conventional ML, Uni-Mol, other), OR the multilabel "within-5% families." Use the grouped form to keep classes populated.
◦	Model = shallow decision tree (max_depth ≤ 3) and L1-penalized multinomial logistic; ≤4 predictors (log10(n_train), internal_diversity or mean_snn, imbalance_ratio/target_skew, task).
◦	Evaluation = leave-one-dataset-out accuracy/top-1, plus balanced accuracy, vs a "always predict the majority family" baseline and a permutation-label baseline.
◦	Return fitted rules (tree) / coefficients (logit) and per-feature importances.
Acceptance tests (test_stats.py):
1.	Bootstrap determinism & correctness: fixed seed reproducible; on synthetic y = 2x + noise, the CI for the slope excludes 0 and covers the true value at ~95% over repeated sims (smoke-level).
2.	Permutation monotonicity: stronger injected signal → smaller p; pure noise → p roughly uniform (p>0.05 in ≥90% of synthetic noise runs).
3.	Crossover recovery: on synthetic curves that cross at n=1,000, crossover_estimate returns ~1,000 within CI.
4.	Recommender floor: on a synthetic dataset where family is perfectly determined by size, LODO accuracy ≈ 1.0; on random labels, LODO ≈ majority baseline (guards against leakage/overfit reporting).
5.	Small-n guard: recommender refuses (raises/warns) if any class has < 4 datasets, forcing the grouped target.

6. Phase 4 — Figures (figures.py)
Match the existing manuscript figure style (fonts, sizes, palette, family colours) used by render_manuscript_assets.py; import its style helpers. Every figure writes both .pdf (vector, for submission) and .png, and every numeric annotation is also emitted to meta_numbers.json.
Figure M1 — Size and the family crossover (main text).Panel (a): per-family relative gap vs log10(n_train), points + fitted trend + dataset-bootstrap CI band, families coloured as in Figs 2/6, Chemprop excluded (annotated). Panel (b): the estimated conventional-vs-pretrained crossover size with CI. Caption states the single-seed caveat and that bands are dataset-bootstrap, not seed variance.
Figure M2 — Difficulty and family robustness to chemical-space shift (main text).Panel (a): achievable-best primary metric (or 1−gap) vs mean_snn / ood_fraction, showing how much apparent hardness is chemical-space shift — the effect that lets a few factors explain most performance variance, e.g. size, train/test similarity and assay noise together accounting for ~81% of performance variance 8. Panel (b): natural experiment — for the seven datasets re-split from random/target-quartile to scaffold between the two runs 3, plot each family's gap under low-shift vs high-shift splits on identical chemistry, testing whether pretrained/GNN families lose less ground as shift rises (the literature's expectation that D-MPNN models generalize across unseen chemical space better than tree-based models 8).
Figure M3 — The family recommender (main text or SI).Panel (a): the depth-≤3 decision tree (rendered) or logit coefficient plot. Panel (b): LODO accuracy vs majority baseline with bootstrap CIs. Frame explicitly as exploratory.
Figure S1 (SI) — Feature-family selection vs diversity.Extend Fig 4: per-dataset selected-feature composition vs internal_diversity (does Morgan/ECFP dominate on diverse sets, RDKit/physchem on homogeneous ones?).
Acceptance tests (test_figures.py): each figure function creates non-empty .pdf and .png; runs headless (matplotlib Agg); all annotated numbers are present as keys in meta_numbers.json; deterministic given fixed seeds (assert stable numeric annotations, not pixel hashes).

7. Phase 5 — Manuscript integration (text + citations)
7.1 New Results subsection. Add §3.13 "When does each family win? A dataset-property meta-analysis." Draft prose (place in docs/meta_analysis/METHODS.md and mirror into the manuscript source) with all quantitative claims injected as macros from meta_numbers.json so the drift guard stays green. Skeleton:
Under one fixed configuration, the relative gap of the best conventional-ML/ensemble model to the per-dataset best [narrows/does not narrow] with training-set size, with an estimated crossover against the 3D-pretrained family at ≈ {{crossover_n}} molecules (95% CI {{crossover_ci}}) — [consistent with / departing from] the ≈500–2,000 crossover reported for tuned D-MPNN-vs-RF comparisons. Chemical-space shift explained the largest share of cross-dataset performance variance (Spearman ρ = {{rho_snn}}), and family rankings [did/did not] reorder under high-shift scaffold splits. A leave-one-dataset-out recommender using {{k}} meta-features selected the best family group with balanced accuracy {{lodo_bacc}} versus {{baseline_bacc}} for the majority-class baseline. We stress these relationships are exploratory: they derive from one split and one seed per dataset, so we model the continuous gap rather than winner identity and bootstrap all intervals across the 44 datasets.
7.2 Abstract / Conclusions (optional, one sentence). Offer an edit adding the conditional finding to the existing message so the paper moves from "no family dominates" to "here is when each does." Keep it flagged for the author to accept.
7.3 Limitations. Add a sentence tying the meta-analysis to the existing single-seed caveat: winners are noisy, hence continuous-gap modeling and dataset-level bootstrap; n=44 limits meta-model complexity; diversity metrics are fingerprint-dependent (report the sensitivity check from §7.5).
7.4 Bibliography additions. Add these to the .bib. Provenance flags matter — do not fabricate author lists. Where I mark "AUTHORS VERIFIED," use exactly those names; where I mark "COMPLETE FROM DOI," resolve the full author list and page numbers from the DOI via CrossRef and leave a % TODO verify comment until confirmed.
•	Meta-QSAR — AUTHORS VERIFIED: Olier I, Sadawi N, Bickerton GR, Vanschoren J, Groşan C, Soldatova L, King RD. Meta-QSAR: a large-scale application of meta-learning to drug design and discovery. Machine Learning, 2017. DOI 10.1007/s10994-017-5685-x. (verified author list and venue) 1
•	Key elements — AUTHORS VERIFIED: Deng J, Yang Z, Wang H, Ojima I, Samaras D, Wang F. A systematic study of key elements underlying molecular property prediction. Nature Communications, 2023. DOI 10.1038/s41467-023-41948-6; PMID 37833262. (verified author list, venue, and PMID) 9
•	Sheridan 2004 — TITLE/VENUE/DOI VERIFIED, COMPLETE AUTHORS FROM DOI: Similarity to Molecules in the Training Set Is a Good Discriminator for Prediction Accuracy in QSAR. J. Chem. Inf. Comput. Sci., 2004. DOI 10.1021/ci049782w. (finding and venue verified) 10
•	Sheridan 2015 — TITLE/VENUE/DOI VERIFIED, COMPLETE AUTHORS FROM DOI: The Relative Importance of Domain Applicability Metrics for Estimating Prediction Errors in QSAR Varies with Training Set Diversity. J. Chem. Inf. Model., 2015. DOI 10.1021/acs.jcim.5b00110. (finding and venue verified) 11
•	Chemprop representation paper — TITLE/VENUE/DOI VERIFIED, COMPLETE AUTHORS FROM DOI: Analyzing Learned Molecular Representations for Property Prediction. J. Chem. Inf. Model., 2019. DOI 10.1021/acs.jcim.9b00237. (finding and venue verified) 12
•	MoleculeNet — TITLE/VENUE/DOI VERIFIED, COMPLETE AUTHORS FROM DOI: MoleculeNet: a benchmark for molecular machine learning. Chemical Science, 2018. DOI 10.1039/c7sc02664a. (finding and venue verified) 6
•	Deep-model limitations — TITLE/VENUE VERIFIED, COMPLETE FROM DOI: Understanding the Limitations of Deep Models for Molecular Property Prediction: Insights and Solutions. NeurIPS 2023. (finding and venue verified; resolve DOI/authors) 13
•	ADMET data-scaling paper — CONTENT VERIFIED, CITATION TO BE RESOLVED: the internal+public ADMET benchmark reporting the ≈500–2,000 D-MPNN-vs-RF crossover and the ~81%-variance model. Source seen as a ChemRxiv preprint ("Data Scaling and Generalization Insights…"/"Performance Insights for Small Molecule Drug Discovery Models…"). (quantitative claims verified) 8 Resolve final DOI/venue/authors before citing; do not state authors until resolved.
•	Reuse existing ref 17 already in the manuscript for the "dataset-dependent" point: Kamuntavičius et al., Journal of Cheminformatics 17:108, 2025, DOI 10.1186/s13321-025-01041-0 3.
Acceptance tests (test_manuscript_integration.py):
1.	Every {{macro}} in the new manuscript text resolves to a key in meta_numbers.json.
2.	verify_manuscript_numbers.py passes with the new macros, and fails when any meta_numbers.json value is perturbed (reuse the repo's existing drift-check pattern).
3.	Each new .bib key is cited at least once and each cited key exists in the .bib.
4.	A provenance check fails CI if any entry flagged "COMPLETE FROM DOI" still contains the literal placeholder author token (forces the agent to resolve them, and prevents shipping fabricated authors).

8. Phase 6 — Orchestration & reproducibility
•	pipeline.run_meta_analysis(out_dir) runs Phases 1–5 deterministically from deposited artifacts and writes: meta_feature_catalog.csv, all figures, meta_numbers.json, and the effect-size table.
•	Wire it into render_manuscript_assets.py so the standard one-command regeneration produces the new assets alongside the old.
•	Add a --meta-analysis flag or equivalent; ensure verify_manuscript_numbers.py consumes meta_numbers.json.
•	Update requirements-*.txt/conda env only if a new dependency proved unavoidable; prefer none. The journal requires that the software be entirely reproducible with source provided 14, so pin any addition.
Acceptance test (test_end_to_end.py): from a clean checkout of the deposited run, run_meta_analysis completes < 10 min on CPU, produces all expected files, and a second run yields byte-stable meta_numbers.json.

9. Phase 7 — OPTIONAL learning-curve experiment (off by default; requires retraining)
Only if compute allows — this is the causal version of Figure M1 and the only phase that trains models. Keep it isolated and clearly gated.
•	Pick 3 large datasets (e.g., LD50 ≈ 7,385; Tox21; Solubility/AqSolDB), subsample train to n ∈ {250, 500, 1k, 2k, 4k, full} via the existing --row-limit, and re-run only family representatives (Random Forest, XGBoost, Uni-Mol V1, ChemML MLP) at fixed config. This mirrors how MoleculeNet generated models across training-set volumes to show graph-based models reaching comparable accuracy only with sufficient samples 6.
•	Output: within-dataset scaling curves + interpolated crossover per dataset → Figure S2.
•	Budget note for the author: provide a printed A100/CPU estimate before running, using the runner's existing dry-run estimator, which scales the median fitting time of each model family measured in the paper's A100 run by dataset size 4. Do not launch without explicit approval.
Acceptance tests: subsampling is deterministic (fixed seed), curve CSVs have one row per (dataset, model, n), and the phase is skipped unless explicitly enabled (default-off assertion).

10. Statistical honesty checklist (enforce in code review & METHODS.md)
•	Primary outcome is continuous gap; winners are never modeled as ground truth (constraint #4).
•	All CIs are dataset-bootstrap; the text never implies seed-level variance was estimated.
•	Meta-model: ≤4 predictors, depth ≤3 / L1, LODO evaluation, permutation baseline, framed exploratory.
•	Multiple-comparison control (Benjamini–Hochberg) applied across the family × meta-feature correlation grid; report q-values.
•	Diversity sensitivity: recompute internal_diversity/SNN at a second fingerprint setting (e.g., Morgan r=3, 4096 bits) and report whether conclusions hold (§7.3).
•	Every headline sentence in §3.13 carries its CI or q-value; no bare point estimates.
References
1.	Olier I, Sadawi N, Bickerton GR, Vanschoren J, Grosan C, Soldatova L, King RD. Meta-QSAR: a large-scale application of meta-learning to drug design and discovery. Mach Learn. 2017;107(1):285-311. doi:10.1007/s10994-017-5685-x. PMID: 31997851.
2.	Kamuntavičius G, Paquet T, Bastas O, Šalkauskas D, Prat A, Aty HA, Pabrinkis A, Norvaišas P, Tal R. Benchmarking ML in ADMET predictions: the practical impact of feature representations in ligand-based models. J Cheminform. 2025;17(1):108. doi:10.1186/s13321-025-01041-0. PMID: 40691635.
3.	proof.pdf. Internal reference: file:26069#pages=29-30. Accessed 2026-09-28.
4.	additional_file_2_qsarena_tutorial.pdf. Internal reference: file:26070#pages=30-31. Accessed 2026-09-28.
5.	Hoyt CT, Zdrazil B, Guha R, Jeliazkova N, Martinez-Mayorga K, Nittinger E. Improving reproducibility and reusability in the Journal of Cheminformatics. J Cheminform. 2023;15(1):62. doi:10.1186/s13321-023-00730-y. PMID: 37391855.
6.	Wu Z, Ramsundar B, Feinberg EN, Gomes J, Geniesse C, Pappu AS, Leswing K, Pande V. MoleculeNet: a benchmark for molecular machine learning. Chem Sci. 2018;9(2):513-530. doi:10.1039/c7sc02664a. PMID: 29629118.
7.	Rethinking the generalization of drug target affinity prediction ... - arXiv. Retrieved 2026-09-28, from https://arxiv.org/html/2504.09481v1
8.	Data Scaling and Generalization Insights for. Retrieved 2026-09-28, from https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/67d5f28f6dde43c908f24f44/original/2503_ChemrXiv_JCIM.pdf
9.	Deng J, Yang Z, Wang H, Ojima I, Samaras D, Wang F. A systematic study of key elements underlying molecular property prediction. Nat Commun. 2023;14(1):6395. doi:10.1038/s41467-023-41948-6. PMID: 37833262.
10.	Sheridan RP, Feuston BP, Maiorov VN, Kearsley SK. Similarity to molecules in the training set is a good discriminator for prediction accuracy in QSAR. J Chem Inf Comput Sci. 2004;44(6):1912-1928. doi:10.1021/ci049782w. PMID: 15554660.
11.	Sheridan RP. The Relative Importance of Domain Applicability Metrics for Estimating Prediction Errors in QSAR Varies with Training Set Diversity. J Chem Inf Model. 2015;55(6):1098-1107. doi:10.1021/acs.jcim.5b00110. PMID: 25998559.
12.	Yang K, Swanson K, Jin W, Coley C, Eiden P, Gao H, Guzman-Perez A, Hopper T, Kelley B, Mathea M, Palmer A, Settels V, Jaakkola T, Jensen K, Barzilay R. Analyzing Learned Molecular Representations for Property Prediction. J Chem Inf Model. 2019;59(8):3370-3388. doi:10.1021/acs.jcim.9b00237. PMID: 31361484.
13.	Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions. Retrieved 2026-09-28, from https://proceedings.neurips.cc/paper_files/paper/2023/hash/cc83e97320000f4e08cb9e293b12cf7e-Abstract-Conference.html
14.	jcheminform-author-guidelines. Retrieved 2026-09-28, from https://jcheminform.github.io/jcheminform-author-guidelines/guidelines/software.html
