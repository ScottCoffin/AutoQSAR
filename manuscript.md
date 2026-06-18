# Title
No Single Architecture Wins: Conventional Machine Learning and Ensemble Fusion Match Foundation Models Across 44 Molecular Property Benchmarks


# Abstract
Accurate prediction of absorption, distribution, metabolism, excretion, and toxicity (ADMET) and related physicochemical properties is central to early drug discovery and chemical safety assessment, yet recent progress has been dominated by an arms race toward ever-larger foundation models whose computational cost, accessibility, and reproducibility increasingly limit practical adoption. We present AutoQSAR, a portable, open-source modeling and benchmarking workspace that predicts molecular properties from SMILES strings through a single, reproducible workflow spanning RDKit standardization, MapLight-style multi-family featurization, leak-free train-only feature selection, and a broad model library that includes conventional machine learning, gradient boosting, deep tabular networks, graph neural networks (Chemprop), a tabular foundation model (TabPFN), MapLight+GNN hybrids, and combinatorial-fusion and stacking ensembles. The tool is delivered through both a code-free, CPU-capable Jupyter/Colab notebook and a resume-safe command-line benchmark runner. We evaluated AutoQSAR across 44 datasets drawn from five benchmark suites—TDC ADMET, MoleculeNet, Polaris ADME, PODUAM, and ChemML—comprising 20 regression and 14 classification tasks under task-aware metrics, with the model achieving estimated top-10 placement on 35 of 37 leaderboard-comparable datasets and estimated first place on 7. 12Critically, no single architecture family dominated: conventional machine learning produced the most per-dataset wins (15), followed by ensemble meta-models (11), MapLight+GNN and TabPFN on regression tasks, and combinatorial fusion, with graph neural networks winning only a small subset of classification tasks. 3Although no family won universally, the MapLight+GNN hybrid was the most consistently near-best model, exhibiting the lowest mean gap to the per-dataset best (0.199). 4MapLight-derived features were the most strongly enriched and most frequently selected representation family, reinforcing the value of rich, combined molecular descriptors even with simple downstream learners.56 Together, these results demonstrate that a carefully engineered, GPU-optional AutoML pipeline matches or approaches specialized and foundation models on the majority of ADMET tasks, that optimal architecture is strongly dataset-dependent, and that competitive molecular property prediction can be made accessible and fully reproducible.

# Introduction

Unfavorable absorption, distribution, metabolism, excretion, and toxicity (ADMET) properties remain among the most consequential causes of failure in drug development. The preclinical stage confronts an attrition rate of approximately 93%, and even among candidates that reach clinical testing, more than 75% ultimately fail; undesirable ADME properties account for roughly 40% of candidate-molecule failures, and toxicity contributes up to a further 30% (Fu et al., 2024). The strategic value of identifying these liabilities early is well established: following the pharmaceutical industry's adoption of systematic early ADMET screening in the late 1990s, the share of clinical failures attributable to ADME and pharmacokinetic causes fell from roughly 40% to 11% (ADDME, 2009). Because experimental ADMET assays are time-consuming, costly, and difficult to scale to the ever-growing number of synthesized and virtual compounds, *in silico* prediction of ADMET endpoints from chemical structure has become an indispensable complement to laboratory screening, enabling medicinal chemists to prioritize compounds and deprioritize liabilities before synthesis (Fu et al., 2024; Komura et al., 2023).

The maturation of this field has been driven in large part by the emergence of standardized, publicly available benchmarks. MoleculeNet established curated datasets and evaluation protocols for molecular machine learning (Wu et al., 2018), and the Therapeutics Data Commons (TDC) subsequently consolidated a large collection of ADMET datasets into a benchmark group with fixed scaffold splits and per-task metrics, together with a public leaderboard that enables direct, side-by-side comparison of competing models (Huang et al., 2021). More recent resources such as the Polaris initiative have emphasized immutable, standardized benchmarks as a route to reproducible model comparison in drug discovery. These resources have catalyzed rapid methodological progress, but they have also fostered a leaderboard culture in which incremental gains are pursued through increasingly elaborate model architectures.

A prominent expression of this trend is the recent turn toward large pretrained "foundation" models for molecular property prediction. Graph-based foundation models such as MolE, pretrained on roughly 842 million molecules (Méndez-Lucio et al., 2024), and MolGPS, scaled to three billion parameters with the aid of phenomics data (Sypetkowski et al., 2024), have established state-of-the-art results on subsets of the TDC ADMET tasks, and parameter-efficient successors such as MiniMol have continued this line of work (Kläser et al., 2024). While impressive, these models require very large pretraining corpora, specialized hardware, and substantial engineering expertise, placing them out of practical reach for many of the academic, regulatory, and small-laboratory settings in which ADMET prediction is most needed.

Critically, the assumption that architectural scale translates into superior ADMET prediction is not well supported by the evidence. In the most comprehensive benchmark of its kind to date, Xia et al. (2023) evaluated twelve representative models—three non-deep and nine deep—and found that deep models are generally unable to outperform non-deep ones, with gradient-boosted trees and random forests built on molecular fingerprints tending to perform best, because tree models are well suited to the non-smooth target functions characteristic of molecular property prediction. A review in the *Annual Review of Biomedical Data Science* reached a concordant conclusion, reporting that a substantial and consistent advantage of deep learning over standard machine-learning approaches across diverse datasets and properties has not been demonstrated, and that success in compound-property prediction does not necessarily scale with model complexity (Rodriguez-Perez et al., 2022). These observations are borne out on the TDC ADMET leaderboard itself, where gradient-boosting methods built on combined fingerprint and descriptor representations remain highly competitive or dominant: extreme gradient boosting (ADMETboost; Tian et al., 2022), CatBoost paired with ECFP, Avalon, and ErG fingerprints plus 200 molecular properties, which achieved top-3 performance on 16 of 22 benchmarks (the MapLight submission; Notwell & Wood, 2023), AutoML over descriptor sets (CaliciBoost, 2025), and automatic feature-combination frameworks built on simple learners (MaxQsaring, which ranked first on 19 of 22 TDC tasks; Xu et al., 2025) all illustrate that careful feature engineering coupled with conventional models matches or exceeds far larger architectures. Systematic feature-representation studies reinforce this conclusion, showing that the choice of molecular representation is often more decisive than model architecture and that optimal choices are strongly dataset-dependent (Kamuntavičius et al., 2025).

Compounding the questionable returns of architectural complexity is a deepening concern over reproducibility. Across the quantitative sciences, data leakage has been identified as a pervasive and often invisible cause of overoptimistic results, with one survey finding it affected at least 294 papers across 17 disciplines (Kapoor & Narayanan, 2023). Cheminformatics is not exempt: leakage through structure duplication, preprocessing on combined train–test data, and feature selection performed before cross-validation systematically inflates QSAR performance estimates, and reproducibility, interpretability, and generalizability deficits have hindered the regulatory uptake of ML-based toxicity models (Belfield et al., 2023). A recent critical assessment of the TDC ADMET leaderboard found that only three of the top-ranked entries—CaliciBoost, MapLight, and MapLight+GNN—passed all reproducibility checks, with most leading submissions exhibiting unavailable code, non-reproducible execution environments, or methodological flaws (Koleiev et al., 2026). Such findings indicate that headline leaderboard rankings frequently fail to reflect either genuine methodological progress or deployable, trustworthy models.

These twin problems—the limited practical benefit of architectural scale and the fragility of reported results—are aggravated by a persistent accessibility gap. Many academic drug-discovery efforts founder in the preclinical "death valley," in part because researchers have limited access to commercial ADME prediction software owing to high licensing fees (Komura et al., 2023). Open tools have begun to address this gap. Web-based predictors such as ADMET-AI provide fast, accurate predictions with the highest average rank on the TDC leaderboard, but do not allow users to retrain or extend models on their own data (Swanson et al., 2024). Automated machine learning (AutoML) frameworks offer a more flexible alternative: DeepMol delivered competitive, fully reproducible pipelines across 22 TDC ADMET datasets while supporting both conventional and deep models (Correia et al., 2024); Auto-ADMET coupled grammar-based genetic programming with a Bayesian network to produce interpretable, personalized pipelines (de Sá et al., 2025); and accessible, code-light tools such as ChemXploreML have explicitly sought to lower the barrier to entry for non-specialists (Marimuthu & McGuire, 2025). Nonetheless, existing tools typically evaluate on a single benchmark suite, expose a limited subset of model families, and rarely combine a fully code-free interface with a rigorous, leakage-controlled batch-evaluation engine.

Here we present AutoQSAR, a portable QSAR modeling and benchmarking workspace designed to close these gaps. AutoQSAR couples a code-free, GPU-optional notebook for interactive model building with a resume-safe command-line benchmark runner, both calling a shared workflow core that enforces train-only feature filtering and selection to guard against leakage. The same pipeline spans the full methodological spectrum—conventional machine learning, gradient boosting, deep tabular and graph neural networks, a tabular foundation model, MapLight-style descriptor–graph hybrids, and combinatorial-fusion and stacking ensembles—and is evaluated uniformly across 44 datasets drawn from five benchmark collections (TDC ADMET, MoleculeNet, Polaris ADME, PODUAM, and ChemML), a breadth of cross-suite coverage that, to our knowledge, exceeds that of prior single-tool studies. Using this framework we show that no single model family dominates across tasks, that conventional machine learning and ensemble fusion match or exceed billion-parameter foundation models on the large majority of benchmarks, and that competitive, leaderboard-comparable ADMET prediction can be delivered through an accessible, fully reproducible, open-source tool that runs without specialized hardware. In doing so, AutoQSAR functions both as a practical instrument for ADMET modeling and as a large-scale empirical test of whether the field's drift toward architectural complexity is warranted.


# Methods

## 2.1 Overview and software architecture

AutoQSAR is a portable QSAR modeling and benchmarking workspace that predicts molecular properties directly from SMILES strings, and is distributed with two interoperable entry points that share a common workflow core: an interactive, widget-driven Jupyter/Colab notebook for code-free model building, and a command-line benchmark runner for systematic, resume-safe model evaluation across curated dataset collections. 
It is built around two entry points — colab_qsar_tutorial.ipynb, an interactive widget-driven notebook for building QSAR models on built-in or user-supplied datasets, and run_autoqsar_ga_benchmarks.py, a command-line benchmark runner for comparing model families across curated ChemML, TDCommons, MoleculeNet, Polaris, PODUAM, and literature datasets.
 Both entry points call a shared feature-generation, splitting, fusion, and evaluation library so that interactive and batch results are produced by identical code paths.

For each dataset, the runner executes a fixed sequence of stages: it 
builds molecular features, runs the train/test split plus train-only ElasticNetCV feature selection, evaluates conventional models, optionally runs a small GA tuning pass, runs deep workflows (ChemML backends, Chemprop v2 graph variants, and MapLight + GNN), optionally runs CFA combinatorial fusion over all successful predictions, builds an optional ensemble over available members, and writes cross-dataset performance tables.
 Optional model families are skipped gracefully when their dependencies, hardware, or dataset-size guardrails are not satisfied, allowing the remainder of a run to continue.

## 2.2 Datasets and curation

Benchmark datasets were drawn from five public collections through dedicated loaders: ChemML bundled examples (cep_homo, organic_density, xyz_polarizability), the Therapeutics Data Commons (TDC) single-prediction ADME and Tox tasks, MoleculeNet physicochemical datasets, Polaris ADME benchmark mirrors, and the PODUAM point-of-departure datasets. Each dataset is represented internally by a `DatasetSpec` that records the SMILES column, target column, recommended split, recommended metric, benchmark suite, and any leaderboard reference metadata.

All molecules were standardized prior to featurization. The canonicalization step coerced the target column to numeric, removed rows with missing or non-finite SMILES or target values, and parsed each SMILES with RDKit; molecules that failed to parse were dropped, and the remaining structures were re-encoded as canonical SMILES with `Chem.MolToSmiles(mol, canonical=True)`. 
For datasets carrying a predefined split column, rows missing the split assignment were also removed, and the canonical-SMILES frame was used for all downstream feature generation.


Target transformation followed a suite-aware policy. 
Under the default "auto" target-transform mode, datasets belonging to the TDC, MoleculeNet, Polaris, literature, and PFAS auxiliary-workbook suites were kept on their native (raw) target scale, while other datasets used a base-10 logarithmic transform when all target values were positive; non-positive targets disabled the log transform and reverted to the raw scale.


For TDC benchmark datasets, official splits were preferred over generic resampling. 
When PyTDC exposes an official admet_group entry for a TDC benchmark dataset, the runner uses that train_val/test split in preference to legacy single-prediction cache entries; official split frames are cached under data/_autoqsar_cache, and stale cache entries lacking an official split are refreshed automatically.
 Where catalog metadata supplied recommended splits or metrics for single-prediction tasks that PyTDC does not annotate, those values were applied so that 
benchmark runs used comparable split and metric choices instead of falling back to generic command-line defaults.


## 2.3 Molecular feature representations

Ten configurable feature families were implemented on top of RDKit, all returning fixed-width numeric matrices. Circular and path-based fingerprints were generated at 1024 bits by default. Morgan fingerprints used a radius of 2; 
ECFP6 and FCFP6 fingerprints were generated with the RDKit MorganGenerator at radius 3, with FCFP6 using feature-based atom invariants, and RDKit layered, atom-pair, topological-torsion, and RDKit path fingerprints were each computed at 1024 bits, with the path fingerprint spanning minimum and maximum path lengths of 1 and 7.
 MACCS keys were generated at their native 167-bit width, and the full RDKit 2D descriptor set was computed from `Descriptors._descList`.

The "MapLight classic" family reproduced the feature construction used by the MapLight TDC submission. 
It concatenates a hashed Morgan count fingerprint (radius 2, 1024 bits), an Avalon count fingerprint (1024 bits), the extended reduced graph (ErG) fingerprint, and a curated panel of RDKit molecular descriptors.
 
The descriptor panel comprises approximately 200 RDKit descriptors spanning connectivity and shape indices (e.g., BalabanJ, BertzCT, Chi and Kappa indices), VSA-family descriptors (EState_VSA, PEOE_VSA, SMR_VSA, SlogP_VSA), physicochemical properties (ExactMolWt, MolLogP, TPSA, FractionCSP3), hydrogen-bonding and ring counts, a large set of functional-group fragment counts, and the QED drug-likeness score.


After assembly, every feature matrix passed through a common finalization step that 
replaced infinities with missing values, coerced all columns to numeric, masked values exceeding the float32 range, imputed remaining gaps with the per-column median (falling back to zero), and cast the matrix to float32.


## 2.4 Persistent feature store

To avoid recomputing expensive descriptors across datasets and runs, features were cached in a persistent, content-addressed store. 
The store writes Parquet shards when pyarrow or fastparquet is available and falls back to CSV otherwise.
 Each representation is identified by a key derived from a SHA-256 hash of a canonical JSON payload listing the selected feature families, the Morgan radius, and the fingerprint bit width, ensuring that distinct feature configurations never collide. 
A schema file records the representation payload and column order, and a mismatch between an existing schema and a requested representation raises an error rather than silently mixing definitions.
 Feature rows are keyed by canonical SMILES, so only molecules absent from the cache are computed on a given run; 
cached and newly generated rows are then merged back into the exact row order of the requested SMILES list.
 Additional caches stored fully assembled benchmark feature matrices and tuned conventional models for cross-run reuse.

## 2.5 Train/test splitting and cross-validation

Four splitting strategies were supported: random, target-quartile, scaffold, and predefined, selected per dataset with a benchmark-aware default. 
The command-line defaults set the split strategy to target-quartile stratification, the test fraction to 0.2, and the random seed to 13.


Scaffold splitting used Bemis–Murcko scaffolds. 
Scaffold keys were computed after removing stereochemistry, with explicit fallbacks for molecules that failed scaffold perception or yielded no scaffold.
 Molecules were grouped by scaffold and assigned to the test set with a greedy, seeded procedure that packs whole scaffold groups until the target test fraction is reached, guaranteeing that no scaffold appears in both partitions. Target-quartile splitting binned the continuous target into quartiles with `pd.qcut` and stratified on those bins, reverting to random splitting when a dataset was too small or too low in target variance to form at least two populated bins per fold. For TDC benchmark datasets, the predefined official train/validation/test partition was used directly, and where SMILES needed reassignment, 
their split was reassigned deterministically by SMILES so that the same molecule was never intentionally placed in both train and test.


Cross-validation mirrored the chosen split geometry: scaffold runs used `GroupKFold` over scaffold groups, target-quartile runs used `StratifiedKFold` over quartile bins, and random runs used shuffled `KFold`, each with a configurable fold count and a defined fallback to random folds when group or bin constraints could not be satisfied.

## 2.6 Feature filtering and selection

Before model fitting, a leak-free pre-filter removed degenerate and redundant columns using statistics computed on the training partition only and applied identically to the held-out set. 
This step dropped near-constant columns below a variance threshold of 1e-8, removed binary columns whose positive prevalence fell outside the 0.005–0.995 range, and pruned exact-duplicate columns.


Feature selection was then performed train-only with a cross-validated elastic net. 
The ElasticNetCV selector was fit in an isolated subprocess with a wall-clock timeout, over a configurable L1-ratio grid and a log-spaced alpha grid, using cross-validation folds matched to the active split strategy.
 Features were retained when the absolute regression coefficient exceeded a threshold, with at least one feature always kept, and the retained set capped at a maximum size. 
By default this cap was the larger of one feature or ten percent of the training-row count.
 To bound runtime on large datasets, the selector used a dataset-size runtime model: 
when the predicted elastic-net runtime exceeded a configurable threshold (default 7200 seconds), or when the elastic net timed out or failed, the selector fell back to RandomForest feature-importance ranking.
 Selected features, coefficients, and selector diagnostics were written per dataset.

## 2.7 Model library

AutoQSAR evaluates a broad model library whose composition adapts to the task type. The task kind was inferred automatically: strict binary 0/1 targets were routed to the classification workflow with classification metrics, and all other numeric targets were treated as regression.

### Conventional machine learning

For regression, the conventional table comprised 
elastic-net regression with an internal cross-validated alpha/L1 search, support vector regression (C = 10, epsilon = 0.1, RBF kernel), random forests (400 trees), extremely randomized trees (500 trees), histogram gradient boosting (learning rate 0.05, up to 500 iterations, maximum depth 8), a soft-voting KNN+SVR regressor with an adaptively chosen neighbor count, AdaBoost (500 estimators, learning rate 0.05), and a tabular multilayer perceptron (hidden layers 512 and 256, ReLU, Adam, L2 1e-4, up to 300 iterations).
 Gradient-boosting libraries were added when installed: 
XGBoost (400 trees, depth 6, learning rate 0.05, subsample and column-sample 0.9), LightGBM (500 trees, learning rate 0.05, 63 leaves), and CatBoost (400 iterations, depth 6, learning rate 0.05).
 Numeric pipelines were preceded by median imputation and standardization where appropriate. The classification table substituted the analogous estimators (logistic regression, SVC with probability estimates, random forest, extremely randomized trees, histogram gradient boosting, soft-voting KNN+SVC, AdaBoost, tabular MLP, and gradient-boosting classifiers) and switched to classification loss functions and metrics.

### Specialized tabular and graph models

A compact one-dimensional convolutional regressor for tabular descriptors was included as an optional deep tabular baseline. 
This TabularCNNRegressor treats each standardized feature vector as a 1D signal and applies two same-padded ReLU convolutional blocks (64 filters, kernel size 5), global max pooling, a 128-unit dense layer with dropout, and a linear output, trained with Adam and early stopping; it is intentionally compact so it can run on CPU.


The MapLight + GNN workflow combined MapLight-style descriptors with pretrained graph embeddings, fitting a CatBoost model on the union of MapLight classic features and graph isomorphism network (GIN) fingerprints when the supporting DGL/PyTorch stack was available. A strict leaderboard-parity variant of the MapLight CatBoost model used mean-absolute-error optimization, target scaling, and five-seed averaging to reproduce the published MapLight evaluation protocol.

Graph neural networks were provided through Chemprop v2 with three configured architectures: 
a default directed message-passing network (D-MPNN); a CMPNN-style configuration using atom-level, undirected message passing; and an AttentiveFP-style proxy using atom messages, normalized aggregation, dropout 0.1, and an RDKit-2D descriptor featurizer.
 Optional variants augmented the graph encoder with train-only selected tabular descriptors or with the RDKit-2D featurizer. Uni-Mol V1 was supported as a 3D pretrained baseline; 
it ran automatically only when a GPU was detected, and could be forced on or off from the command line.
 The TabPFN tabular foundation model was available for both regression and classification, gated by a training-row guardrail (default 1000 rows) consistent with its design constraints.

### Genetic-algorithm tuning

An optional genetic-algorithm stage tuned a small set of estimators — elastic net and CatBoost for regression, and elastic-net-penalized logistic regression and CatBoost for classification. 
GA tuning was disabled by default unless explicitly requested, or unless an "auto" mode found prior evidence that a tuned family was worth rerunning.


## 2.8 Combinatorial fusion (CFA) and ensembling

After all base models produced aligned train and test prediction vectors, AutoQSAR optionally fused them with a combinatorial fusion analysis (CFA) procedure operating in both score and rank spaces. To control combinatorial growth, fusion inputs were first reduced to the best model per workflow, and the subset search space was bounded by a budget guardrail. For each candidate subset, the algorithm computed a performance strength as the inverse of the base model's training error and a diversity strength from the mean pairwise distance between normalized, sorted score profiles, and evaluated three score-space weightings (equal, performance-weighted, and diversity-weighted) plus the corresponding rank-space variants. 
Rank-space combinations were linearly calibrated back to the target scale, and were given a small metric discount when subset diversity exceeded a threshold, so that diverse rank fusions were preferred when genuinely complementary.
 
Candidates were ranked by the adjusted training metric and the best fused predictor was returned together with its selected models, weights, and a candidate-diagnostics table.
 For regression, the fusion objective minimized mean absolute error.

Three additional ensemble strategies were available over the same prediction pool: out-of-fold stacking with a RidgeCV meta-model (and a logistic meta-model for classification), an inverse-training-RMSE weighted average (weighting by the primary classification metric for classification tasks), and a simple average. Ensemble construction supported optional member filtering, including the removal of highly correlated members and the exclusion of members with negative held-out R².

## 2.9 Evaluation metrics

Regression performance was summarized by RMSE (the primary regression metric), mean absolute error, R², and Spearman rank correlation, computed separately on training and held-out sets. Classification performance was summarized by AUROC, AUPRC, balanced accuracy, and the Matthews correlation coefficient. 
Strict 0/1 binary targets were detected automatically and routed to the classification workflow, which reported AUROC, AUPRC, balanced accuracy, and MCC, with the primary metric selected per dataset.
 In the present benchmark, AUROC served as the primary classification metric for the majority of tasks while AUPRC was used for the imbalanced TDC CYP inhibition and substrate datasets, consistent with the TDC leaderboard's per-dataset metric assignments.

## 2.10 Leaderboard comparison

Where curated leaderboard references existed for a dataset, the best AutoQSAR model was compared against published top-1 and top-10 reference values using a normalized metric-matching procedure. Comparisons were only made when the leaderboard metric matched the dataset's primary metric and a numeric reference value was available; the framework then computed the signed gap to the top-1 reference, the gap to the top-10 cutoff, and an estimated rank relative to the published top-10. Reference rows were aggregated from TDC, MoleculeNet, and Polaris leaderboards together with a manually curated set of TDC ADMET reference values.

## 2.11 Reproducibility, caching, and multi-seed evaluation

The benchmark runner was designed for long, resumable runs. Completed datasets and compatible intermediate artifacts were reused on resume, and each run recorded a configuration signature, per-dataset run-status manifests, split-signature hashes, and per-stage runtime diagnostics. Two cost profiles were provided: a default cost-optimized profile that disables historically low-value expensive variants, and a full profile that restores the broader model set. Each dataset directory retained its metrics, predictions, selected features, feature-deduplication report, CFA candidate table, and ensemble weights, supporting independent verification of every reported result.

To provide statistically robust estimates on the core TDC benchmark, a dedicated multi-seed stage re-evaluated the best model per TDC-22 dataset across five random seeds (default seeds 1–5) using the official admet_group splits. 
When the main-run winner for a dataset was an ensemble or CFA fusion, the contributing base models were extracted from the recorded ensemble or CFA membership and evaluated across the seed list in place of the fusion row,
 and the stage wrote per-seed metrics together with mean and standard-deviation summaries.

## 2.12 Computing environment

The workflow targets Python 3.11 and runs on Windows, macOS, Linux, or Google Colab. 
The core workflow runs CPU-only; GPU availability primarily affects runtime and whether optional backends such as local TabPFN and Uni-Mol are practical.
 Pinned conda and pip/uv environment specifications were distributed with the repository to support reproducible installation.


# 3. Results
## 3.1 Benchmark coverage
We executed AutoQSAR across the full benchmark suite under a single, fixed configuration. Of the 45 datasets for which a run-status file was written, 44 completed successfully1, drawn from five public collections—the Therapeutics Data Commons (TDC) ADME and Tox single-prediction tasks, MoleculeNet physicochemical datasets, Polaris ADME mirrors, the PODUAM point-of-departure datasets, and the ChemML bundled examples. Benchmark datasets were drawn from five public collections through dedicated loaders: ChemML bundled examples (cep_homo, organic_density, xyz_polarizability), the Therapeutics Data Commons (TDC) single-prediction ADME and Tox tasks, MoleculeNet physicochemical datasets, Polaris ADME benchmark mirrors, and the PODUAM point-of-departure datasets.2

Metric-level analysis covered 34 datasets across 26 models, with 706 rows in the merged summary table; classification and regression tasks were separated by inferred task kind, yielding 14 classification and 20 regression datasets with successful metrics3. To position AutoQSAR against published baselines, we assembled a leaderboard-comparison layer: 148 leaderboard top-10 reference rows from the run artifact and 190 manually curated benchmark reference rows across 28 datasets were combined into 377 comparison rows spanning 37 datasets4. The dataset catalog—suite, task type, size, primary metric, and split—is summarized in Table 2.

⚠ Co-author note (coverage count). The notebook reports two different denominators that should be reconciled in the final text. The cross-dataset winner and architecture-coverage analyses (used in §3.2–3.3) span all 44 completed datasets (22 regression + 22 classification), whereas the successful-metrics summary reports 34 datasets (20 regression + 14 classification). The most likely explanation is a stricter metric-availability filter applied at one analysis stage; please confirm which denominator should be cited as headline coverage and state it consistently.

## 3.2 No single architecture dominates
The central finding of this work is that no single architecture family won across the benchmark. Aggregating per-dataset winners by architecture family (Figure 2), conventional machine-learning models produced the most outright wins (15: 9 classification + 6 regression), followed by ensemble meta-models (11: OOF stacking won 6 classification datasets and inverse-train-RMSE weighted averaging won 45 plus 1 regression), the MapLight + GNN graph-transfer hybrid (6 regression), the TabPFN tabular foundation model (5 regression), combinatorial fusion (CFA; 5: 4 regression + 1 classification), and graph neural networks (Chemprop v2 won 2 classification datasets6).

On the classification side, the winners were distributed across OOF stacking (6), inverse-RMSE weighted averaging (4), CatBoost (3), AdaBoost (2), Chemprop v2 (2), and one each for CFA fusion, HistGradientBoosting, LogisticRegression, Random forest, and the Tabular MLP7. On the regression side, MapLight + GNN and TabPFN were the most frequent winners, with CFA fusion and gradient-boosted trees (XGBoost/CatBoost) each taking a cluster of datasets (see §3.6).

This breadth of winners is the result we wish to foreground: the optimal model is strongly dataset-dependent, and the family that wins most often (conventional ML) is neither a foundation model nor a graph network.

## 3.3 Consistency versus peak performance
Win counts reward only the single best model per dataset and therefore understate the reliability of models that are consistently near-best without winning. To capture reliability, we computed each family's mean gap to the per-dataset best metric ("mean delta-from-best"; lower is better), shown in Table 3 and reported in the architecture-coverage analysis.

Architecture family	Datasets	Models	Rows	Mean gap-to-best	Mean RMSE	Mean balanced acc.
Graph transfer + tabular head (MapLight + GNN)	44	1	44	0.199	5.883	0.757
Ensemble meta-model	44	2	66	0.339	5.975	0.844
CFA combinatorial fusion	44	1	44	0.452	6.464	0.833
TabPFN foundation model	44	2	44	0.473	10.425	0.778
Deep tabular neural nets	44	1	44	0.690	6.896	0.789
Conventional ML	44	15	484	0.690	6.817	0.818
Graph neural networks (Chemprop)	44	2	84	0.875	7.048	0.822
The MapLight + GNN graph-transfer head had the lowest mean delta-from-dataset-best (0.199), followed by ensemble meta-models (0.339), CFA combinatorial fusion (0.452), the TabPFN foundation model (0.473), deep tabular neural nets (0.690) and conventional ML (0.690), with Chemprop graph neural networks showing the largest mean gap (0.875).8 The interpretation is that, although MapLight + GNN won only a minority of datasets outright (§3.2), it was the most reliable near-best single model—rarely the winner, but rarely far from it. This consistency-versus-peak distinction reconciles the headline finding (conventional ML wins most often) with the practical observation that a single rich-feature hybrid is the safest default when only one model can be run.

## 3.4 Leaderboard competitiveness
Across the leaderboard-comparison layer, 37 comparison rows spanning 37 datasets were leaderboard-comparable; AutoQSAR's best per-dataset model placed in the estimated top-10 on 35 of 37 datasets and at estimated first place on 79 (Figure 3; per-dataset detail in Table 4). Strong placements were concentrated in classification, where AutoQSAR reached AUROC 0.990 on tdc_hia_hou, 0.949 on tdc_clintox, 0.932 on tdc_bbb_martins, and 0.924 on tdc_pgp_broccatelli10, together with AUROC 0.952 on the tdc_cyp1a2_veith CYP-inhibition task11.

The 7 estimated first-place datasets were esol_delaney, tdc_hydrationfreeenergy_freesolv, lipophilicity, tdc_skin_reaction, tdc_bioavailability_ma, tdc_carcinogens_lagunin, and tdc_hia_hou. Two datasets fell below the estimated top-10 and are discussed honestly rather than omitted: polaris_adme_fang_solu_1 (XGBoost, estimated rank >10) and tdc_half_life_obach (TabPFNRegressor, Spearman 0.391, estimated rank >10)12. The Obach half-life task in particular remained difficult for every model in the library, consistent with its known label noise and narrow dynamic range; we return to it as a limitation in the Discussion.

Important interpretive caveat. The "estimated rank" reflects the references available for each dataset, and two of the seven first-place placements (esol_delaney, lipophilicity) are scored against a sparse, dated leaderboard. These should not be presented as current state-of-the-art (see §3.5).

## 3.5 Critical caveat: ESOL and Lipophilicity are not state-of-the-art
The estimated first-place placements for esol_delaney and lipophilicity require an explicit caveat, because they are artifacts of an outdated reference set rather than genuine state-of-the-art results.

For ESOL (Delaney), AutoQSAR's best model (TabPFNRegressor) reached a test RMSE of 0.621, scored against a MoleculeNet reference with a top-1 value of 0.8851 and a top-10 cutoff of 1.7406 drawn from only two reference rows13. Those two rows correspond directly to the public DeepChem MoleculeNet leaderboard, on which the only entries are a GCN at test RMSE 0.8851 (rank 1) and a Random Forest at 1.7406 (rank 2), both dated January 202014. Against this two-model, 2017-era leaderboard an RMSE of 0.621 is nominally "first," but the modern literature is substantially better: PrismNet reports state-of-the-art MoleculeNet performance with an ESOL RMSE of 0.558 ± 0.027 and a Lipophilicity RMSE of 0.549 ± 0.01715, and Uni-Mol reaches an ESOL RMSE of 0.568 and GEM reaches a Lipophilicity RMSE of 0.58716.

For Lipophilicity, AutoQSAR's best model (inverse-RMSE weighted-average ensemble) reached a test RMSE of 0.595 against a MoleculeNet top-1 of 0.7806 and top-10 cutoff of 0.9621, again from only two reference rows17. Placed against current published values (~0.549–0.587), AutoQSAR's 0.595 and ESOL's 0.621 would rank approximately fifth-to-sixth rather than first. We therefore report these two placements transparently as competitive but not state-of-the-art, and we do not claim SOTA on either dataset. This caveat does not affect the broader conclusion of §3.2–3.4, which rests on tasks (e.g., the TDC ADMET leaderboard) with denser, contemporary reference sets.

## 3.6 Task-specific winners
The per-dataset winners (Figure 2; Table 4) clarify where each family excelled.

Regression. MapLight + GNN and TabPFN won most regression datasets, but combinatorial fusion and gradient-boosted trees each claimed a distinct cluster. CFA fusion won four regression datasets: tdc_hydrationfreeenergy_freesolv (RMSE 0.645), poduam_pod_nc_std (RMSE 0.700), tdc_ld50_zhu, and tdc_solubility_aqsoldb. 18XGBoost won four (poduam_pod_rd_std RMSE 0.555, polaris_adme_fang_perm_1 RMSE 0.430, polaris_adme_fang_solu_1 RMSE 0.586, and tdc_lipophilicity_astrazeneca RMSE 0.644)19, while MapLight + GNN took the protein-binding and clearance tasks (polaris_adme_fang_hppb_1, polaris_adme_fang_rppb_1, tdc_clearance_hepatocyte_az, tdc_clearance_microsome_az, tdc_ppbr_az, and tdc_vdss_lombardo)20 and TabPFNRegressor won the small physicochemical/quantum datasets (chemml_cep_homo, chemml_organic_density), esol_delaney, freesolv_sampl, and tdc_half_life_obach21.

Classification. Ensemble meta-models dominated. OOF stacking with a RidgeCV meta-model won 6 datasets and inverse-train-RMSE weighted averaging won 4, with CatBoost winning 3, AdaBoost and Chemprop v2 winning 2 each, and CFA fusion, HistGradientBoosting, LogisticRegression, Random forest, and the Tabular MLP each winning 122.

⚠ Co-author note (multi-seed regression diagnostics). A separate regression "winner diagnostics" pass attributes some regression wins differently and even introduces a model—Uni-Mol V1 (credited with 5 regression wins)—that is not present in the main 26-model library and whose run directory is labeled all_benchmarks_no_unimol. Per our agreed convention, the main best-model-per-dataset / leaderboard analysis is treated as canonical here, and this alternate pass is mentioned only as a robustness check. These two passes should be reconciled (or the Uni-Mol pass explicitly scoped as a separate run) before submission.

## 3.7 Feature-family analysis
We quantified which representation families the leak-free elastic-net selector actually retained, relative to a uniform-selection baseline (Figure 4). The MapLight descriptor family contributed by far the largest share of selected features: MapLight features were selected in all 44 datasets and accounted for 30.8% of all selected features (4,849 of the selected total), with an enrichment of 1.343 versus the uniform baseline; fcfp6 was the next-largest contributor (12.1% of selected features, enrichment 1.080), while ecfp6 (enrichment 0.866) and atom_pair (enrichment 0.881) sat near or just below the uniform baseline. 23No feature family met the strict drop-candidate rule, meaning no family looked both rare across datasets and consistently unselected.24

This supports the thesis that rich, combined molecular descriptors carry most of the predictive signal even with simple downstream learners. However, the precise framing matters:

⚠ Co-author note (enrichment vs. share). The outline states "MapLight features most enriched (enrichment 1.343)." The notebook shows that MapLight has the largest absolute share of selected features (30.8%) and universal selection (44/44 datasets), but it is not the most enriched per feature. On a per-feature enrichment basis, the compact RDKit descriptor panel was most enriched (4.024 versus uniform), and MACCS keys were also positively enriched (1.262)25. Recommend rewording to: "MapLight contributed the largest share of selected features and was universally selected, while the compact RDKit panel showed the highest per-feature enrichment"—this is both accurate and arguably a stronger point (a small, interpretable descriptor set punches above its weight).

## 3.8 Ensemble value-add
Because ensembles and combinatorial fusion add computational cost, we asked whether they actually improved on the best available single base model (Table 5). The purpose-built value-add analysis showed that ensembles rarely beat the best base model, and that their benefit was concentrated in classification.

For regression (22 datasets), CFA fusion won outright on 4 datasets, inverse-RMSE weighted averaging on 1, and OOF stacking on 0; OOF stacking beat the best base model on only 2 of 22 datasets (median rank 5.5, median score improvement −0.029), and CFA beat the best base on 4 (median rank 7.0, median improvement −0.032)26—the remaining 17 regression datasets were won by the best single base model. A complementary head-to-head winner summary over 20 regression datasets reached the same conclusion: the best base model won 16 of 20 (80%), CFA won 2, OOF stacking 1, and the inverse-RMSE average 127.

For classification (22 datasets), the picture was more favorable to fusion: OOF stacking won 7 datasets and beat the best base on 9 of 22 (median rank 3.0), inverse-RMSE weighted averaging won 4, and CFA fusion won 128. In direct head-to-head comparison among the fusion methods, inverse-RMSE model averaging had the highest pairwise win fraction (0.688), followed by OOF stacking (0.667), with CFA fusion lowest (0.222)29.

⚠ Co-author note (two reconcilable points). (1) The value-add tally credits OOF stacking with 7 classification wins, whereas the canonical best-model-per-dataset tally (§3.2) credits it with 6. The difference arises because the value-add analysis compares each fusion method only against its own base-model pool; please add a one-line footnote clarifying this so the two counts do not appear contradictory. (2) The median score improvement of every ensemble method versus the best base was slightly negative in this run (e.g., −0.029 for regression OOF stacking), i.e., the typical ensemble did not beat the best base—gains were real but dataset-specific. We recommend stating the magnitude of gain only on the datasets where ensembles win, and reporting the typical rank when they do not, exactly as the data support.

## 3.9 Genetic-algorithm tuning (negative result)
Genetic-algorithm hyperparameter tuning was disabled by default in this run. The configuration recorded a GA resolution of mode "disabled" with reason "empty_ga_models,"30 and the analysis confirmed that no GA model rows were found in the scanned run directories31. Accordingly, no GA-tuned model contributed any per-dataset win or near-best result, and we report this as a deliberate negative result that informs the recommended default configuration: the broad fixed model library, not per-model GA search, drives AutoQSAR's competitiveness.

## 3.10 Cost-versus-value diagnostics
A core motivation for AutoQSAR is that competitive accuracy need not require GPU-scale compute. The full benchmark, spanning 34 datasets and 26 models, completed in 22.4 total elapsed hours, with a mean estimated rank of 4.07, a top-10 hit rate of 0.87, and a mean feature-selector time of 771 s per dataset32 (Figure 5; per-family runtime in Table 6).

Per-family wall-clock cost varied by more than two orders of magnitude. Median wall-clock time was 198 s for ensemble meta-models, 385 s for TabPFN, 975 s for conventional ML, 1,123 s for other neural baselines, 1,817 s for Chemprop, and 3,781 s for CFA fusion, but 46,515 s (~13 h) for the MapLight + GNN graph-transfer hybrid33. The single most expensive individual runs in the library were the lipophilicity weighted-average ensemble at 12,598 s and, far cheaper, the freesolv Chemprop run at 582 s34. Recorded model sizes underline the accessibility argument: Chemprop carried a median of 395,618 trainable parameters and other neural baselines 341,761, versus published comparators ADMET-AI (GPU-capable Chemprop-RDKit), MolE (~100M parameters, pretrained on ~842M molecules), and MolGPS (~3B parameters)35.

On a paired cost-performance basis, the most favorable single-model choice was an ensemble: OOF stacking gave the best paired cost-performance score (median score 0.375, median delta-from-best 0.0244, median runtime 120.5 s)36, and across the cost-performance summary XGBoost led on median cost-performance score (0.338), with a Friedman repeated-measures test confirming significant differences across models (statistic 143.4, p = 4.8 × 10⁻²²)37.

⚠ Co-author note (selector-scaling slope — correction required). The outline reports the feature-selection time scaling as "log-log slope ≈ 1.09 (near-linear)." The notebook reports a materially different value: the selector scaling fit (log₁₀ seconds vs log₁₀ n_molecules) had slope 2.103, intercept −4.320, and correlation 0.76338. A slope of ~2.1 implies roughly quadratic scaling of selector time with dataset size, not near-linear. The 1.09 figure appears to be an error and should be replaced by 2.103 (or the source of the 1.09 value identified). For context, the longest selector run was lipophilicity at 6,399 s (1.78 h) for 4,200 molecules, versus a few seconds for the smallest datasets39, consistent with super-linear scaling.

## 3.11 Reproducibility audit
Every reported result is backed by a complete, version-pinned artifact trail. The reproducibility audit covered 45 datasets, of which 44 had metrics, predictions, selected-feature records, and split hashes, giving a median required-artifact fraction of 1.0; the run recorded a single git commit (b06baf5e…), a present run configuration, and fixed seeds (random seed 13, Chemprop seed 42)40. The run was executed under a "cost_optimized" benchmark profile recorded in the config signature, with GA models resolved as disabled41, and a SHA-256 archive manifest was generated for all repository-level and per-dataset artifacts to support independent re-execution.

⚠ Co-author note (dirty working tree). The audit records git_is_dirty = True42, i.e., the working tree contained uncommitted changes when the benchmark ran. For a reproducibility claim in the manuscript, we recommend re-tagging the archived release from a clean commit (or explicitly documenting the diff) so reviewers can map the cited commit hash to the exact code state.

Figures and Tables referenced in this section
Figure 2. Best-model win counts by architecture family, split by task kind (headline figure; §3.2).
Figure 3. Estimated leaderboard rank distribution across 37 datasets (§3.4).
Figure 4. Feature-family enrichment versus the uniform-selection baseline (§3.7).
Figure 5. Cost-versus-value: model runtime versus gap-to-dataset-best (§3.10).
Table 2. Dataset catalog: suite, task, size, metric, split (§3.1).
Table 3. Architecture coverage with mean gap-to-best, mean RMSE, mean balanced accuracy (§3.3).
Table 4. Per-dataset best model versus leaderboard top-1 / top-10 cutoff and estimated rank; ESOL/Lipophilicity caveat flagged (§3.4–3.6).
Table 5. Ensemble value-add: base versus CFA versus OOF stacking versus weighted average (§3.8).
Table 6. Per-family median runtime and hardware (§3.10).

# References
*All DOIs and author lists below were verified during preparation, except where explicitly flagged as requiring final confirmation.*

1. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2), 513–530. https://doi.org/10.1039/C7SC02664A

2. Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., Coley, C. W., Xiao, C., Sun, J., & Zitnik, M. (2021). Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. *Proceedings of the NeurIPS Datasets and Benchmarks Track*. arXiv:2102.09548.

3. Fu, L., Shi, S., Yi, J., Wang, N., He, Y., Wu, Z., Peng, J., Deng, Y., Wang, W., Wu, C., Lyu, A., Zeng, X., Zhao, W., Hou, T., & Cao, D. (2024). ADMETlab 3.0: an updated comprehensive online ADMET prediction platform enhanced with broader coverage, improved performance, API functionality and decision support. *Nucleic Acids Research*, 52(W1), W422–W431. https://doi.org/10.1093/nar/gkae236

4. ADDME – Avoiding Drug Development Mistakes Early: central nervous system drug discovery perspective. (2009). *BMC Neurology*, 9(Suppl 1), S1. https://doi.org/10.1186/1471-2377-9-S1-S1 *[Author byline could not be verified from available metadata; confirm before submission.]*

5. Komura, H., Watanabe, R., & Mizuguchi, K. (2023). The Trends and Future Prospective of In Silico Models from the Viewpoint of ADME Evaluation in Drug Discovery. *Pharmaceutics*, 15(11), 2619. https://doi.org/10.3390/pharmaceutics15112619

6. Méndez-Lucio, O., Nicolaou, C. A., & Earnshaw, B. (2024). MolE: a foundation model for molecular graphs using disentangled attention. *Nature Communications*, 15, 9431. https://doi.org/10.1038/s41467-024-53751-y *[Confirm article number; DOI verified.]*

7. Sypetkowski, M., Wenkel, F., Poursafaei, F., Dickson, N., Suri, K., Fradkin, P., & Beaini, D. (2024). On the Scalability of GNNs for Molecular Graphs (MolGPS). *Advances in Neural Information Processing Systems (NeurIPS) 37*.

8. Kläser, K., Banaszewski, B., Maddrell-Mander, S., McLean, C., Müller, L., Parviz, A., Huang, S., & Fitzgibbon, A. (2024). MiniMol: A Parameter-Efficient Foundation Model for Molecular Learning. arXiv:2404.14986.

9. Xia, J., Zhang, L., Zhu, X., Liu, Y., Gao, Z., Hu, B., Tan, C., Zheng, J., Li, S., & Li, S. Z. (2023). Understanding the Limitations of Deep Models for Molecular Property Prediction: Insights and Solutions. *Advances in Neural Information Processing Systems (NeurIPS) 36*.

10. Rodriguez-Perez, R., Miljkovic, F., & Bajorath, J. (2022). Machine Learning in Chemoinformatics and Medicinal Chemistry. *Annual Review of Biomedical Data Science*, 5, 43–65. https://doi.org/10.1146/annurev-biodatasci-122120-124216

11. Tian, H., Ketkar, R., & Tao, P. (2022). ADMETboost: a web server for accurate ADMET prediction. *Journal of Molecular Modeling*, 28, 408. https://doi.org/10.1007/s00894-022-05373-8

12. Notwell, J. H., & Wood, M. W. (2023). ADMET property prediction through combinations of molecular fingerprints. arXiv:2310.00174.

13. CaliciBoost: Performance-driven evaluation of molecular representations for Caco-2 permeability prediction. (2025). *Journal of Cheminformatics*. https://doi.org/10.1186/s13321-025-01137-7 *[Author byline not verified; confirm before submission.]*

14. Xu, C., Xu, Y., Hu, Z., Zhao, X., Xie, W., Chen, W., & Pei, J. (2025). Unveiling optimal molecular features for hERG insights with automatic machine learning (MaxQsaring). *Journal of Pharmaceutical Analysis*, 15(12), 101411. https://doi.org/10.1016/j.jpha.2025.101411

15. Kamuntavičius, G., Paquet, T., Bastas, O., Šalkauskas, D., Prat, A., Abdel Aty, H., Pabrinkis, A., Norvaišas, P., & Tal, R. (2025). Benchmarking ML in ADMET predictions: the practical impact of feature representations in ligand-based models. *Journal of Cheminformatics*, 17, 108. https://doi.org/10.1186/s13321-025-01041-0

16. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804

17. Belfield, S. J., Cronin, M. T. D., Enoch, S. J., & Firman, J. W. (2023). Guidance for good practice in the application of machine learning in development of toxicological quantitative structure-activity relationships (QSARs). *PLOS ONE*, 18(5), e0282924. https://doi.org/10.1371/journal.pone.0282924

18. Koleiev, I., Stratiichuk, R., Shevchuk, N., Melnychenko, M., Nyporko, O., Todoryshyn, D., Husak, V., Starosyla, S., Yesylevskyy, S., & Nafiiev, A. (2026). Critical Assessment of ML models for ADMET Prediction in TDC leaderboards. *bioRxiv*. https://doi.org/10.64898/2026.02.26.708193

19. Swanson, K., Walther, P., Leitz, J., Mukherjee, S., Wu, J. C., Shivnaraine, R. V., & Zou, J. (2024). ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries. *Bioinformatics*, 40(7), btae416. https://doi.org/10.1093/bioinformatics/btae416

20. Correia, J., Capela, J., & Rocha, M. (2024). DeepMol: an automated machine and deep learning framework for computational chemistry. *Journal of Cheminformatics*, 16(1), 136. https://doi.org/10.1186/s13321-024-00937-7

21. de Sá, A. G. C., et al. (2025). Auto-ADMET: An Effective and Interpretable AutoML Method for Chemical ADMET Property Prediction. arXiv:2502.16378. *[Full author list not verified; confirm before submission.]*

22. Marimuthu, A. P. R., & McGuire, B. A. (2025). ChemXploreML: A Machine Learning Pipeline for Molecular Property Prediction. arXiv:2505.08688. *[Confirm final journal/venue details.]*