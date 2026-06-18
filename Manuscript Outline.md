# AutoQSAR Manuscript Outline

Working title options:

- **Option A (tool-forward):** "AutoQSAR: An Accessible, Reproducible AutoML Workspace for Molecular Property Prediction Benchmarked Across 44 ADMET and Physicochemical Datasets"
- **Option B (finding-forward):** "No Single Architecture Wins: Conventional Machine Learning and Ensemble Fusion Match Foundation Models Across 44 Molecular Property Benchmarks"
- **Recommended:** Option B as the title, with the tool name (AutoQSAR) in the subtitle. The empirical finding is the more citable hook; the tool is the delivery vehicle.

---

## Abstract (structured, ~250 words)

- **Background:** ADMET and physicochemical property prediction is dominated by an arms race toward larger foundation models (e.g., billion-parameter GNNs), yet reproducibility and accessibility lag behind.
- **Methods:** AutoQSAR, a portable workspace that runs a unified workflow — RDKit standardization, MapLight-style multi-family featurization, train-only feature selection, conventional ML, gradient boosting, deep tabular, graph (Chemprop), foundation (TabPFN), MapLight+GNN, and combinatorial/stacking ensembles — across 44 datasets from five benchmark suites (TDC ADMET, MoleculeNet, Polaris ADME, PODUAM, ChemML).
- **Results:** Across 44 completed datasets (20 regression, 14 classification with full metric analysis), AutoQSAR placed in the estimated top-10 on 35/37 leaderboard-comparable datasets and estimated #1 on 7. No single architecture dominated: ensembles won 11 datasets, conventional ML 15, MapLight+GNN 6, TabPFN 5, CFA fusion 5, and Chemprop 2. MapLight+GNN had the lowest mean gap-to-best (0.199). MapLight features were the most enriched selected family.
- **Conclusions:** A well-engineered, CPU-runnable AutoML pipeline matches or approaches specialized and foundation models on the majority of ADMET tasks, with optimal architecture being strongly dataset-dependent. The tool is open-source, code-free via a Colab notebook, and produces fully reproducible artifacts.


**Full Abstract**
Accurate prediction of absorption, distribution, metabolism, excretion, and toxicity (ADMET) and related physicochemical properties is central to early drug discovery and chemical safety assessment, yet recent progress has been dominated by an arms race toward ever-larger foundation models whose computational cost, accessibility, and reproducibility increasingly limit practical adoption. We present AutoQSAR, a portable, open-source modeling and benchmarking workspace that predicts molecular properties from SMILES strings through a single, reproducible workflow spanning RDKit standardization, MapLight-style multi-family featurization, leak-free train-only feature selection, and a broad model library that includes conventional machine learning, gradient boosting, deep tabular networks, graph neural networks (Chemprop), a tabular foundation model (TabPFN), MapLight+GNN hybrids, and combinatorial-fusion and stacking ensembles. The tool is delivered through both a code-free, CPU-capable Jupyter/Colab notebook and a resume-safe command-line benchmark runner. We evaluated AutoQSAR across 44 datasets drawn from five benchmark suites—TDC ADMET, MoleculeNet, Polaris ADME, PODUAM, and ChemML—comprising 20 regression and 14 classification tasks under task-aware metrics, with the model achieving estimated top-10 placement on 35 of 37 leaderboard-comparable datasets and estimated first place on 7. 12Critically, no single architecture family dominated: conventional machine learning produced the most per-dataset wins (15), followed by ensemble meta-models (11), MapLight+GNN and TabPFN on regression tasks, and combinatorial fusion, with graph neural networks winning only a small subset of classification tasks. 3Although no family won universally, the MapLight+GNN hybrid was the most consistently near-best model, exhibiting the lowest mean gap to the per-dataset best (0.199). 4MapLight-derived features were the most strongly enriched and most frequently selected representation family, reinforcing the value of rich, combined molecular descriptors even with simple downstream learners.56 Together, these results demonstrate that a carefully engineered, GPU-optional AutoML pipeline matches or approaches specialized and foundation models on the majority of ADMET tasks, that optimal architecture is strongly dataset-dependent, and that competitive molecular property prediction can be made accessible and fully reproducible.

---

## 1. Introduction

1.1. **Importance of ADMET/property prediction** in early drug discovery and chemical safety assessment (efficacy, toxicity, exposure).

1.2. **The benchmark landscape and its fragmentation.** TDC ADMET Benchmark Group (22 tasks), MoleculeNet, Polaris, PODUAM. Most published methods evaluate on a single suite, limiting cross-suite generalizability claims.

1.3. **The scaling narrative and the counter-evidence.** Foundation models (MolGPS 3B params; MolE 100M params, 842M pretraining molecules) push the SOTA frontier, but gradient-boosted trees with rich fingerprints (MapLight, ADMETboost, MaxQsaring, CaliciBoost) remain top-ranked on the TDC leaderboard. Recent critical assessments report that only a minority of top leaderboard models are fully reproducible.

1.4. **Gaps this work addresses:**
   - No accessible, code-free tool spans conventional → foundation models with consistent preprocessing.
   - Few studies systematically compare architecture families across multiple benchmark suites under one pipeline.
   - Reproducibility and compute-cost transparency are rarely reported.

1.5. **Contributions:**
   - (i) AutoQSAR, an open-source, resume-safe, CPU-capable QSAR/AutoML workspace with notebook and CLI entry points.
   - (ii) A unified 44-dataset, 26-model, 5-suite benchmark.
   - (iii) The empirical finding that no single architecture dominates and that conventional ML + ensembles are competitive with foundation models.
   - (iv) Feature-family, cost-vs-value, ensemble value-add, and reproducibility analyses.

---

## 2. Methods

2.1. **Workflow overview** (Figure 1: pipeline schematic). Load data → select SMILES/target → assess missingness → RDKit parse/standardize/canonicalize → optional duplicate collapse → featurization → split → train-only feature filtering/selection → model training → leaderboard comparison → artifact export.

2.2. **Molecular featurization.** RDKit descriptors plus MapLight-style fingerprint families: morgan, ecfp6, fcfp6, layered, atom_pair, topological_torsion, rdk_path, maccs, maplight classic. Persistent feature store keyed by canonical SMILES.

2.3. **Data splitting.** Random, scaffold, target-quartile, predefined. For TDC datasets, the official `admet_group` train_val/test split is preferred when available; molecules are deterministically reassigned by SMILES to prevent train/test leakage.

2.4. **Feature selection.** Train-only filtering and selection (report selector type, coefficients). Emphasize leak-free design.

2.5. **Model families** (Table 1: model inventory by family):
   - Conventional ML (ElasticNetCV, SVR/SVC, RF, Extra Trees, HistGradientBoosting, Voting, AdaBoost, Tabular MLP, XGBoost, LightGBM, CatBoost, LogisticRegression).
   - MapLight CatBoost (Strict Parity) and MapLight + GNN (CatBoost + pretrained GIN embeddings).
   - Deep/graph: Chemprop v2 (D-MPNN, CMPNN, AttentiveFP), ChemML MLP, Tabular CNN.
   - Foundation: TabPFN (regressor/classifier), Uni-Mol (noted as excluded in this run).
   - GA-tuned: ElasticNet, CatBoost (disabled by default; see Results).
   - Fusion/ensemble: CFA combinatorial fusion, OOF stacking (RidgeCV / logistic), inverse-RMSE weighted average, simple average.

2.6. **Evaluation protocol.** Regression: RMSE (primary), R². Classification: AUROC for 16 datasets, AUPRC for 6 (TDC-designated imbalanced CYP/substrate tasks), with balanced accuracy as secondary. Task kind auto-inferred (strict binary 0/1 → classification).

2.7. **Leaderboard comparison framework.** 377 combined reference rows across 37 datasets from TDC, MoleculeNet, Polaris, and a manually curated TDC ADMET reference set (190 rows / 28 datasets). Estimated rank, top-10 placement, gap-to-top1, gap-to-top10-cutoff.

2.8. **Benchmark execution.** Resume-safe runner, `cost_optimized` vs `full` profiles, caching layers, per-stage runtime logging. Hardware/environment specification (Python 3.11, conda/uv pinned packages).

2.9. **Reproducibility.** Config signatures, run-status manifests, deterministic split reassignment, artifact export (metrics, predictions, selected features, runtime).

---

## 3. Results

3.1. **Benchmark coverage** (Table 2). 44 datasets completed of 45 status files; 34 with full metric analysis (20 regression, 14 classification); 26 models; 706 merged metric rows across 5 suites.

3.2. **Headline: no single architecture dominates** (Figure 2: best-model win counts by family). Best-architecture-family wins: ensemble meta-model (10 classification + 1 regression = 11), conventional ML (9 + 6 = 15), MapLight+GNN/graph-transfer-head (6 regression), TabPFN (5 regression), CFA fusion (4 regression + 1 classification = 5), Chemprop GNN (2 classification). Frame as the central scientific finding.

3.3. **Consistency vs. peak performance** (Table 3: architecture coverage with mean gap-to-best). MapLight+GNN had the lowest mean delta-from-dataset-best (0.199), followed by ensemble meta-models (0.339), CFA (0.452), TabPFN (0.473), conventional ML and deep tabular (~0.690), and Chemprop (0.875). Interpretation: MapLight+GNN is the most *reliable* near-best single model, even when it does not win outright.

3.4. **Leaderboard competitiveness** (Figure 3: estimated rank distribution; Table 4: per-dataset best vs. top-1/top-10). 35/37 estimated top-10; 7 estimated #1. Highlight strong placements: tdc_hia_hou (0.990 AUROC), tdc_clintox (0.949), tdc_pgp_broccatelli (0.924), tdc_bbb_martins (0.932), tdc_cyp1a2_veith (0.952). The two datasets below top-10 (polaris_adme_fang_solu_1, tdc_half_life_obach) should be discussed honestly.

3.5. **Important caveat — ESOL and Lipophilicity** (dedicated subsection or clearly flagged in Table 4). The estimated #1 placements for esol_delaney (RMSE 0.621) and lipophilicity (RMSE 0.595) are scored against 2017 MoleculeNet baselines (GCN 0.885 and 0.781). Against current published results (e.g., PrismNet ESOL 0.558, Lipophilicity 0.549), these rank ~5th–6th. Report transparently; do not claim SOTA.

3.6. **Task-specific winners.**
   - Regression: MapLight+GNN and TabPFN win most often; CFA fusion wins 4 (e.g., tdc_hydrationfreeenergy_freesolv 0.645, tdc_solubility_aqsoldb, tdc_ld50_zhu, poduam_pod_nc_std); XGBoost wins 4 (Polaris/PODUAM).
   - Classification: OOF stacking ensembles win most (6), followed by inverse-RMSE weighted averaging (4) and conventional CatBoost/AdaBoost.

3.7. **Feature-family analysis** (Figure 4: enrichment vs. uniform baseline). MapLight features most enriched (enrichment 1.343, 30.8% of selected features) and most frequently selected; fcfp6 enriched (1.080); ecfp6/atom_pair near baseline. Supports the "rich combined representations matter" thesis even with simple downstream models.

3.8. **Ensemble value-add** (Table 5). Best base model won 17/22 regression datasets; CFA 4/22; inverse-RMSE averaging 1/22; OOF stacking 0/22 (regression). For classification, OOF stacking won 6 and weighted averaging 4. Quantify magnitude of gain where ensembles win and typical rank when they don't.

3.9. **GA tuning value** (brief). GA disabled by default; no GA-tuned rows added value in this run. Report as a negative result informing default configuration.

3.10. **Cost-vs-value diagnostics** (Figure 5: runtime vs. gap-to-best; Table 6: per-family median runtime). Contrast CPU-minutes for conventional/MapLight models against GPU-hours for graph/foundation models. Note extreme cases (e.g., lipophilicity ensemble at ~12,600 s; freesolv Chemprop at ~582 s). Selector-time scaling ~log-log slope 1.09 (near-linear).

3.11. **Reproducibility audit.** Run-status completeness (44/45), config signatures, deterministic splits, full artifact export.

---

## 4. Discussion

4.1. **Interpretation of the central finding.** Architecture choice is dataset-dependent; an AutoML approach with best-per-task selection is a practical and defensible strategy. Conventional ML + ensembles remain competitive with foundation models on the majority of ADMET tasks.

4.2. **When do foundation/graph models help?** TabPFN excels on small, clean regression datasets (ChemML examples, ESOL). MapLight+GNN is consistently near-best across both task types. Chemprop is competitive only on a subset of classification tasks. Provide practical guidance.

4.3. **Why MapLight features matter.** Connect to the broader evidence that richer combined fingerprint representations drive predictive power.

4.4. **Accessibility and democratization.** Competitive results without GPU or bespoke architecture lower the barrier for practitioners (e.g., regulatory toxicologists, small labs). Position against web-only tools (ADMET-AI) and compute-heavy foundation models.

4.5. **Reproducibility positioning.** Frame against published reproducibility concerns in the TDC leaderboard ecosystem; AutoQSAR's artifact-export and deterministic design are a deliberate response.

4.6. **Limitations.**
   - Single-run results for the main benchmark (multi-seed evaluation in progress/recommended); state clearly.
   - Some leaderboard references use heterogeneous split protocols (e.g., MoleculeNet vs. TDC scaffold); ESOL/lipophilicity baselines outdated.
   - Uni-Mol excluded from this run.
   - "Estimated rank" is an approximation against curated references, not a live leaderboard submission.

---

## 5. Conclusion

Restate: a unified, accessible AutoML workspace achieves top-10 performance on 35/37 leaderboard-comparable datasets across five suites; no architecture family dominates; conventional ML and ensemble fusion are competitive with foundation models at a fraction of the compute. Tool and artifacts are openly available.

---

## 6. Figures and Tables (proposed)

- **Figure 1.** AutoQSAR workflow schematic (data → features → split → models → fusion → comparison → artifacts).
- **Figure 2.** Best-model win counts by architecture family (split by task kind). *Headline figure.*
- **Figure 3.** Estimated leaderboard rank distribution across 37 datasets.
- **Figure 4.** Feature-family enrichment vs. uniform baseline.
- **Figure 5.** Cost-vs-value: model runtime vs. gap-to-dataset-best.
- **(Optional) Figure 6.** Dataset-by-model metric heatmap.
- **Table 1.** Model inventory by family and availability conditions.
- **Table 2.** Dataset catalog (suite, task, size, metric, split).
- **Table 3.** Architecture coverage with mean gap-to-best, mean RMSE, mean balanced accuracy.
- **Table 4.** Per-dataset best model vs. leaderboard top-1 / top-10 cutoff, estimated rank, gaps. *Flag ESOL/lipophilicity caveat.*
- **Table 5.** Ensemble value-add (base vs. CFA vs. stacking vs. weighted average).
- **Table 6.** Per-family median runtime and hardware.

---

## 7. Data and Code Availability

- GitHub repository (notebook builder + CLI runner + workflow core + benchmark registry).
- Zenodo archive of benchmark artifacts (metrics, predictions, selected features, run configs) with DOI.
- Environment specifications (conda YAML, pinned requirements), Git commit hash, random seeds.

---

## 8. Pre-Submission Checklist (mapped to prior recommendations)

1. **Add multi-seed (5-seed) results** for TDC-22 using official `admet_group` splits → strengthens Table 4 and enables significance testing.
2. **Correct ESOL/lipophilicity references** to current published SOTA → Section 3.5.
3. **Add computational-cost table** with hardware and parameter counts → Table 6 / Figure 5.
4. **Add ablation (component value) figure** isolating conventional ML → +MapLight → +DL → +fusion → strengthens Section 3.8.
5. **Publish Zenodo artifact + environment manifest** → Section 7.
6. **Optional head-to-head vs. ADMET-AI** on shared TDC-22 datasets → Supplementary.

---

## 9. Suggested Target Journals (in priority order)

1. **Journal of Cheminformatics** — best fit; precedent for ADMETboost, FATE-Tox, CaliciBoost, Kamuntavičius et al.; open access; software/benchmark focus.
2. **Digital Discovery (RSC)** — ML-for-chemistry focus; accessibility/democratization angle fits well.
3. **Journal of Chemical Information and Modeling** — emphasize MapLight+GNN and CFA fusion as methodological elements.
4. **Bioinformatics (Application Note)** — if a shorter, tool-centric framing is preferred; precedent for ADMET-AI.