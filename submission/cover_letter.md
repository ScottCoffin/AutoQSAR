# Cover letter — Journal of Cheminformatics

> Paste into the submission system's cover-letter field, or use `cover_letter.pdf` (built from this
> file with pandoc; see `submission/README.md`). Items in **[AUTHOR]** need your input before sending.

---

Scott Coffin (ORCID 0000-0002-7035-1282)\
California Office of Environmental Health Hazard Assessment\
1001 I Street, Sacramento, CA 95814, USA\
scott.l.coffin@gmail.com

**[AUTHOR: date]**

To the Editors\
*Journal of Cheminformatics*

Dear Editors,

Please consider the enclosed manuscript, **"No Single Model Family Dominates: Ensembles and Conventional Machine Learning Perform Comparably to Pretrained Molecular Models Across 44 Property-Prediction Benchmarks,"** for publication as a **Research article** in the *Journal of Cheminformatics*. **[AUTHOR: update the title here if you change it; and, optionally, name the collection "Evaluating AI and machine learning models in cheminformatics: benchmarking techniques and case studies".]**

**What the paper reports.** We present QSARena, an open-source QSAR/AutoML workspace that predicts molecular properties from SMILES through one leakage-controlled workflow, delivered both as a code-free notebook that runs in Google Colab without any installation and as a command-line runner sharing the same core. We use it to run what is, to our knowledge, the broadest single-tool cross-suite benchmark published to date: 44 datasets from five collections (Therapeutics Data Commons, MoleculeNet, Polaris ADME, PODUAM and ChemML), 30 models and 1047 valid model–dataset evaluations, all under one fixed configuration. Base models were trained on an NSF ACCESS Jetstream2 A100 node, and every ensemble was built from out-of-fold member predictions only. The model library spans conventional machine learning and gradient boosting, a tabular foundation model (TabPFN), deep tabular and message-passing graph networks (Chemprop v2), 3D pretrained models (Uni-Mol V1 and V2), MapLight-style descriptor–graph hybrids, and combinatorial-fusion and stacking ensembles.

**Why it fits this journal.** Four aspects of the paper speak directly to debates the *Journal of Cheminformatics* has hosted and shaped:

1. **A uniform cross-suite benchmark, and an architecture-agnostic finding.** The paper evaluates 30 models across seven architecture families and five benchmark suites under one leakage-controlled pipeline and one fixed configuration. No single model family won more than 8 of the 44 datasets; ensembles built from those families won 13. This widens the evidence base for something your journal has published repeatedly: representation and model-family choice matter more to ADMET performance than architectural scale. A dataset-property meta-analysis then asks which dataset characteristics predict which family wins, and reports honestly that a regret-based family selector does not beat a single best default. The Introduction distinguishes this work from DeepChem, QSPRpred, OCHEM, ChemSAR and ADMET-AI, each of which covers part of the same ground; none reports a uniform evaluation of this kind.

   We also state plainly where we do **not** lead. On accuracy QSARena is competitive but not first: MaxQsaring reports first place on 19 of 22 TDC tasks (7 when every published value is re-scored on one consistent ranking), ADMET-AI holds a high average TDC rank, and Schrödinger's DeepAutoQSAR is described as a top performer in its own white paper. A landscape table positions the tool against open, commercial and web-based comparators across licence, cost, code-free access, retrainability, architecture breadth, suites benchmarked and published TDC performance, conceding the axes on which competitors lead.

2. **We decompose apparent leaderboard standing.** Choosing the best of 30 models on held-out scores, the protocol implicit in most leaderboard submissions, places the best model in the estimated published top ten on 35 of 37 comparable datasets, with five estimated first places. Choosing it by cross-validation alone drops that to 27 of 37 and one first place. A matched-candidate-set control shows that 7 of those 8 placements are the value of a broad model library and 1 the cost of honest selection. Both protocols come from one run, so this decomposition is far less sensitive to the quality of published reference values than the absolute ranks, which we present as provisional because the reference set contains entries with documented leakage.

3. **No tuning, no installation and no specialized hardware.** Every result was produced under a *single fixed configuration*, applied unchanged across all 44 datasets, five suites, two task types and a 48-fold range of dataset size, with per-model evolutionary search disabled. There was no per-dataset feature engineering, architecture choice or hyperparameter tuning, so what a practitioner gets from one run is what we report. The notebook entry point opens in Google Colab, installs its own dependencies in the hosted runtime and is driven entirely through form widgets, so a user with a CSV of SMILES and measured values writes no code and needs no local hardware. The same core installs with `pip install qsarena` for local, batch and HPC use. Rerunning the benchmark on a consumer laptop GPU moved the best score by a median of +0.3% across the 37 datasets with byte-identical splits, so the accuracy does not depend on datacentre hardware. We present the code-free path as a usability feature for the regulatory and small-laboratory settings your readership includes; it is not the scientific claim.

4. **Reproducibility is built in, not asserted.** Every figure, table and quoted number regenerates from the deposited artifacts with one command (`render_manuscript_assets.py`), and a companion script (`verify_manuscript_numbers.py`) asserts that the manuscript text still matches those artifacts: 96 automated checks that fail loudly if the text and the data drift apart.

**We have been deliberately conservative about what we claim.** The benchmark is single-split and single-seed; replicating it across five seeds was beyond our compute budget. We state this as the principal limitation and restrict our conclusions to findings that rest on categorical rather than close differences. We also report backend failures rather than analysing only successes, flag the comparisons that are not leaderboard-equivalent because they use locally generated splits, build every ensemble from out-of-fold predictions only, and quantify a remaining optimism in model cross-validation scores (feature selection is fitted on the full training split) rather than leaving it unstated. We would rather present a limited result honestly than a stronger one we cannot support.

**On naming and prior art.** An earlier draft of this work used the name *AutoQSAR*, which collides with an established Schrödinger product of the same name in the same application area. We have renamed the tool **QSARena**, added an explicit disambiguation in the Introduction, and compare QSARena with Schrödinger's AutoQSAR/DeepAutoQSAR and the earlier QSAR Workbench across benchmark breadth, reported performance, licensing, cost and reproducibility. Schrödinger's DeepAutoQSAR benchmark reports qualitative criteria without per-dataset metrics or released predictions, so we state plainly that a direct accuracy comparison is not possible.

**Declarations.** This manuscript is original, is not under consideration elsewhere, and has not been published previously. The author declares no competing interests. Use of large language model assistance is documented in the Methods (Section 2.14), as the journal requires. The work was supported by the California Office of Environmental Health Hazard Assessment. Computation used NSF ACCESS Jetstream2 under allocation CIS261142. All software and benchmark artifacts are openly available under the MIT License at https://github.com/ScottCoffin/QSARena, with a public issue tracker, and a versioned Zenodo archive will be deposited before publication. **[AUTHOR: if you post a preprint, add: "A preprint of this manuscript has been deposited at arXiv (DOI/ID: …) under a CC BY license."]**

**Suggested reviewers. [AUTHOR: optional — the journal invites 3–5 suggestions. Candidates whose work this paper engages directly include authors of the TDC reproducibility assessment, the MapLight submission, DeepMol, Auto-ADMET and the feature-representation benchmarking literature. Please confirm none has a conflict of interest with you.]**

Thank you for considering this work.

Sincerely,

Scott Coffin\
California Office of Environmental Health Hazard Assessment
