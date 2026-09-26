# Cover letter — Journal of Cheminformatics

> Paste into the submission system's cover-letter field, or compile `cover_letter.tex` for a PDF
> on letterhead. Items in **[AUTHOR]** need your input before sending.

---

Scott Coffin
California Office of Environmental Health Hazard Assessment
1001 I Street, Sacramento, CA 95814, USA
scott.l.coffin@gmail.com

**[AUTHOR: date]**

To the Editors
*Journal of Cheminformatics*

Dear Editors,

Please consider the enclosed manuscript, **"No Single Model Family Dominates: Ensembles and Conventional Machine Learning Perform Comparably to Pretrained Molecular Models Across 44 Property-Prediction Benchmarks,"** for publication as a **Software article** in the *Journal of Cheminformatics*. **[AUTHOR: a focused review (2026-09-25) recommends submitting as a Research (benchmarking) article instead, possibly to the collection "Evaluating AI and machine learning models in cheminformatics: benchmarking techniques and case studies". Decide before sending, and change this sentence if you switch.]**

**What the paper reports.** We present QSARena, an open-source QSAR/AutoML workspace that predicts molecular properties from SMILES through one leakage-controlled workflow, delivered both as a code-free notebook that runs in Google Colab without any installation and as a command-line runner sharing the same core, and we use it to run what is, to our knowledge, the broadest single-tool cross-suite benchmark published to date: 44 datasets from five collections (Therapeutics Data Commons, MoleculeNet, Polaris ADME, PODUAM and ChemML), 28 models and 837 valid model–dataset evaluations, all under one fixed configuration, executed on an NSF ACCESS Jetstream2 A100 node. The model library spans conventional machine learning and gradient boosting, deep tabular and message-passing graph networks, 3D pretrained models (Uni-Mol V1 and V2), MapLight-style descriptor–graph hybrids, and combinatorial-fusion and stacking ensembles.

**Why it fits this journal.** Three of the paper's contributions speak directly to debates the *Journal of Cheminformatics* has hosted and shaped:

1. **A uniform cross-suite benchmark, and an architecture-agnostic finding.** The paper evaluates 28 models across seven architecture families and five benchmark suites under one leakage-controlled pipeline and one fixed configuration. No family won more than 36% of datasets, which widens the evidence base for something your journal has published repeatedly: representation and model-family choice matter more to ADMET performance than architectural scale. The Background distinguishes this from DeepChem, QSPRpred, OCHEM, ChemSAR and ADMET-AI, each of which covers part of the same ground; none reports a uniform evaluation of this kind.

   We also state plainly where we do **not** lead. On accuracy QSARena is mid-pack: ADMET-AI holds the highest average TDC rank, MaxQsaring reports first place on 19 of 22 tasks, and Schrödinger's DeepAutoQSAR is described as a top performer on 20 of 22. A landscape table (Table 6) positions the tool against twenty open, commercial and web-based comparators across licence, cost, code-free access, retrainability, architecture breadth, suites benchmarked and published TDC performance — conceding the axes on which competitors lead. We think this is more useful to your readers, and more credible, than a bare novelty assertion.

2. **We decompose apparent leaderboard standing.** Choosing the best of 28 models on held-out scores, the protocol implicit in most leaderboard submissions, places the best model in the estimated published top ten on 35 of 37 comparable datasets. Choosing it by cross-validation alone drops that to 25 of 37 and removes every first place. A matched-candidate-set control shows that 7 of those 10 placements are the value of a broad model library and 3 the cost of honest selection. Both protocols come from one run, so this decomposition is far less sensitive to the quality of published reference values than the absolute ranks, which we present as provisional because the reference set contains entries with documented leakage. We are not aware of another benchmark study that isolates this effect on its own results.

3. **No tuning, no installation and no specialized hardware.** Every result was produced under a *single fixed configuration*, applied unchanged across all 44 datasets, five suites, two task types and a 48-fold range of dataset size, with per-model evolutionary search disabled. There was no per-dataset feature engineering, architecture choice or hyperparameter tuning. What a practitioner gets from one run is therefore what we report — not the endpoint of an expert tuning campaign. That run need not happen on their own machine: the notebook entry point is a single self-contained file that opens in Google Colab, installs its own dependencies in the hosted runtime, and is driven entirely through form widgets, so a user with a CSV of SMILES and measured values writes no code and needs no local CPU, GPU or Python environment. For users who prefer a local or scripted workflow, the same workflow core installs with `pip install qsarena` and exposes a command-line entry point, and is equally suited to batch and HPC use. We also verified this empirically: rerunning the benchmark on a consumer laptop GPU moved the best score by a median of -0.2% across the 37 datasets with byte-identical splits, so the accuracy does not depend on datacentre hardware. We present the code-free path as a usability feature for the regulatory and small-laboratory settings your readership includes; it is not the scientific claim.

4. **Reproducibility is built in, not asserted.** Every figure, table and quoted number regenerates from the deposited artifacts with one command (`render_manuscript_assets.py`), and a companion script (`verify_manuscript_numbers.py`) asserts that the manuscript text still matches those artifacts — 78 automated checks that fail loudly if the text and the data drift apart. This directly answers the code-availability and archival criteria set out in your editorial on improving reproducibility and reusability.

**We have been deliberately conservative about what we claim.** The benchmark is single-split and single-seed: replicating it across five seeds was beyond our compute budget, since the single-seed run alone consumed 112 hours of A100 time. We state this as the principal limitation, spell out its four specific consequences for our tables, and restrict our conclusions to findings that rest on categorical rather than close differences. We also report backend failures rather than analysing only successes, flag that two comparisons are not leaderboard-equivalent because they use locally generated splits, and disclose that our ensemble member filtering consults held-out data. We would rather present a limited result honestly than a stronger one we cannot support.

**On naming and prior art.** An earlier draft of this work used the name *AutoQSAR*, which collides with an established Schrödinger product of the same name in the same application area. We have renamed the tool **QSARena** (verified free on PyPI and GitHub), added an explicit disambiguation in the Background, and added a new subsection comparing QSARena against both Schrödinger's AutoQSAR/DeepAutoQSAR and the earlier QSAR Workbench across benchmark breadth, reported performance, licensing, cost and reproducibility. We were careful not to overclaim there: Schrödinger's DeepAutoQSAR benchmark reports qualitative criteria without per-dataset metrics or released predictions, so we state plainly that a direct accuracy comparison is not possible and restrict the comparison to the axes that can be evidenced.

**Declarations.** This manuscript is original, is not under consideration elsewhere, and has not been published previously. The author declares no competing interests. Use of generative AI assistance is disclosed in the Declarations. The work was supported by the California Office of Environmental Health Hazard Assessment. Computation used NSF ACCESS Jetstream2 under allocation CIS261142. All software and benchmark artifacts are openly available under the MIT License at https://github.com/ScottCoffin/QSARena, with a public issue tracker, and a versioned Zenodo archive will be deposited before publication. **[AUTHOR: if you post a preprint, add here — "A preprint of this manuscript has been deposited at arXiv (DOI/ID: …) under a CC BY license."]**

**Suggested reviewers. [AUTHOR: optional — the journal invites 3–5 suggestions. Candidates whose work this paper engages directly include authors of the TDC reproducibility assessment, the MapLight submission, DeepMol, Auto-ADMET and the feature-representation benchmarking literature. Please confirm none has a conflict of interest with you.]**

Thank you for considering this work.

Sincerely,

Scott Coffin
California Office of Environmental Health Hazard Assessment
