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

Please consider the enclosed manuscript, **"No Single Architecture Wins: Ensembles and Conventional Machine Learning Match Pretrained Molecular Models Across 45 Property-Prediction Benchmarks,"** for publication as a **Software article** in the *Journal of Cheminformatics*.

**What the paper reports.** We present AutoQSAR, an open-source QSAR/AutoML workspace that predicts molecular properties from SMILES through one leakage-controlled workflow, and we use it to run what is, to our knowledge, the broadest single-tool cross-suite benchmark published to date: 45 datasets from five collections (Therapeutics Data Commons, MoleculeNet, Polaris ADME, PODUAM and ChemML), 25 models and 828 valid model–dataset evaluations, all under one fixed configuration. The model library spans conventional machine learning and gradient boosting, deep tabular and message-passing graph networks, a tabular foundation model (TabPFN), a 3D pretrained model (Uni-Mol V1), MapLight-style descriptor–graph hybrids, and combinatorial-fusion and stacking ensembles.

**Why it fits this journal.** Three of the paper's contributions speak directly to debates the *Journal of Cheminformatics* has hosted and shaped:

1. **No architecture family dominates.** No family won more than a third of the 45 datasets. Ensembles over conventional learners won the most (15), conventional machine learning next (11), while the 3D pretrained model won seven — all in regression, none in classification — at roughly 115 times the median wall-clock cost. This substantially widens the evidence base for a finding your journal has published repeatedly: that representation and model-family choice, not architectural scale, drive ADMET performance.

2. **We quantify the cost of optimistic model selection.** Choosing the best of 25 models using held-out scores — the protocol implicit in most leaderboard submissions — places AutoQSAR in the estimated top ten on 35 of 37 comparable datasets. Choosing the model by cross-validation alone, with the test set genuinely untouched, drops that to 26 of 37 and first places from six to one. We report both and argue the second is the honest number. We are not aware of another benchmark study that isolates this effect on its own results, and we think it accounts for a meaningful share of the apparent distance between competing leaderboard entries.

3. **Reproducibility is built in, not asserted.** Every figure, table and quoted number regenerates from the deposited artifacts with one command (`render_manuscript_assets.py`), and a companion script (`verify_manuscript_numbers.py`) asserts that the manuscript text still matches those artifacts — 69 automated checks that fail loudly if the text and the data drift apart. This directly answers the code-availability and archival criteria set out in your editorial on improving reproducibility and reusability.

**We have been deliberately conservative about what we claim.** The benchmark is single-split and single-seed: replicating it across five seeds was beyond our compute budget, since the single-seed run alone consumed 155 hours. We state this as the principal limitation, spell out its four specific consequences for our tables, and restrict our conclusions to findings that rest on categorical rather than close differences. We also report backend failures rather than analysing only successes, flag that two comparisons are not leaderboard-equivalent because they use locally generated splits, and disclose that our ensemble member filtering consults held-out data. We would rather present a limited result honestly than a stronger one we cannot support.

**Declarations.** This manuscript is original, is not under consideration elsewhere, and has not been published previously. The author declares no competing interests. The work was supported by the California Office of Environmental Health Hazard Assessment. All software and benchmark artifacts are openly available at https://github.com/ScottCoffin/AutoQSAR, and a versioned Zenodo archive will be deposited before publication. **[AUTHOR: if you post a preprint, add here — "A preprint of this manuscript has been deposited at arXiv (DOI/ID: …) under a CC BY license."]**

**Suggested reviewers. [AUTHOR: optional — the journal invites 3–5 suggestions. Candidates whose work this paper engages directly include authors of the TDC reproducibility assessment, the MapLight submission, DeepMol, Auto-ADMET and the feature-representation benchmarking literature. Please confirm none has a conflict of interest with you.]**

Thank you for considering this work.

Sincerely,

Scott Coffin
California Office of Environmental Health Hazard Assessment
