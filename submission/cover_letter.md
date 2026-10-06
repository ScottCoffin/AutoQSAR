# Cover letter — Journal of Cheminformatics

> Paste into the submission system's cover-letter field, or use `cover_letter.pdf` (built from this
> file with pandoc; see `submission/README.md`). Open items are marked `TODO(author)` in HTML comments.

<!-- TODO(author): The collection "Evaluating AI and machine learning models in cheminformatics: benchmarking
techniques and case studies" was CLOSED FOR SUBMISSIONS when checked on 2026-10-06
(https://link.springer.com/collections/eeefbcafeb; its deadline was 16 January 2026). This letter is therefore
addressed to the journal's editors as a regular Research article. If the guest editors confirm they will still
consider it for the collection, restore the collection lines below. Guest editors, as listed on that page on
2026-10-06: Gonzalo Colmenarejo (IMDEA Food, Madrid), Sebastian Lobentanzer (Helmholtz Zentrum München) and
Oscar Méndez-Lucio (Recursion, Madrid).

  Re: Submission to the collection "Evaluating AI and machine learning models in cheminformatics:
      benchmarking techniques and case studies"
  Dear Dr. Colmenarejo, Dr. Lobentanzer and Dr. Méndez-Lucio,
-->

---

6 October 2026 <!-- TODO(author): set the submission date. -->

To the Editors, *Journal of Cheminformatics*

Dear Editors,

I am pleased to submit **"Ensembles Across Model Families Outperform Any Single Architecture: A Single-Configuration, Leakage-Controlled Benchmark of 31 Molecular Property Models Across 44 Datasets"** for consideration as a
**Research article**.

QSARena is an open-source, MIT-licensed, code-free and command-line benchmarking pipeline that evaluates
31 models — conventional machine learning, gradient boosting, deep tabular and graph neural networks, 3D
pretrained models and ensembles — across 44 molecular property-prediction datasets from five public suites (TDC,
MoleculeNet, Polaris, PODUAM and ChemML) under a single fixed, leakage-controlled configuration.

The contribution is methodological rather than a new state-of-the-art model, and I have framed it that way. That
trees and descriptor-based models remain competitive with deep and pretrained architectures is by now a
well-replicated result, and the manuscript cites the recent cross-suite literature to that effect. What this paper
adds is a uniform, single-configuration benchmark applied across five suites with train-only feature selection,
and — its central result — a decomposition of the gap between test-selected and cross-validation-selected
leaderboard standing into two separable causes: the breadth of the candidate model library and the cost of
honest, held-out selection. Because both protocols derive from one run and one reference set, the decomposition is
less sensitive to the known quality problems of published leaderboards than any absolute rank. The manuscript also
quantifies how much cross-validation optimism nested, train-only feature selection removes.

The work fits the journal's stated commitment to publishing benchmarking studies and to full reproducibility, and
it is in the scope of the journal's recent collection on evaluating AI and machine learning models in
cheminformatics. All code is openly available under an OSI-approved license at github.com/ScottCoffin/QSARena, a
versioned archive including the per-molecule prediction files will be deposited at Zenodo before publication with
its DOI cited in the manuscript, and the manuscript regenerates every reported number from the deposited artifacts
via an automated check. <!-- TODO(author): replace "will be deposited ... before publication" with the minted
Zenodo DOI once it exists. -->

I have stated the principal limitation prominently: all results derive from a single split and seed per dataset,
so conclusions that rest on close margins are presented as provisional. The conclusions I emphasize — the breadth
of the winner distribution, the one-sided selection effect and the order-of-magnitude cost differences — do not
depend on those margins.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. The
author declares no competing interests. The work was supported by the California Office of Environmental Health
Hazard Assessment, and computation used NSF ACCESS Jetstream2 under allocation CIS261142. Use of large language
model assistance is documented in the Methods (Section 2.13), as the journal requires.
<!-- TODO(author): suggested reviewers (optional; the journal invites 3-5). Candidates whose work this paper
engages directly include authors of the TDC leaderboard reproducibility audit, the MapLight submission, DeepMol,
Auto-ADMET and the feature-representation benchmarking literature; confirm none has a conflict of interest. If a
preprint is posted, add its DOI and license here. -->

Thank you for considering this submission.

Sincerely,

Scott Coffin\
California Office of Environmental Health Hazard Assessment, Sacramento, CA, USA\
scott.l.coffin@gmail.com
