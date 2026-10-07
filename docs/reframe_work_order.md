QSARena Manuscript Revision — Instructions for IDE Agent
Target venue: Journal of Cheminformatics (Software or Methodology article; submit to the collection "Evaluating AI and machine learning models in cheminformatics: benchmarking techniques and case studies").Goal: Execute a scientific-merit + presentation revision of the QSARena manuscript and its repository. Reframe the contribution, compress the paper by ~40% into a main text + Additional files, run two no-new-compute reanalyses, and bring the repo into the journal's reproducibility compliance.
v3 changes: added the drop-in reframed Conclusions (Phase 2.4), a consumer-GPU figure reconciliation (Phase 6.5), and a drop-in editor cover letter (Phase 9). v2 had added the exact drop-in abstract, Introduction and BibTeX to Phases 1–2.
GLOBAL GUARDRAILS (read first)
1.	DO NOT run any new benchmark, and DO NOT run multi-seed evaluation. Compute resources are unavailable. The runner's --run-tdc22-multiseed-best stage must remain unused. Do not invoke it, do not schedule it, do not imply it was run.
2.	DO NOT alter, overwrite, or regenerate the deposited benchmark artifacts (the per-model/per-dataset result tables from the A100 run). All reanalysis below reads those existing tables; it does not retrain any model.
3.	DO NOT invent numbers. Every new numeric value inserted into the text must be computed by a script that reads committed artifacts and must be registered with the text-vs-artifact verification harness described in Phase 8.
4.	Preserve scientific meaning. This is a positioning/compression/compliance pass, not a change to results. If a proposed edit would change a reported finding, stop and leave a % TODO(author): ... note instead.
5.	Work on a branch (e.g., revision/jcheminf-r1) and make atomic commits per phase.

PHASE 0 — Repository & manuscript discovery
Before editing, inventory the repo and report findings in a file REVISION_NOTES.md:
•	Detect the manuscript source format: look for *.tex (Springer Nature / BMC template likely, given the "Scientific Contribution" statement and [AUTHOR] placeholders) vs *.docx vs Markdown. Record the main file, the bibliography file (*.bib), and the figure directory.
•	Locate the analysis/plotting scripts and the deposited artifact directory. The manuscript states the artifact trail regenerates every number and that a check script fails if text and artifacts disagree — find that script; it is the verification harness for Phase 8.
•	Confirm which result tables are present. Only per-molecule prediction files are excluded for size; the per-dataset per-model metric tables needed for Phases 4a/4b should be present.
•	Produce a map: section number → source file/line; figure number → source file + generating script. All subsequent phases reference this map.
•	Record the exact in-text citation style and the .bib key convention, so the new references in Phase 2 match it.

PHASE 1 — Title and abstract
1.1 Replace the title. The current title leads with the least-novel claim and omits the tool name. Set the primary title to:
QSARena: separating model-library breadth from held-out selection in a leakage-controlled, cross-suite benchmark of molecular property-prediction models
Leave the following alternatives as a commented block directly above the title for the author to choose from:
•	QSARena: a leakage-controlled, code-free benchmark of 31 molecular property-prediction models across 44 datasets and five suites
•	No model family dominates — and how much of a leaderboard rank is library breadth versus test-set selection: a unified benchmark (QSARena)
Rationale to record in REVISION_NOTES.md: the "no single family dominates" result is now well-replicated; e.g. a prior benchmark found deep models are generally unable to outperform non-deep ones, with tree models on fingerprints performing best 1, and a 2026 Journal of Cheminformatics paper reports the tabular foundation model TabPFNv2 often outperforms customized molecular foundation models, while expert features with tabular models remain highly competitive 2.
1.2 Replace the abstract with the exact text below. Preserve the manuscript's Background / Results / Conclusions + Scientific Contribution + repo-URL structure. The only substantive change is ordering: Results now lead with the nested-selection optimism reduction and the selection decomposition, and "no single family dominated" is demoted to confirmatory. No number has been changed. Paste verbatim:
Background. Molecular property and ADMET prediction is increasingly dominated by ever-larger
pretrained models, whose compute cost and reproducibility limit adoption in academic, regulatory
and small laboratories. Recent cross-suite benchmarks report that this added scale rarely translates
into accuracy, but they rarely ask a second question: how much of a model's apparent leaderboard
standing comes from the breadth of the candidate library and from selecting among candidates on
held-out data, rather than from the model itself. Neither question has been tested across several
benchmark suites under one fixed, leakage-controlled pipeline.

Results. We benchmarked 31 models across 44 datasets (22 regression, 22 classification; 1094
model-dataset evaluations) from five suites under a single fixed configuration with no per-dataset
tuning. Nested, train-only feature selection cut median cross-validation optimism from 15.9% to 3.4%;
cross-validation selection then picked the per-dataset winner on 4 datasets and sat a median of 7.7%
from the test-selected best. Against curated published values, estimated top-ten placement fell from
35 to 27 of 37 comparable datasets when the model was chosen by cross-validation rather than on
held-out data; a matched-candidate-set control attributes 7 of those 8 lost placements to the breadth
of the model library and 1 to held-out selection. Consistent with recent reports, no single model
family dominated: conventional machine learning won 6 datasets, Chemprop and 3D pretrained models 5
each and TabPFN 3, while out-of-fold ensembles built from these families won 22 and were the most
consistent (within 5% of the best model on 84% of datasets). Estimated ranks are provisional, as some
published references have documented leakage, and a consumer-laptop GPU changed the best single-model
score by a median of +0.5%.

Conclusions. Across suites, library breadth drives most leaderboard standing, while held-out selection
still contributes measurable optimism; added model scale was not required for competitive accuracy, and
no tuning, installation or specialized hardware is needed. All results derive from a single split and
seed, so close margins are provisional.

Scientific Contribution. We report a uniform, leakage-controlled benchmark across five suites (44
datasets, 31 models) under one fixed configuration. We decompose the gap between test-selected and
cross-validation-selected standing into library breadth (7 of 37 placements) and held-out selection
(1 of 37), an inflation affecting any comparably selected entry but rarely reported. Both protocols
share one run and reference set, so the decomposition is less sensitive to reference quality than
absolute ranks.

github.com/ScottCoffin/QSARena
Provenance (confirm each number is unchanged): nested selection cut optimism from 15.9% to 3.4%; cross-validation selection then picked the per-dataset winner on 4 datasets and sat a median 7.7% from the test-selected best; top-ten placement fell from 35 to 27 of 37; a matched-candidate-set control attributes 7 of those 8 placements to library breadth and 1 to held-out selection 3; family wins are conventional ML 6, Chemprop and 3D pretrained 5 each, TabPFN 3, ensembles 22, within 5% of best on 84% 3; the consumer GPU figure is a median +0.5% for the best single-model score 3 (see Phase 6.5 on reconciling this with the Conclusions); and the Scientific Contribution block is retained verbatim from the manuscript's decomposition into library breadth (7 of 37) and held-out selection (1 of 37), both protocols sharing one run and reference set 3.

PHASE 2 — Introduction / positioning reframing + citations
2.1 Replace the Introduction with the exact text below. Preserves every reference number ⟨1⟩–⟨31⟩ and inserts five [NEW-*] placeholders (resolved in 2.2). Changes: (a) a convergent-literature sentence in the "architectural scale not well supported" paragraph; (b) a significance-testing citation in the TDC-competitiveness paragraph; (c) the final paragraph rewritten to open with two explicit research questions. Paste verbatim:
1 Introduction

Unfavourable absorption, distribution, metabolism, excretion and toxicity (ADMET) properties remain
among the most consequential causes of failure in drug development. Approximately 90% of drug
candidates that enter clinical testing fail across phases I-III and the subsequent approval process;
the dominant causes are lack of clinical efficacy and unmanageable toxicity, with poor drug-like
properties - including unfavourable pharmacokinetics - contributing a smaller but material share,
having fallen from 30-40% of failures in the 1990s to 10-15% today ⟨1⟩. The value of screening such
liabilities early is well established: inappropriate pharmacokinetics and bioavailability accounted
for roughly 40% of clinical attrition in the early 1990s, a share that fell to about 10% by 2000 as
the industry adopted routine early ADME profiling ⟨2⟩. Because experimental ADMET assays are slow,
costly and hard to scale to the growing number of synthesized and virtual compounds, in silico
prediction of ADMET endpoints from chemical structure has become an indispensable complement to
laboratory screening [3, 4].

Critically, the assumption that architectural scale translates into superior ADMET prediction is not
well supported. In the most comprehensive benchmark of its kind, Xia et al. evaluated twelve
representative models - three non-deep and nine deep - and found that deep models generally fail to
outperform non-deep ones, with gradient-boosted trees and random forests on molecular fingerprints
tending to perform best, because tree models suit the non-smooth target functions characteristic of
molecular property prediction ⟨11⟩. A review in the Annual Review of Biomedical Data Science reached a
concordant conclusion: a substantial and consistent advantage of deep learning over standard machine
learning across diverse datasets and properties has not been demonstrated, and success in
compound-property prediction does not necessarily scale with model complexity ⟨12⟩. Cross-suite
studies since have reinforced this picture rather than overturned it: across diverse real-world
datasets the best method is dataset-dependent, and engineered features with classical learners often
match or exceed deep models [NEW-realworld]; and a 2026 systematic survey and a large reliability
study both report that model rankings are unstable across evaluation protocols, with tabular
foundation models and expert-feature tabular models remaining competitive with - and sometimes
surpassing - customized molecular foundation models [NEW-survey, NEW-revisiting].

These observations are borne out on the TDC leaderboard itself, where gradient boosting on combined
fingerprint and descriptor representations remains highly competitive: extreme gradient boosting
(ADMETboost) ⟨13⟩; CatBoost paired with ECFP, Avalon and ErG fingerprints plus ~200 molecular
properties, which achieved top-3 performance on 16 of 22 benchmarks (the MapLight submission) ⟨14⟩;
AutoML over descriptor sets (CaliciBoost) ⟨15⟩; and automatic feature-combination frameworks built on
simple learners (MaxQsaring, ranked first on 19 of 22 TDC tasks) ⟨16⟩. Systematic representation
studies reinforce this: the choice of molecular representation is often more decisive than model
architecture, and optimal choices are strongly dataset-dependent ⟨17⟩. Controlled comparisons with
formal significance testing likewise find RDKit descriptors to be among the strongest single
representations across ADMET tasks [NEW-hypothesis], including under out-of-distribution evaluation
[NEW-jcim].

Compounding the questionable returns of architectural complexity is a deepening concern over
reproducibility. Across the quantitative sciences, data leakage has been identified as a pervasive and
often invisible cause of over-optimistic results, affecting at least 294 papers across 17 disciplines
in one survey ⟨18⟩.

Among AutoML frameworks, QSARtuna automates the comparison of molecular representations and learners
with Optuna, with uncertainty quantification and explainability by design ⟨22⟩; ZairaChem provides a
fully automated, low-resource AutoML pipeline reported to reach state-of-the-art performance out of
the box on the TDC ADMET binary classification tasks ⟨23⟩; QSPRpred offers a flexible open QSPR
toolkit with standardized serialization for reproducibility ⟨24⟩; DeepMol delivered competitive, fully
reproducible pipelines across 22 TDC ADMET datasets ⟨25⟩; DeepPurpose exposes 15 encoders and more
than 50 architectures in a few lines of code, though it is drug-target-interaction-centric and used
mainly as a TDC baseline ⟨26⟩; and AutoADMET coupled grammar-based genetic programming with a Bayesian
network to produce interpretable pipelines ⟨27⟩. Code-light tools such as ChemXploreML have separately
sought to lower the barrier for non-specialists through a desktop GUI, validated on physicochemical
rather than ADMET endpoints ⟨28⟩.

Most of the capabilities in the present work already exist in this landscape, and several closely
related tools should be distinguished explicitly. DeepChem offers a broader model and featurizer
library than QSARena, with standardized MoleculeNet loaders and a unified
load-featurize-split-train-evaluate API ⟨29⟩; it is a programming library, and it does not report a
uniform cross-suite evaluation of its own model zoo. QSPRpred, published in this journal, overlaps with
QSARena on modularity and reproducibility, offering descriptor x learner model building,
hyperparameter optimization and a standardized serialization scheme through a CLI and a Python API
⟨24⟩; it is a modelling toolkit, not a benchmark, and it has no published cross-suite results. OCHEM
⟨30⟩ and ChemSAR ⟨31⟩ already provide code-free web pipelines that run many QSAR methods, including
descriptor selection, validation and, in OCHEM, applicability-domain assessment, so a code-free
interface with many methods is not new either. ADMET-AI is the natural accuracy comparator ⟨21⟩; it is
a single, well-tuned Chemprop-RDKit architecture trained on 41 TDC datasets, and it does not compare
model families.

Two questions follow, and this paper addresses both. First, whether the scale of modern pretrained
models is warranted for molecular property prediction when every model is run under one fixed
configuration across several benchmark suites. Second, how much of an automated system's apparent
leaderboard standing comes from the breadth of its candidate library and from selecting among
candidates on held-out data, rather than from any single model. What the tools above do not report is
a single fixed configuration applied across several benchmark suites with train-only feature
selection, or an accounting of how much of an AutoML system's apparent leaderboard standing comes from
selecting among candidates on held-out data. These two things are the scientific contribution of this
paper. The first is a uniform, leakage-controlled benchmark: one pipeline, one configuration and no
per-dataset tuning across 44 datasets from five collections, spanning conventional machine learning,
gradient boosting, deep tabular and graph neural networks, 3D pretrained models, descriptor-graph
hybrids, and fusion and stacking ensembles. The second is a decomposition of the gap between
test-selected and cross-validation-selected standing into the value of a broad model library and the
cost of honest selection (Section 3.4). Because both protocols come from the same run, this
decomposition is less exposed to the known problems of the published reference set than any absolute
rank. All results derive from a single split and seed per dataset, so claims that rest on close
margins are reported as provisional; the conclusions we emphasize - the breadth of the winner
distribution, the one-sided selection effect and the order-of-magnitude cost differences - do not.
Provenance of retained verbatim text: paragraph 1 is the ~90% clinical-failure framing with refs 1, 2, 3,4 3; Xia et al./Annual Review are verbatim with refs 11 and 12 3; TDC-competitiveness is verbatim with ADMETboost 13, MapLight 14, CaliciBoost 15, MaxQsaring 16, representation study 17 3; the leakage sentence with the 294-paper, 17-discipline survey 18 3; the AutoML landscape with QSARtuna 22, ZairaChem 23, QSPRpred 24, DeepMol 25, DeepPurpose 26, AutoADMET 27, ChemXploreML 28 3; the tool-distinction paragraph with DeepChem 29, QSPRpred 24, OCHEM 30, ChemSAR 31, ADMET-AI 21 3; and the final paragraph reuses the manuscript's two-part contribution statement, both parts from the same run 3 and its margin-robust claim scoping 3.
2.2 Add the five new references. Paste into the .bib, matching the existing key convention. Every entry has author = {} and % VERIFY tags: resolve each DOI/record on the publisher site, fill the author list and missing metadata, and remove tags only after confirming against the published record. Do not hand-type author names. Then replace each [NEW-*] marker with the corresponding key.
% --- NEW-revisiting : strongest "same finding, independently" citation (J. Cheminform 2026) ---
@article{revisiting_admet_2026,
  title   = {Revisiting ADMET prediction reliability under real-world distribution shift}, % VERIFY exact title/subtitle
  author  = {}, % TODO(author): VERIFY authors from publisher record
  journal = {Journal of Cheminformatics},
  year    = {2026},
  doi     = {10.1186/s13321-026-01217-2},
  note    = {VERIFY volume and article/page number}
}

% --- NEW-realworld : dataset-dependent best method; classical+features competitive ---
@article{realworld_drug_property_2023,
  title        = {Current Methods for Drug Property Prediction in the Real World}, % VERIFY
  author       = {}, % TODO(author): VERIFY
  journal      = {arXiv preprint},
  eprint       = {2309.17161},
  archivePrefix= {arXiv},
  year         = {2023}, % VERIFY (arXiv 2309 = Sep 2023); check for peer-reviewed version
  note         = {VERIFY whether a journal version exists}
}

% --- NEW-survey : rankings unstable across protocols (2026 systematic survey/benchmark) ---
@article{dl_mpp_survey_2026,
  title   = {A Systematic Survey and Benchmark of Deep Learning Methods for Molecular Property Prediction}, % VERIFY exact title
  author  = {}, % TODO(author): VERIFY
  year    = {2026},
  note    = {PMC identifier PMC13218365; VERIFY journal, volume, DOI}
}

% --- NEW-hypothesis : RDKit descriptors strongest single representation (significance-tested) ---
@article{admet_hypothesis_testing,
  title   = {Benchmarking Machine Learning in ADMET Predictions: A Focus on Hypothesis-Testing Practices}, % VERIFY exact title
  author  = {}, % TODO(author): VERIFY
  journal = {ChemRxiv preprint},
  year    = {}, % VERIFY year
  doi     = {}, % VERIFY ChemRxiv DOI (asset URL under chemrxiv.org/engage/.../6578c39f...)
  note    = {VERIFY whether a peer-reviewed version exists}
}

% --- NEW-jcim : out-of-distribution robustness of classical ML and GNNs ---
@article{eval_ml_mpp_jcim_2025,
  title   = {Evaluating Machine Learning Models for Molecular Property Prediction: Performance and Robustness on Out-of-Distribution Data}, % VERIFY exact published title
  author  = {}, % TODO(author): VERIFY
  journal = {Journal of Chemical Information and Modeling},
  volume  = {65},
  number  = {19},
  pages   = {9871}, % VERIFY full page range
  year    = {2025},
  doi     = {} % VERIFY DOI (article at pubs.acs.org/jcisd8/article/65/19/9871)
}
Citation-key mapping:

Introduction marker	BibTeX key	Role
[NEW-realworld]	realworld_drug_property_2023	best method dataset-dependent; features+classical competitive
[NEW-survey]	dl_mpp_survey_2026	rankings unstable across protocols
[NEW-revisiting]	revisiting_admet_2026	TabPFN/expert-feature tabular competitive with foundation models
[NEW-hypothesis]	admet_hypothesis_testing	RDKit descriptors strongest single representation
[NEW-jcim]	eval_ml_mpp_jcim_2025	out-of-distribution robustness

Support: [NEW-realworld] — best method depends on the dataset; engineered features with classical ML often outperform deep learning 4; [NEW-survey] — rankings unstable across evaluation protocols, against any single paradigm being universally optimal 5; [NEW-revisiting] — TabPFNv2 often outperforms customized molecular foundation models; expert-feature tabular models remain competitive 2; [NEW-hypothesis] — RDKit descriptors the best-ranking feature set across ADMET datasets under the Nemenyi test 6; [NEW-jcim] — out-of-distribution robustness of classical ML and GNNs [index 124]. Xia et al. is already ⟨11⟩ — do not duplicate.
2.3 Research question. The two explicit research questions are embedded in the final paragraph of the drop-in Introduction; confirm they mirror the abstract's framing that whether that scale is warranted has not been tested across suites under one pipeline 3.
2.4 Replace the Conclusions (§5) with the exact text below. Reframed to lead with the selection/decomposition contribution and treat "no family dominates" as confirmatory, preserving every reported finding. Note the consumer-GPU sentence uses "well under 1%" pending the Phase 6.5 reconciliation — replace it with the single reconciled figure once resolved. Paste verbatim:
5 Conclusions

Across 44 molecular property benchmarks from five suites, evaluated under one leakage-controlled
pipeline and one fixed configuration, the clearest lessons are about how models are selected and
compared, not about which single model is best. Nested, train-only feature selection cut median
cross-validation optimism from 15.9% to 3.4%, and decomposing estimated leaderboard standing shows
that most of it - 7 of 8 top-ten placements - comes from the breadth of the candidate library and only
1 from selecting on held-out rather than cross-validated data, an inflation that applies to any
comparably selected entry but is rarely reported. Consistent with recent cross-suite reports, no model
family dominated: out-of-fold ensembles over the other families won the most datasets (22 of 44) and
were the most consistent, no single family won more than 6, and conventional machine learning was the
most consistent single family. Compact descriptors - the RDKit 2D panel and pharmacophore fingerprints
- were selected far above their numerical share, and large hashed fingerprints below theirs.
Genetic-algorithm tuning was disabled throughout, so its value is untested here.

Every number above came from one configuration, with no per-dataset feature engineering, architecture
choice or hyperparameter tuning, and from a single split and seed, so conclusions that rest on close
margins are provisional. Repeating the benchmark on a consumer laptop GPU moved the best single-model
score by a median well under 1% across identically split datasets, and the code-free notebook runs in
Google Colab with no installation (Section 3.11). For practitioners who can run only one configuration,
our results support a specific default: a MapLight-style descriptor panel with gradient-boosted trees,
ensembled by inverse-error averaging, adding a 3D pretrained model where the compute budget allows and
the endpoint plausibly depends on molecular shape.
Provenance: the family/descriptor/GA findings are verbatim from the manuscript's Conclusions — no family dominated; out-of-fold ensembles won 22 of 44 and were most consistent; no single family won more than 6; conventional ML was the most consistent single family; compact RDKit 2D and pharmacophore descriptors selected above their share, large hashed fingerprints below; GA tuning disabled 3; the practitioner default and Colab sentence are verbatim from the MapLight-style descriptor panel with gradient-boosted trees, inverse-error averaging, plus a 3D pretrained model where compute allows; code-free notebook in Colab with no installation 3; the decomposition numbers match the 15.9%→3.4% optimism reduction and the 7-of-8 / 1 split 3; and the "single split and seed → provisional" scoping is the manuscript's own stated limitation.

PHASE 3 — Structural compression (main text → Additional files)
Target a main text of ~20–25 pages. Create/extend Additional_file_1 (supplementary PDF) and move the items below. For each move, leave a 1–3 sentence summary in the main text with a pointer to the SI table/figure.
3.1 Move entire exploratory analyses to SI (largest length saving).
•	The family-recommender decision tree (Fig 10 / §3.14). Null result: balanced accuracy 0.32 (95% CI 0.23–0.41) versus a majority-class baseline of 0.25 and label permutation 0.24 (permutation p = 0.138) 3. Move Figure 10 and its text to SI; retain one sentence.
•	The training-size crossover analysis (Fig 8): no crossover in range, with only 30% of bootstrap replicates crossing 3. Move Figure 8 to SI; retain one sentence.
•	Figure 9 (SNN vs achievable metric) → SI.
Keep Figures 1–7 in the main text.
3.2 Compress operational narratives to Methods + SI.
•	§2.13 three-run repair narrative: condense to 3–4 sentences. Full account — a base run where a text-encoding fault discarded most Chemprop results and TabPFN was disabled, a repair run that trained only the missing models, and an ensemble rebuild from out-of-fold predictions 3 — moves to an SI "Run provenance" note.
•	§3.7 computational cost: keep Table 5 and the headline that per-family median cost spans more than three orders of magnitude (0.3 s for CFA up to 374 s for Uni-Mol) 3; per-model breakdown to SI.
•	§3.8 backend failures: two sentences in main; full coverage to SI Table S4. Retain that Uni-Mol V2 produced valid results on only 17 of 44 datasets at 84 M parameters, and the 164 M and 310 M variants produced none 3.
•	§3.10 consumer-GPU reproduction: keep the headline figure (see Phase 6.5); move the dataset-by-dataset comparison to SI Table S5.
3.3 Condense Methods detail.
•	Table 1 (full 31-model inventory with hyperparameters): condensed summary in main, full table to SI.
•	§2.3 featurization: condense; full enumeration (10 feature families, bit widths, radii) to SI.
3.4 Protect against over-claiming on ensembles. Everywhere the ensemble win count (22) appears in main text, keep adjacent the caveat that the ensemble row is not a like-for-like competitor because ensembles are built from the other families' predictions 3.
3.5 Apply the "single-seed → structural" rule (M1, no-compute version). Keep in main text only the margin-robust claims the manuscript already identifies: no single family won more than 6 of 44 datasets; the model-selection gap is large enough to change the leaderboard interpretation; and the cost differences span orders of magnitude 3. Move per-dataset win lists and narrow family-median-gap orderings to SI. Retain the limitation that every result derives from one split and one random seed per dataset, and the TDC convention of mean ± SD over five seeds is not met 3, phrased as a deliberate scope boundary.

PHASE 4 — Reanalyses that require NO new benchmark (run from committed artifacts)
Add each script under the analysis directory, register outputs with the Phase 8 harness, write results into the text.
4.1 Common-subset sensitivity for the pretrained-family comparison (M3). Coverage is uneven — Uni-Mol valid on 17/44 and Chemprop on 38 to 42 of 44 datasets 3. Write a script that identifies the subset of datasets on which all compared families produced a valid result, recomputes each family's median gap-to-best and win counts on that subset, and reports in a short SI table plus a 1–2 sentence main-text statement whether the "pretrained comparable" conclusion holds. Do not retrain anything.
4.2 Bound the post-hoc descriptor-model effect (M5). The descriptor model was trained into the run after it performed well in an exploratory analysis, which can flatter test-selected comparisons 3. Recompute the family-win tally with and without that model from existing artifacts; insert a sentence reporting the delta; add numbers to SI.
4.3 Elevate the nested-selection optimism result. No new computation. Promote to a named main-text subsection with both numbers: the reduction to 3.4% median optimism 3 and the controlled experiment showing fitting selection outside the folds raised the overstatement of CV over test RMSE by a median of 22.7 pp for ElasticNetCV, 6.1 for random forest and 2.5 for SVR, and raised it in 26 of 27 dataset-model pairs (sign test, p < 0.001) 3.

PHASE 5 — Reproducibility & journal compliance
The journal requires work be entirely reproducible by third parties, with datasets, software and algorithms accessible without registration or login, and source code provided 7.
5.1 Zenodo deposit. The manuscript defers to a versioned Zenodo release, including the per-molecule prediction files excluded from the repo for size, to be deposited before publication with its DOI cited here 3. Create deposit metadata (.zenodo.json, CITATION.cff), stage the archive (including per-molecule prediction files, since the repo excludes them and prediction-level diagnostics cannot be reproduced from the public artifacts alone 3), and insert % TODO(author): insert minted Zenodo DOI. Leave minting to the author.
5.2 Cross-hardware feature-selection determinism. The manuscript reports ElasticNetCV selection has a 7,200-second limit after which it falls back to random-forest importance, and which datasets hit the limit depends on hardware; re-running on the RTX 4060 changed the selected features on 6 of 44 datasets 3. Implement (a) a --deterministic mode fixing the selector (disable the time-based fallback or trigger on a dataset-size threshold; pin seeds and thread counts), and (b) ship the canonical A100 selection as a loadable artifact. Document in Methods and REPRODUCIBILITY.md.
5.3 Remove all author/draft placeholders from the body:
•	[AUTHOR: add any other assistants used, for example ChatGPT or GitHub Copilot.] 3 → resolve the AI-assistant disclosure.
•	[AUTHOR: confirm required agency disclaimer wording.] 3 → resolve the OEHHA/agency disclaimer.
•	The Zenodo DOI placeholder from 5.1.Where content can't be supplied, leave a single % TODO(author).
5.4 Strip internal/operational strings from the body (move to SI/README/commit metadata only):
•	Run directory names, e.g. benchmark_results/autoqsar_benchmark_20260623_153839 3.
•	Commit hashes, e.g. repository revision bbfb188 3.
•	CLI flags in prose, e.g. --run-tdc22-multiseed-best 3 and --run-chemprop-chemeleon 3.
•	Internal strings, e.g. "mode disabled, reason empty - ga models" 3 → plain prose.

PHASE 6 — Data-hygiene and consistency fixes
6.1 Fix the dataset catalog mislabeling. Correct the four datasets: tdc cyp1a2 veith, tdc cyp2c19 veith, tdc herg karim and tdc pampa ncats are mislabelled as regression tasks in the dataset catalog, though the analysis correctly infers classification from strictly binary targets 3. Fix the catalog; reduce the mention to one sentence.
6.2 Normalize the incomplete-dataset accounting. State n=44 consistently; footnote the exclusions once: tdc herg central was abandoned after >24 h, and polaris adme fang hppb 1 was interrupted but retained because it had produced a full model table 3.
6.3 De-duplicate the GA-disabled statement. Keep one statement that the genetic-algorithm search was disabled throughout, so its value is untested in this study 3.
6.4 Keep absolute-leaderboard claims marked provisional. Table 6 / §3.12 must retain that no head-to-head comparison was run and none should be inferred from the estimated ranks 3 and that the paper makes no claim to accuracy leadership; MaxQsaring's median rank is better under either protocol 3. Consider dropping the ADMET-AI row, since the Chemprop v2 (D-MPNN + RDKit2D) variant produced valid results on 38 of 44 datasets and parity with ADMET-AI is untested 3.
6.5 Reconcile the consumer-GPU figure (new in v3). The same quantity — the median change in the best single-model score across the 37 identically split datasets — is reported with two different values. Section 3.10 and the abstract give a median of +0.50% (A100 better on 22, consumer GPU on 13) 3 and a median +0.5% 3, while the Conclusions and Limitations give a median of 0.19% 3 and a median of 0.19% across 37 datasets with identical splits 3. Determine from the committed artifacts which is correct (likely signed median vs median absolute change), set the right value, and make all four mentions (abstract, §3.10, Conclusions, Limitations/§3.14) consistent. Until resolved, the drop-in abstract uses "+0.5%" and the drop-in Conclusions uses "well under 1%"; replace both with the single reconciled figure. Register the recomputation with the Phase 8 harness.

PHASE 7 — Optional value-add: regulatory framing (recommended)
Add a short subsection (or Discussion paragraph) mapping QSARena onto the OECD QSAR validation principles. The manuscript already positions MetaQSAR as the closest comparator for a regulatory audience, with explicit alignment to OECD and ECHA good-practice guidance 3, and contains an applicability-domain study using a random forest on Morgan + RDKit descriptors with 20% of each training partition reserved for conformal calibration 3. Keep to <1 page.

PHASE 8 — Verification, cross-references, and build
1.	Run the text-vs-artifact check script (Phase 0). Every number — including Phase 4 reanalyses and the Phase 6.5 reconciliation — must pass. The manuscript advertises a deposited artifact trail from which every number regenerates, together with a script that fails if the text and the artifacts disagree 3 — do not break this guarantee.
2.	Renumber figures and tables after the Phase 3 moves (Figs 8–10 leave the body) and fix every cross-reference, including Figs 1–2.
3.	Rebuild (LaTeX: full latexmk/pdflatex + bibtex; resolve all warnings; no undefined references).
4.	Report the new page/word count; confirm main text ~20–25 pp.
5.	Update REVISION_NOTES.md with a per-phase changelog and every remaining % TODO(author).

PHASE 9 — Editor cover letter (new in v3)
Create cover_letter.md (and/or .docx) with the text below. It foregrounds the decomposition as the contribution and the fit with the benchmarking collection. Resolve every [...] placeholder; verify the current guest-editor roster before using names.
[Date]

To the Editors, Journal of Cheminformatics
Re: Submission to the collection "Evaluating AI and machine learning models in cheminformatics:
    benchmarking techniques and case studies"

Dear Dr. Colmenarejo, Dr. Lobentanzer and Dr. Mendez-Lucio,   % VERIFY current guest-editor roster

I am pleased to submit "[FINAL TITLE]" for consideration as a [Software / Methodology] article in the
above collection.

QSARena is an open-source, MIT-licensed, code-free and command-line benchmarking pipeline that
evaluates 31 models - conventional machine learning, gradient boosting, deep tabular and graph neural
networks, 3D pretrained models and ensembles - across 44 molecular property-prediction datasets from
five public suites (TDC, MoleculeNet, Polaris, PODUAM and ChemML) under a single fixed,
leakage-controlled configuration.

The contribution is methodological rather than a new state-of-the-art model, and I have framed it that
way. That trees and descriptor-based models remain competitive with deep and pretrained architectures
is by now a well-replicated result, and the manuscript cites the recent cross-suite literature to that
effect. What this paper adds is a uniform, single-configuration benchmark applied across five suites
with train-only feature selection, and - its central result - a decomposition of the gap between
test-selected and cross-validation-selected leaderboard standing into two separable causes: the
breadth of the candidate model library and the cost of honest, held-out selection. Because both
protocols derive from one run and one reference set, the decomposition is less sensitive to the known
quality problems of published leaderboards than any absolute rank. The manuscript also quantifies how
much cross-validation optimism nested, train-only feature selection removes.

The work fits the collection and the journal's stated commitment to publishing benchmarking studies
and to full reproducibility. All code is openly available under an OSI-approved license at
github.com/ScottCoffin/QSARena, a versioned archive including the per-molecule prediction files is
deposited at Zenodo ([DOI]), and the manuscript regenerates every reported number from the deposited
artifacts via an automated check.

I have stated the principal limitation prominently: all results derive from a single split and seed per
dataset, so conclusions that rest on close margins are presented as provisional. The conclusions I
emphasize - the breadth of the winner distribution, the one-sided selection effect and the
order-of-magnitude cost differences - do not depend on those margins.

This manuscript is original, has not been published previously, and is not under consideration
elsewhere. [Competing interests: none / as declared.] [Funding/allocation acknowledgements as in the
manuscript.] [Suggested reviewers: ...]  % TODO(author)

Thank you for considering this submission.

Sincerely,
Scott Coffin
California Office of Environmental Health Hazard Assessment, Sacramento, CA, USA
[email]
Support for the cover-letter framing: the collection exists and is guest edited by Dr. Gonzalo Colmenarejo, Dr. Sebastian Lobentanzer and Dr. Oscar Méndez-Lucio 8; the journal has editorially committed to publishing benchmark studies for ML and AI to better understand the utility of different algorithms 9 and to reproducibility; the open-code/no-login requirement is the journal's policy that software and data be accessible without registration and that source code be provided 7; and the contribution/limitation wording tracks the manuscript's own decomposition statement 3 and margin-robust claim scoping 3.

DEFINITION OF DONE
•	[ ] Title replaced; alternatives left as comments.
•	[ ] Abstract replaced with the Phase 1.2 drop-in text verbatim; all numbers preserved.
•	[ ] Introduction replaced with the Phase 2.1 drop-in text verbatim; [NEW-*] markers swapped for BibTeX keys.
•	[ ] Conclusions replaced with the Phase 2.4 drop-in text verbatim.
•	[ ] Five new .bib entries added; all author/% VERIFY fields resolved against publisher records.
•	[ ] Figs 8, 9, 10 and the operational narratives moved to Additional file 1; main text ~20–25 pp.
•	[ ] Phase 4a (common-subset) and 4b (post-hoc bounding) scripts written, run from committed artifacts, numbers inserted and harness-registered. No model retrained; no multi-seed run.
•	[ ] Nested-selection optimism promoted to its own main-text subsection.
•	[ ] Consumer-GPU figure reconciled (Phase 6.5) and consistent across abstract, §3.10, Conclusions and Limitations.
•	[ ] Zenodo deposit metadata staged; cross-hardware determinism addressed; all [AUTHOR] placeholders resolved or marked % TODO(author).
•	[ ] Internal paths/hashes/flags removed from body; dataset catalog fixed; GA statement de-duplicated.
•	[ ] (Optional) OECD validation-principles subsection added.
•	[ ] cover_letter.md created from Phase 9 with all placeholders resolved and the editor roster verified.
•	[ ] Verification harness passes; figure/table cross-refs correct; document builds clean.
EXPLICIT DO-NOT LIST
•	Do not run --run-tdc22-multiseed-best or any multi-seed/new benchmark.
•	Do not retrain any model or regenerate deposited artifacts.
•	Do not write any number into the text that is not produced by a harness-registered script reading committed artifacts.
•	Do not hand-type reference author names or guest-editor names; verify from the source.
•	Do not soften or remove the single-split limitation; scope the claims instead.
References
1.	Why Deep Models Often Cannot Beat Non-. Retrieved 2026-10-06, from https://arxiv.org/pdf/2306.17702
2.	Zhao D, Zhu Y, Wu Z, Wan Y, Liu X, Li S, Xu H, Hou T, Hsieh CY. Revisiting ADMET prediction reliability under real-world challenges in the foundation model era. J Cheminform. 2026;18(1):95. doi:10.1186/s13321-026-01217-2. PMID: 42152045.
3.	manuscript.pdf. Internal reference: file:27834#pages=1-3. Accessed 2026-10-06.
4.	Current Methods for Drug Property Prediction in the Real World. Retrieved 2026-10-06, from https://ar5iv.labs.arxiv.org/html/2309.17161
5.	Li Z, Chen X, Wen H, Zhang RQ, Li M, Zhang X, Yin H, Yang Q, Lam KY, Lio P, Yiu SM. A Systematic Survey and Benchmark of Deep Learning for Molecular Property Prediction in the Foundation Model Era. J Chem Theory Comput. 2026;22(10):4866-4887. doi:10.1021/acs.jctc.5c02081. PMID: 42096352.
6.	Benchmarking ML in ADMET predictions: A. Retrieved 2026-10-06, from https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6578c39fbec7913d2774d6e6/original/benchmarking-ml-in-admet-predictions-a-focus-on-hypothesis-testing-practices.pdf
7.	Review | Journal of Cheminformatics | Springer Nature Link. [bibliographic details unresolved] https://link.springer.com/journal/13321/submission-guidelines/review?error=cookies_not_supported&code=6734f403-be72-4d41-b1bd-e720ea3b757e
8.	Journal of Cheminformatics | Springer Nature Link. [bibliographic details unresolved] https://link.springer.com/journal/13321
9.	Zdrazil B, Guha R. Diversifying cheminformatics. J Cheminform. 2022;14(1):25. doi:10.1186/s13321-022-00597-5. PMID: 35468863.
