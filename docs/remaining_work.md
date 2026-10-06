QSARena Manuscript — Remaining-Work Instructions for IDE Agent (Final Push)
Target venue: Journal of Cheminformatics, Software/Methodology article, benchmarking collection.Context: The major reframing and supplementary-offload passes are already applied (title/abstract/Conclusions reframed; Figs 8–10 and many tables moved to Additional File 1 as S1–S15; the OECD §3.13 added; the common-subset and post-hoc reanalyses done as S13/S14; the consumer-GPU figure, the LLM disclosure, the deterministic-selection mode, the convergent-literature citations and the dataset-catalog fix all completed). This document is self-contained and covers only the work that remains.Primary goal: reduce the main-text narrative from ~30 pages to under 20 pages without moving any reported result, then close out the final submission-readiness items.

GLOBAL GUARDRAILS
1.	No new benchmark, no multi-seed run, no model retraining. Compute is unavailable. Do not invoke --run-tdc22-multiseed-best or any training.
2.	No invented numbers. Do not change any reported value. Content is being relocated and condensed, not recomputed. If a cut would change a finding, stop and leave % TODO(author).
3.	Move, don't delete, substantive content. Everything cut from the main text lands in Additional File 1 (tables/notes/figures) or is already documented in Additional File 2 (the tutorial). Nothing is lost.
4.	Work on a branch (e.g. revision/jcheminf-r2-length) with atomic commits per phase.
5.	Re-run the verification harness after every phase. The repo ships a script by which numbers regenerate from the deposited artifacts with one command, and a companion script fails if the manuscript text and the artifacts disagree 1. It must pass at the end of each phase.

PHASE 0 — State check (confirm, do not redo)
Open the current manuscript and Additional Files and confirm the following are already present; if any is NOT, flag it in REVISION_NOTES_R2.md and complete it, otherwise leave untouched:
•	Consumer-GPU figure consistent at +0.50% in abstract, §3.10, Limitations and Conclusions — currently +0.50% across the 37 identically split datasets 1 and +0.50% in the Conclusions 1. ✅ expected done.
•	LLM-use section — §2.13, documenting the Claude models used via Claude Code for code, analysis and drafting 1. ✅
•	Deterministic feature-selection mode — removes the time limit, runs single-threaded, can load the deposited A100 selection 1. ✅
•	Convergent-literature citations — refs 13–15 1. ✅ (metadata completeness is checked in Phase 4.)
•	Dataset-catalog mislabeling — "corrected since" 1. ✅
•	Ensemble "not a like-for-like competitor" caveat in Results — present in §3.2 1. ✅ (abstract placement is handled in Phase 5.)
Record the current page map before editing: body ≈ pp. 3–32, with Results & discussion (§3) ≈ p. 13 1 to p. 29 (Limitations) 1, and the reference list from p. 34 1 to ~p. 41 (ref 70) 1. Also confirm which *.tex source file and figure-generation scripts exist, and the exact Additional File 2 section numbers (the config tutorial) for the cross-reference pattern in Phase 2.

PHASE 1 — Set the final title
Replace the title with exactly:
Ensembles Across Model Families Outperform Any Single Family: A Single-Configuration, Leakage-Controlled Benchmark of 31 Molecular Property Models Across 44 Datasets
Remove any previous title variants and any commented alternatives. Confirm the running header / PDF metadata / CITATION.cff title match.

PHASE 2 — Length reduction to < 20 pages (the main task)
Target ~10 body pages of savings. Apply all items below. For each relocation, leave a 1–3 sentence stub in the main text with a pointer to the exact Additional File + section/table.
2A — Compress Methods via Additional File 2 (largest, lowest-risk saving, ~3 pp)
Additional File 2 already documents the full configuration system — featurization (nine fingerprint/descriptor families plus the MapLight-classic composite) 2, splitting (strategy, test fraction, seed, CV folds) 2, the model-library profiles 2, deep-model settings and fusion/CFA 2, the full ensemble parameter set 2, the applicability-domain options and evaluation metrics 2. The main-text Methods duplicate this. Collapse each of the following to the minimum needed to understand the Results, deferring specifics to Additional File 2:

Main-text Methods content	Keep in main (stub)	Defer to
§2.3 molecular representations — the ten RDKit feature families and the MapLight-classic composite 1
one sentence naming the families and that all ten are used by default	Additional File 2, Featurization section
Model library — the full regression/classification estimator enumeration 1
one sentence; the enumeration is already shown in the workflow figure (Fig 2) and in Table S12	AF2 Model-library section + Table S12
Splitting & cross-validation	two sentences (the four strategies, default target-quartile, seed 13, nested CV)	AF2 Splitting + Feature-selection sections
Feature filtering/selection	two sentences (train-only filter + ElasticNetCV, RF fallback, nested in folds)	AF2 Feature-selection section
Fusion / ensemble configuration	two sentences (CFA; OOF stacking; inverse-RMSE averaging)	AF2 Fusion + Ensemble sections
§2.9 evaluation metrics	one sentence (primary metric rule; all metrics written out)	AF2 Evaluation section

Do not remove: the leakage-control description, the nested-selection rationale, the three split-protocol distinction, or anything a reviewer needs to judge validity.
2B — Move the large per-dataset detail figures to Additional File 1
•	Fig 7 — the full-page family-gap heatmap, each family's best-model gap across all 44 datasets with stars marking the winner 1 (currently p. 25–26). Move to Additional File 1. Its message is already carried by Table 1 (family consistency) and Fig 3 (wins by family). Leave one sentence pointing to the SI figure. (~1 pp)
•	Fig 4 panel (b) — the per-dataset rank strip listing all 37 datasets coloured by winning family 1 (p. 18–19). Keep panel (a) (the rank distribution) in main; move panel (b) to Additional File 1. (~0.5 pp)
2C — Condense Results subsections whose detail now lives in the SI
•	§2.12 three-run repair narrative — the base run, the Chemprop-fault repair run, the OOF ensemble rebuild and the retraining exceptions 1 (p. 12–13). Reduce to 2 sentences; move the full account to an Additional File 1 note. (~0.5 pp)
•	§3.9 model coverage/backend failures — already backed by Table S4 1. Reduce to 2 sentences + pointer. (~0.5 pp)
•	§3.10 hardware sensitivity (p. 26–27) — keep the headline +0.50% noise-floor result; move dataset-by-dataset detail to Table S5. (~0.5–1 pp)
•	§3.14 "When does each family win?" meta-analysis (p. 28–29) — the family-vs-meta-feature analysis 1, figures already in SI, backed by Tables S9–S11. Compress to one short paragraph stating it was exploratory and inconclusive; defer to SI. (~0.5–1 pp)
•	§3.12 comparison with automated platforms — the positioning table is already S15 1. Compress the MetaQSAR / Schrödinger / QSAR-Workbench prose to a short paragraph. (~1 pp)
2D — Trim remaining prose
•	Introduction — tighten the AutoML-landscape enumeration (QSARtuna, ZairaChem, DeepMol, DeepPurpose, AutoADMET, ChemXploreML) from a tool-by-tool list to 2–3 sentences. (~0.5 pp)
•	Table 4 (cost) — keeps per-family wall-clock plus published comparators (MolGPS, MolE, ADMET-AI) 1. Keep the core cost columns in main; move the published-comparator columns to the SI cost table. (~0.3 pp)
Expected result
Items 2A–2C recover ~6–7 pp (body → ~23–24); adding 2D reaches ~20–22. If still above 20, lightly condense §3.3 consistency into §3.2 and tighten §3.6/§3.8 prose (§3.8 cost currently spans ~pp. 22–25). Stop there rather than cutting validity-relevant content. Note in REVISION_NOTES_R2.md the final body page count achieved; Journal of Cheminformatics has no hard page limit, so the goal is a tight narrative, not a number for its own sake.

PHASE 3 — Close out reproducibility / availability items
•	3.1 Zenodo DOI. The Availability section still reads "A versioned release archived on Zenodo … will be deposited before publication and its DOI cited here." 1 Prepare the deposit (metadata files, archive staged with the per-molecule prediction files) and insert % TODO(author): mint Zenodo DOI and replace this sentence. The actual minting is the author's step; do not fabricate a DOI.
•	3.2 Unfilled template path. The Availability section lists the consumer-GPU run directory as "benchmark results/benchmark name date/" 1, which is an unfilled template string. Replace it with the real run-directory name (or a generic descriptor) so no placeholder ships. Leave % TODO(author) if the real name can't be resolved from the repo.
•	3.3 Availability-section paths are acceptable where they are. The run-directory and script names in the Availability of data and materials section (e.g. the OOF-ensemble run and verify_manuscript_numbers.py) are appropriate there and should stay — do not move them into the Results/Methods body, and confirm none have leaked back into the narrative.

PHASE 4 — Reference and caveat finishing
•	4.1 Verify the new references. Confirm refs 13, 14, 15 (the dataset-dependent-best-method study, the 2026 splitting/evaluation survey, and the TabPFNv2 reliability study 1) have complete, correct bibliographic metadata (authors, year, venue, DOI) against the publisher record. Do not hand-type author names from memory; verify each.
•	4.2 Abstract caveat. Because the title now asserts "Ensembles … Outperform Any Single Family," add a short clause to the abstract making explicit that ensembles are built from the other families' out-of-fold predictions and are not a like-for-like single model — the caveat already stated in the body ("the ensemble row is not a like-for-like competitor … their member selection and weights use only out-of-fold predictions" 1 and repeated in §3.3 1). One sentence suffices.
•	4.3 Title/graphical-abstract coherence. Confirm the assertive title coexists cleanly with the graphical-abstract taglines that "model-library breadth drives most leaderboard standing" and "no model family dominates" 1; if they read as contradictory, add "of them" / "built from them" to the ensemble line so the hierarchy (no single family dominates → ensembles of the families win most) is unambiguous.

PHASE 5 — Final build and verification
1.	Renumber all figures and tables after the Phase 2 moves (Fig 7 and Fig 4b leave the body; Table 4 columns change) and fix every in-text cross-reference, including Figs 1–2 and all Additional file 1, Table S# / Additional file 2, §# pointers.
2.	Run the verification harness (verify_manuscript_numbers.py or equivalent); it must pass with every relocated and condensed number intact — the paper guarantees a companion script that fails if the manuscript text and the artifacts disagree 1.
3.	Rebuild (LaTeX: full latexmk / pdflatex + bibtex; resolve all warnings; no undefined references or dangling \ref).
4.	Report the final body page count (goal < 20) and the total page count in REVISION_NOTES_R2.md, with a per-phase changelog and every remaining % TODO(author).

DEFINITION OF DONE
•	[ ] Final title set exactly as in Phase 1; headers/metadata/CITATION match.
•	[ ] Methods compressed with pointers to Additional File 2 (Phase 2A); no validity-relevant content lost.
•	[ ] Fig 7 and Fig 4(b) moved to Additional File 1; stubs + pointers left (Phase 2B).
•	[ ] §2.12, §3.9, §3.10, §3.12, §3.14 condensed with SI pointers (Phase 2C).
•	[ ] Introduction tool-list and Table 4 comparator columns trimmed (Phase 2D).
•	[ ] Body narrative < 20 pages (or the closest achievable with validity preserved), recorded in notes.
•	[ ] Zenodo DOI staged with % TODO(author); "benchmark name date" template path replaced (Phase 3).
•	[ ] Refs 13–15 metadata verified; abstract ensemble caveat added; title/graphical-abstract coherence confirmed (Phase 4).
•	[ ] Figures/tables renumbered; verification harness passes; document builds clean (Phase 5).
EXPLICIT DO-NOT LIST
•	Do not run any benchmark, multi-seed, or retraining step.
•	Do not change, recompute, or re-round any reported number; only relocate and condense.
•	Do not delete leakage-control, nested-selection, split-protocol, or other validity-relevant text to save space.
•	Do not fabricate a Zenodo DOI or reference author names; mark as % TODO(author) / verify from source.
•	Do not move Availability-section artifact paths/script names into the narrative body.
References
1.	manuscript.pdf. Internal reference: file:27898#pages=12-13. Accessed 2026-10-06.
2.	additional_file_2_qsarena_tutorial.pdf. Internal reference: file:27897#pages=10-11. Accessed 2026-10-06.
