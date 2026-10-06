# Revision notes R2: length reduction and submission close-out (work order `docs/remaining_work.md`)

Branch `revision/jcheminf-r2-length`, started 2026-10-06 from `f4b3cb3` (main after the R1 merge). One commit per
phase. No benchmark was run, no model was retrained, no multi-seed stage was invoked, and no reported value was
changed: text was relocated and condensed. `portable_colab_qsar_bundle/verify_manuscript_numbers.py` was run after
every phase. R1's notes (`REVISION_NOTES.md`) remain the record of the previous revision.

## Result

| | Before (`f4b3cb3`) | After |
|---|---|---|
| Main text, Introduction to end of Conclusions (Springer `sn-jnl` build) | pp. 3-31, 29 pages | **pp. 3-24, 22 pages** |
| `submission/manuscript.pdf`, total | 40 pages | 34 pages |
| `submission/proof.pdf` (article class), total | 33 pages | 27 pages |
| `submission/additional_file_1.pdf` | 45 pages | 49 pages |
| `submission/body.tex` words (incl. markup) | 10,173 | 8,815 |
| Abstract | 350 words | 348 words |
| Verifier | 141 checks | 144 checks, all pass |
| Build (`manuscript.tex`) | 0 errors, 0 undefined, 5 overfull boxes | 0 errors, 0 undefined references or citations, 0 overfull boxes |

**The < 20-page goal was not reached; the body is 22 pages.** Every item of Phase 2 was applied, plus the
optional tightening of 3.3, 3.6 and 3.8. The remaining pages are dense text with four figures and four tables;
another 2-3 pages would have to come out of the OECD mapping (3.13, whose numbers the verifier requires in the
main text), the nested-selection evidence (3.4), the decomposition (3.5), the Introduction's argument or the
Limitations, all of which the work order puts out of bounds. The work order says to stop there (the journal has no
page limit). Remaining options for the author, none applied: move Section 3.13 to Additional file 1 with a
two-sentence stub (about 1 page, but the verifier would then check its numbers in the supplement); drop Figure 4
(feature enrichment; Table S3 carries the numbers; about 0.6 page); merge 3.3 into 3.2 (renumbers every later
section and every literal "main-text Section 3.x" in Additional file 1 and the META templates).

Additional file 1 and the proof keep their pre-existing overfull boxes (Additional file 1: 520, all in the wide
generated supplementary tables; proof: 3 long URLs/identifiers); the counts are identical to a build of `f4b3cb3`.

## Phase 0: state check

| Item | Status | Where |
|---|---|---|
| Consumer-GPU figure +0.50% | Present: abstract (+0.5%, rounded to one decimal like every abstract figure), Section 3.10, Limitations, Conclusions; `0.19%` appears nowhere (verifier check `hardware +0.50 everywhere`). | `body.tex`, `abstract.tex` |
| LLM-use section | Present as Section 2.13 (Claude Sonnet 4.6, Claude Opus 5 and Claude Opus 5.5 via Claude Code; Human Chemical, added by the author). | `body.tex` `sec:llm` |
| Deterministic feature-selection mode | Present in Section 2.5 (no time limit, single-threaded, can load the deposited A100 selection); `--deterministic-selection` / `--selected-features-from` in the runner. | `body.tex` `sec:selection` |
| Convergent-literature citations | Present: Green 2023, Li 2026, Zhao 2026 (PDF references 13-15). Metadata re-verified in Phase 4. | `body.tex` Introduction |
| Dataset-catalog mislabeling | Present: "(corrected since)" in Section 2.2. | `body.tex` `sec:datasets` |
| Ensemble "not a like-for-like competitor" caveat | Present in Section 3.2 and repeated in 3.3. Abstract placement: Phase 4. | `body.tex` |

All six were already done; nothing was redone.

### Page map before this revision (Springer `sn-jnl` build, `submission/manuscript.pdf`, 40 pages)

| Part | Pages |
|---|---|
| Title, abstract, graphical abstract | 1-2 |
| Introduction | 3-6 |
| Methods (2.1-2.13) | 6-13 |
| Results and discussion (3.1-3.14) | 13-29 |
| Limitations | 29-31 |
| Conclusions | 31 |
| Abbreviations, supplementary-file list, Declarations | 31-34 |
| References (1-76) | 34-40 |

Main text (Introduction to Conclusions): pp. 3-31, i.e. 29 pages.

Space consumers in the body: Figure 4 (leaderboard ranks; panel b lists all 37 datasets, about one page), Figure 7
(family-gap heatmap, full page), Table 4 (cost; landscape, so a page of its own), Table 1 (families, about one page).

**Figure numbering.** The graphical abstract is a `figure*` float, so in the Springer build it consumed the number
"Figure 1" and the workflow figure printed as Figure 2 (the work order's "Fig 4" and "Fig 7" are the rank and
heatmap figures under that numbering). Fixed in Phase 5: the graphical abstract no longer takes a figure number.

### Sources and scripts

- Main text: `submission/body.tex` (shared by `manuscript.tex` and `proof.tex`) and its Markdown twin `manuscript.md`.
- Abstract: `manuscript.md` -> `python submission/abstract_sync.py` -> `submission/abstract.tex` (350-word cap).
- Additional file 1: `submission/additional_file_1.tex` (hand-maintained notes; generated tables in `submission/tables/`).
- Additional file 2: generated from `docs/tutorial.md` by `submission/build_additional_file_2.py` (pandoc from the
  `pypandoc` bundle of system Python, `.../site-packages/pypandoc/files`, must be on PATH).
- Figures 1-6 (notebook names): `portable_colab_qsar_bundle/benchmark_results_summary.ipynb`, export cell
  `# MANUSCRIPT_FIGURE_EXPORT`, run by `render_manuscript_assets.py`; meta-analysis figures by
  `qsarena/meta_analysis/figures.py`; graphical abstract by `render_graphical_abstract.py`.
- Generated LaTeX tables: `portable_colab_qsar_bundle/render_latex_tables.py`.
- Verification harness: `portable_colab_qsar_bundle/verify_manuscript_numbers.py` (141 checks at the start).

### Additional file 2 section numbers (for the Methods pointers of Phase 2A)

| Topic | Additional file 2 section |
|---|---|
| Featurization | 4.3 |
| Splitting | 4.4 |
| Feature selection | 4.5 |
| Model library | 4.6 |
| GA tuning | 4.7 |
| Deep / pretrained backends | 4.8 |
| Fusion | 4.9 |
| Ensembles | 4.10 |
| Applicability domain | 4.11 |
| Evaluation | 4.12 |
| Model-selection protocol | 4.13 |
| Reproducibility | 9 |

## Phase 1: title

Title set to "Ensembles Across Model Families Outperform Any Single Family: A Single-Configuration,
Leakage-Controlled Benchmark of 31 Molecular Property Models Across 44 Datasets" in `manuscript.tex`, `proof.tex`,
`manuscript.md`, Additional file 1, Additional file 2 (builder and regenerated `.tex`), `cover_letter.md` and
`CITATION.cff` (`preferred-citation`). The commented alternative titles were removed from `manuscript.tex` and
`manuscript.md`. Running header: "Ensembles Across Model Families" (the `\title[...]` short title); the Markdown's
"Running title" line, which still carried an older QSARena running title, now matches it. PDF metadata: `pdftitle`
and `pdfauthor` are now set in both `manuscript.tex` and `proof.tex` (they were empty). `CITATION.cff`'s top-level
`title` and `.zenodo.json` describe the software, not the paper, and were left as they are.

## Phase 2: length reduction

Section numbers are unchanged (2.1-2.13, 3.1-3.14, 4, 5), so every literal "main-text Section x" in Additional
file 1 and the META templates still points to the right place.

### 2A Methods compressed via Additional file 2

| Main text | Now | Detail moved to |
|---|---|---|
| 2.1 Software architecture | Notebook/runner paragraph tightened; the per-dataset stage list is one sentence; the family/inventory sentence moved to 2.6. | Table S12 |
| 2.2 Datasets | Standardization paragraph shortened (all steps kept). The split-protocol distinction (27 official, 17 local) is untouched. | Additional file 2, 4.2 |
| 2.3 Representations | One paragraph naming the ten families (all ten used) and the MapLight composite's three components. | Additional file 2, 4.3; Note S2 |
| 2.4 Splitting | Two sentences: four strategies, target-quartile default, test fraction 0.2, seed 13, folds mirror the split, nested selection. | Additional file 2, 4.4; Note S2 (the full splitting paragraph) |
| 2.5 Selection | One paragraph: train-only pre-filter and ElasticNetCV, 10% cap, RF fallback at 7,200 s, deterministic mode, deposited A100 selection; the nested-selection META paragraph is unchanged. | Additional file 2, 4.5; Note S2 (pre-filter thresholds) |
| 2.6 Model library | One paragraph listing the eight families; the post-hoc descriptor model and the GA statement kept. | Additional file 2, 4.6 and 4.8; Note S2, Table S12 |
| 2.7 Fusion and ensembles | CFA and the three ensembles in two sentences; the leakage-control statements kept (OOF-only selection, filtering, weighting and stacking; test predictions never consulted; label-free clipping; exclusions incl. TabPFN). | Additional file 2, 4.9-4.10; Note S2 (member filters) |
| 2.8 Metrics | One sentence on the metrics written, the primary-metric rule and leaderboard re-selection; the relative-gap definition kept. | Additional file 2, 4.12 |
| 2.11 Environment | Resume/manifest sentence; the A100 paragraph kept. | Additional file 2, 6 and 9; Availability |
| 2.12 Run lineage | Two sentences (three runs; the two retraining exceptions; post-hoc model flag). | Note S1 |

### 2B Figures moved to Additional file 1

- Leaderboard-rank figure: panel (a), the rank distribution, stays as Figure 3; panel (b), the 37-dataset strip,
  is Additional file 1, **Figure S4** (Note S5). The notebook export cell now writes the two as separate figures
  (`figure3_leaderboard_rank`, `figureS4_leaderboard_rank_per_dataset`).
- Family-gap heatmap: Additional file 1, **Figure S5** (Note S6); the main text points to it from 3.2 and 3.9. The
  Note S4 META template now refers to it as Figure S5.

### 2C Results condensed

- 3.9 coverage: one paragraph with pointers to Table S4, Note S6 and Figure S5.
- 3.10 hardware: headline +0.50% (A100 better on 22, RTX on 13, 37 identical splits) kept; detail in Table S5 and
  Note S6.
- 3.12 platforms: one paragraph (QSAR Workbench, Schrodinger, MetaQSAR, ADMET-AI) plus "Where QSARena does not
  lead"; the descriptor-model paragraph moved to Note S3 ("Exploratory analysis behind the descriptor model").
- 3.14 meta-analysis: one paragraph stating that the analysis was exploratory and inconclusive, with the key
  statistics (META template `SUMMARY` in `qsarena/meta_analysis/text.py`).

### 2D Introduction and Table 4

- The AutoML tool-by-tool list is three sentences (all citations kept); the "closest tools" paragraph and the
  research-question paragraph were tightened.
- Table 4 (cost) keeps this run's eight families and the core cost columns (`table6_cost_core`, written by the
  notebook); the full table with MolGPS, MolE and ADMET-AI and the notes column is Additional file 1,
  **Table S16** (Note S6).

### Further tightening (the work order's fallback)

3.3 opening sentence, 3.5 (seed-variance and reference-set paragraphs), 3.6 (fusion and ablation paragraphs), 3.8
(opening; the published comparators are now a pointer), the 3.11 limits sentence, the Introduction's opening and
deep-vs-non-deep paragraphs, and the single-seed Limitations paragraph. Every number in them is still there.

### Layout fixes (no content change)

- Tables 1 and 3 have nine columns, so the table generator set each on a rotated landscape page of its own. They
  are now portrait with abbreviated LaTeX headers (`PORTRAIT_HEADERS` in `render_latex_tables.py`; same cells as
  the CSV and the Markdown) and narrower column gutters. This saved about a page and a half.
- The Section 3.1 footnote (45th dataset abandoned; hPPB completed by the repair run) overfilled its page by
  7.8 pt; it is now the last sentence of that paragraph, with the same content, in both formats.

### Corrections found against the artifacts

- Section 3.3 said "the 15-model conventional family"; Table 1 lists 16 (the post-hoc descriptor model is the
  16th). Now 16.
- Additional file 1, Note S2 said the "remaining 18" datasets used local splits; the artifacts and the main text
  say 17. Now 17 in both formats.

## Phase 3: availability

- The consumer-GPU run's directory is literally `benchmark_results/benchmark_name_date` (the runner's default
  output name); in the PDF it read as an unfilled template. Renaming it would break its recorded paths, the Zenodo
  archive names and the analysis code, so the Availability section now describes it in words and Note S1 names it
  with that explanation.
- Zenodo: `% TODO(author): mint Zenodo DOI and replace this sentence` (both formats). The staged deposit was
  re-verified: all 44 + 45 per-molecule prediction files match the sha256 manifests in `dist/zenodo/`.
- No run-directory or script name appears in the narrative body; they are in the Availability section and
  Additional file 1 only.

## Phase 4: references and caveats

- 4.1 Verified against the publisher records (Crossref; arXiv for the preprint): Li et al. 2026, J Chem Theory
  Comput 22(10):4866-4887, doi:10.1021/acs.jctc.5c02081; Zhao et al. 2026, J Cheminform 18:95,
  doi:10.1186/s13321-026-01217-2; Green et al. 2023, arXiv:2309.17161 (no journal version found), arXiv DOI added.
  Authors, years and venues in `references.bib` match.
- 4.2 Abstract: ensembles are "built from these families' out-of-fold predictions (not like-for-like single
  models)"; to stay within the cap, "As recently reported," and two longer phrases were shortened (348 words).
- 4.3 Graphical abstract: subtitle "no single model family dominates" and panel header "NO SINGLE FAMILY
  DOMINATES" sit above "ensembles of them win the most", so the hierarchy reads the same way as the title.

## Phase 5: build and verification

- The graphical abstract no longer consumes a figure number (sn-jnl stepped the counter for it, so the workflow
  figure printed as Figure 2). Main-text figures are now 1 workflow, 2 wins, 3 rank distribution, 4 features,
  5 cost; tables 1 families, 2 leaderboard summary, 3 ensemble value-add, 4 cost. In-text references are `\ref`s
  in LaTeX and were checked by hand in the Markdown.
- Additional file 1: Tables S1-S16, Notes S1-S6, Figures S1-S5; its preamble paragraph and the "Supplementary
  information" lists in `manuscript.tex` and `manuscript.md` were updated.
- `verify_manuscript_numbers.py`: 144 checks pass. Changes: the descriptor-model paragraph is checked in main text
  plus Additional file 1 (it moved); new checks that the main-text cost table equals Table S16 minus the published
  rows and notes, that Figures S4-S5 and Table S16 are in Additional file 1 and not in the body, and that the title
  matches in every file. No check was loosened.
- Builds: `manuscript.tex` (pdflatex, bibtex, pdflatex twice): 0 errors, 0 undefined references or citations,
  0 overfull boxes. `proof.tex` and Additional file 1 build without errors or undefined references. Additional
  file 2 rebuilt (new subtitle).
- Tests: `pytest -m "not gpu and not slow" tests`: 363 passed, 1 skipped (9 min 50 s); ruff clean on the
  changed Python files.

## Open items for the author (`TODO(author)`)

- Mint the Zenodo DOI and replace the Availability sentence (`declarations.tex`, `manuscript.md`); then add the DOI
  to `REPRODUCIBILITY.md`, `cover_letter.md` and `CITATION.cff`.
- Agency disclaimer wording (`declarations.tex`, `manuscript.md`); remaining acknowledgements (`manuscript.md`).
- Cover letter: submission date, suggested reviewers, preprint DOI if any.
- Optional further length cuts listed under "Result" above.
