# Revision notes R2: length reduction and submission close-out (work order `docs/remaining_work.md`)

Branch `revision/jcheminf-r2-length`, started 2026-10-06 from `f4b3cb3` (main after the R1 merge). One commit per
phase. No benchmark was run, no model was retrained, no multi-seed stage was invoked, and no reported value was
changed: text was relocated and condensed. `portable_colab_qsar_bundle/verify_manuscript_numbers.py` was run after
every phase. R1's notes (`REVISION_NOTES.md`) remain the record of the previous revision.

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
