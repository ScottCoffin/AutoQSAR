# Submission package — Journal of Cheminformatics

LaTeX submission built on the **Springer Nature template** (`sn-jnl.cls`), per
<https://www.springernature.com/gp/authors/campaigns/latex-author-support>.

## Files

| File | Purpose |
|---|---|
| `manuscript.tex` | **The submission file.** Springer Nature `sn-jnl` class, Vancouver numbered references. |
| `body.tex` | All manuscript sections. Shared by `manuscript.tex` and `proof.tex` — edit prose here, once. |
| `references.bib` | BibTeX, 40 entries, diacritics written as TeX commands per Springer guidance. |
| `additional_file_1.tex` | Additional file 1: supplementary tables S1–S8. Compiles standalone. |
| `additional_file_2_qsarena_tutorial.pdf` / `.tex` | Additional file 2: guided installation and usage tutorial. **Generated** from `../docs/tutorial.md` by `build_additional_file_2.py` — edit the Markdown, never the `.tex`. |
| `cover_letter.md` / `.pdf` | Cover letter. |
| `proof.tex` | Local proof build (standard `article` class, same `body.tex`). **Not for submission.** |
| `tables/*.tex` | Generated table fragments (Tables 1–6, S1–S8) — **do not hand-edit** (see Regenerating). |
| `figures/*.pdf` | Vector figures at publication resolution, including `graphical_abstract.pdf`. |

## Building

`sn-jnl.cls` is **not** bundled (Springer distributes it themselves, and it is not on CTAN). Two routes:

**Overleaf (recommended).** Open the Springer Nature template —
<https://www.overleaf.com/latex/templates/springer-nature-latex-template/gsvvftmrppwq> — then upload
this folder into it. The class is preinstalled there, and Springer's own instructions assume this route.

**Local.** Download the template `.zip` from the Springer page above, copy `sn-jnl.cls` and the
`sn-*.bst` files into this directory, then:

> Verified 2026-09-29 with TeX Live 2026 (`C:\texlive\2026\bin\windows`) and the December 2024
> template (<https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12>).
> That template renamed the class option `sn-vancouver` to `sn-vancouver-num`. With the old name,
> no `\bibliographystyle` is written and every citation is undefined, so `manuscript.tex` now uses
> `sn-vancouver-num`. `sn-jnl.cls` and `sn-*.bst` are gitignored.

```bash
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

**Proofing without the Springer class** (verified working on TeX Live 2025):

```bash
pdflatex proof && bibtex proof && pdflatex proof && pdflatex proof
pdflatex additional_file_1 && pdflatex additional_file_1
```

Additional file 2 (the tutorial) is built from its Markdown source, which is tested command by command
(`tests/docs/test_tutorial_runs.py` runs every command on the bundled example data and checks every
output excerpt). It needs pandoc, `pdflatex` with `fvextra`, and `cairosvg` for the figures:

```bash
python build_additional_file_2.py            # from this directory: .tex + .pdf
```

`proof.tex` compiles the identical `body.tex` under the `article` class, so it catches every content
error; only the class-specific front matter differs. Current status: **0 errors, 0 undefined
references or citations, 38 pages** (Additional file 1: 17 pages; Additional file 2: 32 pages).

## Regenerating tables and figures

Never edit `tables/*.tex` by hand. They are generated from the benchmark artifacts, so that the
LaTeX tables, the Markdown manuscript and the deposited CSVs cannot drift apart:

```bash
cd ..                                                          # repository root
python portable_colab_qsar_bundle/render_manuscript_assets.py  # figures, CSV/MD tables, LaTeX tables, numbers JSON
python portable_colab_qsar_bundle/verify_manuscript_numbers.py # 64 assertions; non-zero exit on drift
```

The first command also refreshes `submission/tables/*.tex`. If a benchmark is rerun, expect
`verify_manuscript_numbers.py` to fail: update the prose in **both** `../manuscript.md` and
`body.tex`, then update the expected values in the verifier.

Figures are copied from `../manuscript_assets/figures/*.pdf`; re-copy after regenerating.

## Journal requirements — status

Structure follows the J. Cheminform. **Research article** format (decided 2026-10-02; previously Software).

- [x] Structured abstract (Background / Methods / Results / Conclusions), 350 words (limit 350; any addition needs a cut)
- [x] **Scientific Contribution** section in the abstract (journal-specific requirement, max 3 sentences)
- [x] Keywords
- [x] Introduction, Methods, Results and discussion, Conclusions
- [x] LLM use documented in the **Methods** (§2.14 "Use of large language models"), as the journal requires;
  LLMs are not authors; no AI-generated images
- [x] Software fields (project name, home page, OS, language, requirements, license) given under Availability of data
  and materials (a separate `Availability and requirements` section is a Software-article requirement only)
- [x] Declarations: Ethics approval · Consent for publication · Availability of data and materials · Competing interests · Funding · Authors' contributions · Acknowledgements
- [x] Abbreviations section
- [x] Additional file 1 cited in the text
- [x] Additional file 2 (guided tutorial, `additional_file_2_qsarena_tutorial.pdf`) cited and listed under
  Supplementary information; every command in it is tested (`tests/docs`)
- [x] Vancouver numbered references (`sn-vancouver-num`)
- [x] Figures as vector PDF, cited in order, captions below
- [x] Code repository linked in Availability of data and materials
- [x] ACCESS/Jetstream2 acknowledgement with allocation CIS261142 and required NSF grant numbers
- [x] Graphical abstract (`figures/graphical_abstract.pdf`; also `manuscript_assets/figures/graphical_abstract.svg`)
- [x] MIT (OSI-approved) license stated under Availability of data and materials
- [x] Results computed from `benchmark_results/qsarena_benchmark_oof_ensemble` (base models from the A100 runs,
  ensembles rebuilt from out-of-fold predictions); verifier: 96 checks
- [x] ORCID 0000-0002-7035-1282 on the title page, in `CITATION.cff` and `.zenodo.json`
- [x] The four formerly `[VERIFY]` references checked against Crossref (2026-10-02)

### Before you submit — outstanding items

1. **ORCID** — done; also enter it in the submission system.
2. **Zenodo DOI (last blocking item)** — archive a tagged release and cite it under Availability of
   data and materials. Follow `../ZENODO.md`; `.zenodo.json` and `CITATION.cff` are already in place. The journal's
   reproducibility editorial specifically asks for an external archive (Zenodo/FigShare) referenced
   from the README, not a bare GitHub link.
3. **Agency disclaimer wording** — confirm OEHHA's required text.
4. **Four references** — verified against Crossref 2026-10-02 (ADDME group author added).
5. **Suggested reviewers** — the journal invites 3–5; see the cover letter.
6. **Preprint** — if posting to arXiv, disclose it at submission (DOI and license). Springer Nature
   does not treat preprints as prior publication.

### Repository checks the journal pilots

From *Improving reproducibility and reusability in the Journal of Cheminformatics*:

- [x] LICENSE file in repository root
- [x] README in repository root
- [x] **Public issue tracker enabled** — enabled, with bug-report and feature-request templates in
  `.github/ISSUE_TEMPLATE/` and submission instructions in the repository README
- [ ] **Externally archived (Zenodo/FigShare) and referenced in the README** — see item 2; `ZENODO.md`
  documents the procedure and `.zenodo.json` is staged
- [x] Installation documentation in README
- [x] Straightforward install (conda/pip specs, Apptainer container)
- [x] **Conforms to an external linter** — `ruff check qsarena tests` runs in CI (`.github/workflows/ci.yml`); the
  legacy `portable_colab_qsar_bundle` modules are excluded in `pyproject.toml`

## Cover letter

`cover_letter.md` is the source; rebuild the PDF after editing it (pandoc ships in the `pypandoc_binary` pip package):

```bash
cd submission
PANDOC=$(python -c "import pypandoc; print(pypandoc.get_pandoc_path())")
"$PANDOC" cover_letter.md -o cover_letter.pdf --pdf-engine=pdflatex -V geometry:margin=1in -V fontsize=11pt
```
