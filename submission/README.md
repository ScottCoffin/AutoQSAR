# Submission package — Journal of Cheminformatics

LaTeX submission built on the **Springer Nature template** (`sn-jnl.cls`), per
<https://www.springernature.com/gp/authors/campaigns/latex-author-support>.

## Files

| File | Purpose |
|---|---|
| `manuscript.tex` | **The submission file.** Springer Nature `sn-jnl` class, Vancouver numbered references. |
| `body.tex` | All manuscript sections. Shared by `manuscript.tex` and `proof.tex` — edit prose here, once. |
| `references.bib` | BibTeX, 40 entries, diacritics written as TeX commands per Springer guidance. |
| `additional_file_1.tex` | Additional file 1: supplementary tables S1–S4. Compiles standalone. |
| `cover_letter.md` / `.pdf` | Cover letter. |
| `proof.tex` | Local proof build (standard `article` class, same `body.tex`). **Not for submission.** |
| `tables/*.tex` | Generated table fragments (Tables 1–6, S1–S5) — **do not hand-edit** (see Regenerating). |
| `figures/*.pdf` | Vector figures at publication resolution, including `graphical_abstract.pdf`. |

## Building

`sn-jnl.cls` is **not** bundled (Springer distributes it themselves, and it is not on CTAN). Two routes:

**Overleaf (recommended).** Open the Springer Nature template —
<https://www.overleaf.com/latex/templates/springer-nature-latex-template/gsvvftmrppwq> — then upload
this folder into it. The class is preinstalled there, and Springer's own instructions assume this route.

**Local.** Download the template `.zip` from the Springer page above, copy `sn-jnl.cls` and the
`sn-*.bst` files into this directory, then:

```bash
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

**Proofing without the Springer class** (verified working on TeX Live 2025):

```bash
pdflatex proof && bibtex proof && pdflatex proof && pdflatex proof
pdflatex additional_file_1 && pdflatex additional_file_1
```

`proof.tex` compiles the identical `body.tex` under the `article` class, so it catches every content
error; only the class-specific front matter differs. Current status: **0 errors, 0 undefined
references or citations, 33 pages** (supplementary: 10 pages).

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

Structure follows the BMC/Springer **Software article** format.

- [x] Structured abstract (Background / Implementation / Results / Conclusions), 349 words (limit 350)
- [x] **Scientific Contribution** section in the abstract (journal-specific requirement, max 3 sentences)
- [x] Keywords
- [x] Background, Implementation, Results and discussion, Conclusions
- [x] `Availability and requirements` with all seven required fields
- [x] Declarations in order: Ethics approval · Consent for publication · Availability of data and materials · Competing interests · Funding · Authors' contributions · Acknowledgements
- [x] Abbreviations section
- [x] Additional file 1 cited in the text
- [x] Vancouver numbered references (`sn-vancouver-num`)
- [x] Figures as vector PDF, cited in order, captions below
- [x] Code repository linked in Availability of data and materials
- [x] ACCESS/Jetstream2 acknowledgement with allocation CIS261142 and required NSF grant numbers
- [x] Graphical abstract (`figures/graphical_abstract.pdf`; also `manuscript_assets/figures/graphical_abstract.svg`)
- [x] MIT (OSI-approved) license stated in `Availability and requirements`
- [x] Canonical run is the NSF ACCESS Jetstream2 A100 benchmark (`autoqsar_benchmark_20260623_153839`)

### Before you submit — outstanding items

1. **ORCID** for the author.
2. **Zenodo DOI (last blocking item)** — archive a tagged release and cite it under Availability of
   data and materials. Follow `../ZENODO.md`; `.zenodo.json` and `CITATION.cff` are already in place. The journal's
   reproducibility editorial specifically asks for an external archive (Zenodo/FigShare) referenced
   from the README, not a bare GitHub link.
3. **Agency disclaimer wording** — confirm OEHHA's required text.
4. **Four references** marked `[VERIFY]` in `references.bib` (ADDME byline, MolE article number,
   ChemXploreML venue, CFA pagination).
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
- [ ] **Conforms to an external linter** — not currently enforced; consider adding `ruff`/`black` in CI
