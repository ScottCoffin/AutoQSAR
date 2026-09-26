# Response to the focused peer review (2026-09-25)

Source: `Peer_Review_Report_QSARena_No_Single_Model_Family_Dominates.docx` (repository root).
Scope as the reviewer set it: everything except the Chemprop failures, ensemble blinding and
multi-seed replication, which are being handled separately.

Every prose change below was made in both `manuscript.md` and `submission/body.tex` (or
`abstract.tex` / `declarations.tex` / `manuscript.tex`). `verify_manuscript_numbers.py` passes
(78 checks), and `manuscript.tex` compiles with no errors, no undefined citations and no new
overfull boxes.

## 2. Major issue A: novelty and positioning

**Done.** The paragraph that claimed a "combination" is gone. The Background now:

- Names DeepChem, QSPRpred, OCHEM, ChemSAR and ADMET-AI, and for each says what it does that
  QSARena does not claim to do (DeepChem: a broader library; OCHEM/ChemSAR: code-free web
  pipelines with many methods already exist) and what it does not report (a uniform cross-suite
  evaluation, or a selection-inflation accounting).
- Moves the stated contribution to the two things the reviewer identified: (1) the uniform,
  leakage-controlled cross-suite benchmark under one fixed configuration, and (2) the
  breadth-vs-selection decomposition.
- Describes the code-free notebook as a usability feature, citing OCHEM and ChemSAR as the reason
  it is not a scientific advance by itself.

§3.12 opens by pointing back to that differentiation, and its closing claim now names the
benchmark and the decomposition as the contribution. Table 6 gains DeepChem, OCHEM and ChemSAR rows.
New references: DeepChem (Ramsundar et al. 2019), OCHEM (Sushko et al. 2011), ChemSAR (Dong et
al. 2017).

The reviewer's DeepChem counts ("60+ architectures, 50+ featurizers") came from a third-party
tool listing, not a primary source, so the manuscript says "a broader model and featurizer library"
without numbers.

## 3. Major issue B: claims vs evidence

**Done.** The estimated ranks are now presented as a provisional secondary analysis everywhere they
appear as a headline:

- **Abstract.** Results now lead with the family distribution, then the internal CV-vs-test gap
  (CV selection never picked the winner; median relative gap 16.1%). The 35→25 of 37 placements
  appear only as the input to the 7+3 decomposition, followed by "These ranks are provisional: the
  reference set includes entries with documented leakage." Conclusions and Scientific Contribution
  lead with the decomposition, and the Scientific Contribution now says why it is sturdier than
  the absolute ranks (both protocols share one run and one reference set). Word count: 345 (tex)
  and 349 (md), both under the 350 cap; Scientific Contribution is 3 sentences.
- **§2.10** gains a third stated limit: the reference set contains self-reported ranks from
  different dates and leaked entries, so every estimated rank is provisional and secondary.
- **§3.4** opens by separating the primary result (the within-run protocol comparison) from the
  secondary one (the ranks).
- **Limitations.** "What survives the single-seed limitation" no longer lists the leaderboard
  placement. It now notes that the placements depend on reference quality, which seed replication
  would not fix. "Estimated ranks are not submissions" now covers contamination as well.
- **Conclusions** rewritten. They lead with the family result and then the selection result, with
  the CV-vs-test gap first. The absolute ranks get their own short, explicitly provisional paragraph.
- **Graphical abstract.** The title changed from "top-10 on 35 of 37" to "model-library breadth
  drives most leaderboard standing" (the renderer computes which effect is larger from the
  numbers). The panel header now reads "(PROVISIONAL)". The caption was rewritten in both formats.
- **Cover letter** point 2 now reports the 7+3 decomposition. Before, it still attributed the whole
  gap to selection, which was the pre-fix error.

## 4. Article type

**Not changed. Needs your decision.** The manuscript is still structured and labelled as a
Software article. The reviewer recommends a Research (benchmarking) article, possibly submitted
to the collection "Evaluating AI and machine learning models in cheminformatics: benchmarking
techniques and case studies". Otherwise, a real head-to-head comparison (QSARena vs
QSPRpred/DeepChem/ADMET-AI on the 22 TDC splits under one protocol) becomes required. Switching
changes the required section structure (Methods instead of Implementation, and no "Availability
and requirements" block), so it was left for you. The cover letter carries an [AUTHOR] flag at
the article-type sentence.

## 5. Minor and mechanical

| # | Item | Status |
|---|---|---|
| 5.1 | Dataset accounting | **Done.** §3.1 now reads: 45 attempted − 1 abandoned (`tdc_herg_central`) = 44 analysed, of which 43 completed normally and 1 (`polaris_adme_fang_hppb_1`) was interrupted but retained. |
| 5.2 | Reviewer anonymity | **Clarified, no mirror created.** The journal's requirement is that *reviewers* can test the software anonymously, not that the author is hidden: J. Cheminform. review is single-blind, and the author is named on the title page. The repository, `pip install` and the tutorial need no account. The one path that does is Google Colab, which needs a Google account. Availability of data and materials now says this and points to local Jupyter as the alternative. If you would still like a code archive as an additional file, the Zenodo deposit (5.3) serves that purpose. |
| 5.3 | Zenodo DOI | **Open, [AUTHOR].** Can't be done from here. The [AUTHOR] placeholder stays in both formats. When depositing, confirm the per-molecule `predictions.csv` files are included (they are gitignored and absent from the repository). |
| 5.4 | Funding / Acknowledgements | **Already present; strengthened.** Both sections already existed in `declarations.tex` and `manuscript.md`, and Acknowledgements already named allocation CIS261142. Funding now also states the in-kind ACCESS compute. |
| 5.5 | Generative-AI statement | **Added, [AUTHOR] to confirm.** A "Use of generative AI" declaration says LLM assistants, including Anthropic's Claude, helped with code, analyses and manuscript editing, and that the author verified all output. Confirm which tools were used and the scope. |
| 5.6 | Data-availability wording | **Done.** Now uses the journal's "The datasets analysed during the current study are available in…" phrasing, with a citation or persistent identifier for each source: TDC, ESOL, FreeSolv, Lipophilicity (ChEMBL DOI 10.6019/CHEMBL3301361), Polaris plus the Fang et al. 2023 ADME source, PODUAM and ChemML (new refs: ChemML, Fang 2023, AstraZeneca ChEMBL deposition). In `manuscript.md` the software fields now sit under their own "Availability and requirements" heading, matching the tex, and the licence line says MIT explicitly. |
| 5.7 | Duplicated sentence | **Fixed.** "We are explicit in Section We are explicit in Section…" was a line-break artifact in `body.tex` only. It is gone with the rewritten Background. Tables 2, 4 and 5 compile without new overfull boxes; the remaining ~20pt overflows are in the landscape Table 6 and are identical at HEAD. |

## 6. Writing style

**Done as a targeted pass, not a rewrite.** Every example the reviewer quoted was removed or
rewritten:
"parity is untested, not lost"; "What the A100 bought was not accuracy but…"; "directional rather than
quantitative"; "The finding, the breadth and the accessibility are the contribution… not in
tension"; "the most practically important number in this paper"; "should be stated plainly rather
than minimized"; "getting it right changed our own conclusions". Also removed or rewritten: "the
corrected decomposition is the more useful result" (along with the sentence narrating an earlier
wrong attribution), "the single most valuable change" in §3.12, "worth stating explicitly", "worth
stating plainly", "a first-class result rather than a caveat", "stronger than the earlier manuscript
allowed", a duplicated ensemble caveat in §3.3, and several "rather than" antitheses. `body.tex`
went from 41 to 20 uses of "rather than". Most of the rest are literal contrasts (MAE rather
than RMSE, 112 rather than 155 hours) and were kept. The title is unchanged, as the reviewer
advised.

## Deferred by the reviewer (not touched here)

Chemprop backend coverage, ensemble member blinding, five-seed replication.
