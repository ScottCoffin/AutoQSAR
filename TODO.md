# QSARena — outstanding work

Live to-do list for the manuscript submission and the tool. Items are ordered by what blocks what.
`publication_recommendations.md` holds the longer-form reviewer-risk analysis; this file is the
actionable checklist.

Last updated: 2026-09-25.

---

## Ensemble OOF repair (2026-09-26) — blocks the manuscript update

- [x] Runner: `--ensemble-member-selection-split oof` (default) builds ensembles from out-of-fold
      member predictions; `train` mode rewarded memorisation (chemprop_fixed ensembles invalid).
- [ ] **Run the OOF ensemble repair on the A100** → `benchmark_results/qsarena_benchmark_oof_ensemble`
      ([submission/chemprop_rerun_command.md](submission/chemprop_rerun_command.md) §7). No full model
      is retrained: Uni-Mol reads its saved `cv.data` OOF predictions, and CPU members refit on 5 folds
      (~10 h in total). **Decide on Chemprop**: `--ensemble-oof-scope all` refits it per fold (~65 h,
      or ~40 h with `--ensemble-oof-folds 3`); `cpu` leaves it out of the ensembles, and the paper
      must say so. TabPFN (metered API, credits capped until 2026-10-01) is left out of the
      ensembles unless `--ensemble-oof-allow-api-refits` is passed or local `tabpfn` is installed.
- [ ] Then regenerate every asset from that run, move `verify_manuscript_numbers.py` to it, and
      update the prose (both formats), abstract, graphical abstract and AGENTS.md framing. Take
      selector scaling and dataset wall-clock from the canonical run, not the repair runs.
- [ ] Disclose the ensemble history in the paper: held-out selection (canonical), then in-sample
      selection (rejected, memorisation), then OOF. Report how much the leak was worth.
- [ ] **Colab notebook ensembles still leak** (`build_colab_qsar_tutorial.py` ~L10990-11360, a
      separate copy of the ensemble code). Member exclusion and the correlated-pair tie-break read
      *test* R² and RMSE. CFA best-per-workflow picks on *test* metrics. Weights and stacking use
      in-sample training predictions. §2.1 says the notebook and runner share identical code paths,
      which is false for ensembles until this is fixed or the text is qualified.

## Raised by focused peer review (2026-09-25, second report)

Source: `Peer_Review_Report_QSARena_No_Single_Model_Family_Dominates.docx`; itemised response in
[submission/response_to_focused_review_2026-09-25.md](submission/response_to_focused_review_2026-09-25.md).

- [x] Novelty re-centred on the uniform cross-suite benchmark + the breadth-vs-selection
      decomposition; explicit differentiation vs DeepChem, QSPRpred, OCHEM, ChemSAR, ADMET-AI.
- [x] Estimated ranks demoted to a provisional, bias-flagged secondary analysis in the abstract,
      conclusions, graphical abstract, Limitations and cover letter.
- [x] 45/43/44 dataset reconciliation; data-availability wording with per-dataset identifiers;
      duplicated-sentence artifact; style pass.
- [ ] **Decide the article type**: Research (benchmarking) article, recommended by the reviewer, or
      Software article, which then needs a real head-to-head vs QSPRpred/DeepChem/ADMET-AI on the
      TDC-22 splits. Switching changes the section structure. The cover letter has an [AUTHOR] flag.
- [ ] **Confirm the generative-AI statement** (tools and scope) in `declarations.tex` / `manuscript.md`.
- [ ] Zenodo deposit must include the per-molecule `predictions.csv` files (reviewer 5.3).
- [ ] Rebuild `submission/cover_letter.pdf` from the updated `cover_letter.md` (it is stale).

## Raised by peer review (2026-09-25)

- [x] **Rename the tool** — `AutoQSAR` collided with Schrödinger's trademarked product in the same
      domain. Now **QSARena**, verified free on PyPI (all spellings) and unused on GitHub. Propagated
      to the manuscript, package, import namespace, console scripts and docs.
- [ ] **Rename the GitHub repository** `ScottCoffin/AutoQSAR` → `ScottCoffin/QSARena`. Not done here
      because it is an outward-facing change on your account; GitHub will redirect the old URL. The
      manuscript and `CITATION.cff` already use the new URL, so this must happen before submission.
- [ ] **Trademark sanity check** (USPTO/EUIPO) on "QSARena" — a free package name is necessary but
      not sufficient given the Schrödinger mark.
- [ ] **Five-seed replication on the 22 official TDC datasets** — the reviewer's primary required
      change. Until it is run, every rank/win figure and table caption is explicitly marked
      provisional (the reviewer's stated alternative). This is the single highest-value open item.
      The runner-side work is now separated from the compute job: use
      `qsarena-benchmark --tdc22-multiseed --tdc22-multiseed-source-run <existing-run>
      --output-dir <new-run>` to run/resume it independently and write `results/tdc22_multiseed.csv`.
- [x] Reference-set contamination (MC-2) discussed and linked to the CV-vs-test analysis.
- [x] Ensemble information-access caveat (MC-3) moved adjacent to the headline claim.
- [x] Chemprop coverage boundary (MC-4) stated in the abstract.
- [x] Self-benchmarking asymmetry (MC-5) added to Limitations.
- [x] All `[VERIFY]` reference tags resolved; Koleiev and ChemXploreML updated to peer-reviewed
      versions; Uni-Mol2, Chemprop v2 and Polaris cited for the versions actually run; intro
      rewritten from primary sources (Sun 2022, Kola & Landis 2004).
- [ ] **Confirm figure quality for production**: the reviewer could not certify resolution, font
      embedding or colour-accessibility from the proof. Figures are vector PDF with a colourblind-safe
      Okabe-Ito palette, but state this explicitly in the cover letter or figure captions.
- [ ] **Reviewer noted missing front matter in the proof.** The structured abstract, keywords and
      declarations live in `manuscript.tex`, not `proof.tex` (a local proofing shim), so they were
      invisible in the reviewed build. Submit `manuscript.tex`, not `proof.tex`.

## Raised by competitive-positioning review (2026-09-25)

- [x] **Repositioned** from an accuracy claim to a framework-and-finding contribution, with an
      explicit concession that accuracy is mid-pack under honest selection.
- [x] **Corrected a factual error**: the manuscript said ADMET-AI does not allow retraining. Verified
      against the repository — `train_tdc_admet_group.py` and `train_tdc_admet_all.py` ship in it, so
      the text now distinguishes the hosted service from the open-source release.
- [x] **Cited the omitted open peer cluster**: QSARtuna (AstraZeneca), ZairaChem (Ersilia), QSPRpred
      (Leiden) and DeepPurpose — all verified against primary sources.
- [x] **Added the landscape table** (Table 7): 15 tools across licence, cost, code-free access,
      retrainability, architecture breadth, suites benchmarked and published TDC performance.
- [x] Corrected the Schrödinger white-paper byline to Kaplan, Ehrlich & Leswing (2022).
- [ ] **Consider a like-for-like accuracy line** against published competitor numbers on the shared
      TDC-22 subset, rather than only estimated ranks. The positioning review asks for this to
      substantiate "match pretrained models" in the title. It needs competitor per-dataset values,
      which are available for MapLight and MaxQsaring but not for DeepAutoQSAR.
- [ ] **Reconsider the title.** It leads with "Ensembles and Conventional Machine Learning Match
      Pretrained Molecular Models", which is a performance framing; the paper's defensible core is
      now the framework plus the selection-effect finding. A title foregrounding the benchmark or the
      honest-selection result may survive review better.

## Chemprop backend failure — root-caused 2026-09-25

- [ ] **Fix the Chemprop harness encoding bug (highest open engineering item).** The A100 run's 86%
      Chemprop failure rate is **not** a model failure: `subprocess.run(..., text=True)` without an
      explicit encoding decoded Chemprop's UTF-8 output as ASCII under the instance's C/POSIX locale,
      raising `UnicodeDecodeError` before any result was read. Windows (cp1252) cannot hit this, which
      is why the RTX run failed differently (genuine `exit=1` training failures, ~50%). Full analysis
      and agent scope in **[CHEMPROP_FIX_PLAN.md](CHEMPROP_FIX_PLAN.md)**.
- [ ] **Then re-establish ADMET-AI parity.** The `Chemprop v2 (D-MPNN + RDKit2D)` variant is our
      ADMET-AI-equivalent architecture; restoring it across all 44 datasets is the cleanest way to earn
      the title's "match pretrained molecular models" claim, which is currently scoped to Uni-Mol only.
- [ ] **Diagnose the Windows Chemprop failures separately** (136 `exit=1`, plus prediction-length
      mismatches that look like a row-alignment bug independent of platform).

## Blocking submission to *Journal of Cheminformatics*

- [ ] **Deposit the Zenodo archive and cite the DOI.** Full instructions in
      [ZENODO.md](ZENODO.md); `.zenodo.json` and `CITATION.cff` are already staged. This is the last
      hard requirement — the journal's reproducibility criteria ask for an external archive
      referenced from the README, not a bare GitHub link, and the paper's *Availability of data and
      materials* section currently carries a placeholder. Needs a Zenodo login, so it cannot be
      automated.
- [ ] **Add your ORCID** to `submission/manuscript.tex`, `CITATION.cff` and the submission system.
- [ ] **Confirm the OEHHA disclaimer wording** in the Acknowledgements.
- [ ] **Verify four references** flagged `[VERIFY]` in `submission/references.bib`: the ADDME 2009
      byline, the MolE article number, the ChemXploreML venue, and the CFA chapter pagination.
- [ ] **Suggest 3–5 reviewers** (the journal invites them); see the note in `submission/cover_letter.md`.
- [ ] **Decide on the arXiv preprint.** Permitted — Springer Nature does not treat preprints as prior
      publication — but the DOI and license must be disclosed at submission.

## Tool: packaging and distribution

- [x] **`pip install qsarena` works locally.** `pyproject.toml` (PEP 621) with 6 CPU-only core
      dependencies, 9 optional extras and two console scripts (`qsarena-benchmark`,
      `qsarena-applicability-domain`). Wheel + sdist build, install into a clean venv, and run from
      outside the repo; the clone-and-run path is byte-identical to before. See the Packaging section
      of `AGENTS.md`.
- [ ] **Publish to PyPI.** The name `qsarena` is free, but note Schrödinger ships a commercial
      product of the same name — worth a trademark sanity check first. Set up a PyPI Trusted
      Publisher for `ScottCoffin/QSARena` plus a release workflow rather than an API token, and test
      on TestPyPI first. Add an author email to `pyproject.toml` if you want it public.
- [ ] **Smoke-test the extras on Python 3.11.** Only 3.14 exists on this machine, so `boosting`,
      `deep`, `graph`, `foundation`, `notebook` and `chemml` were dependency-resolved but never
      actually installed. `python_requires = ">=3.10"` is asserted, not tested — CI on 3.10/3.11
      would make the claim real.
- [ ] **`qsarena[tdc]` is knowingly broken** and excluded from `[all]`: PyTDC 0.4.11 hard-pins
      `numpy==1.26.4`, `pandas==2.1.4`, `scikit-learn==1.2.2` and more, which cannot co-resolve with
      modern versions. Keep the conda/`uv` path as the documented way to get PyTDC. Likewise `dgl`
      and `dgllife` are in no extra, since PyPI only carries ancient versions — the README documents
      the DGL wheel index instead.
- [ ] **Decide on `run_one.py` and the manuscript tooling in the wheel.** `run_one.py` resolves the
      runner by file path so it breaks once installed (it still works from a checkout), and five
      manuscript-tooling scripts (~35 KB) ship in the wheel because setuptools cannot exclude
      individual files from a package — moving them to a `tools/` directory would fix both.
- [ ] **Tag a release** (`v1.0.0`) from a clean tree — needed for the Zenodo webhook anyway.

## Science: the known gaps in the benchmark

- [ ] **Multi-seed replication (highest scientific value).** The single-split, single-seed design is
      the paper's principal stated limitation. The runner now supports both the original end-of-run
      stage (`--run-tdc22-multiseed-best`, seeds 1–5) and an independent/resumable stage
      (`--tdc22-multiseed`) that reads selected models from an existing source run and writes
      `results/tdc22_multiseed.csv`. Re-running the TDC-22 subset would let us report mean ± SD and
      drop most of the hedging in §4.
- [ ] **Fix the Chemprop backend.** Four of five configured variants failed, leaving Chemprop valid
      on only 6 of 44 datasets. This is the largest known threat to the completeness of Figure 2.
- [ ] **Restore TabPFN** (disabled in the canonical run) and **diagnose Uni-Mol V2 164M/310M**, which
      produced no valid results.
- [ ] **Emit cross-validated metrics from the deep backends.** Chemprop, Uni-Mol, MapLight + GNN and
      the fusion methods currently do not, which makes the cross-validation-selected protocol in §3.4
      a conservative lower bound rather than a like-for-like comparison.
- [ ] **Blind the ensemble member filter.** `exclude_negative_test_r2_members` and the
      correlated-member tie-break consult held-out data; disclosed in §4, but it should be changed.

## Repository hygiene

- [ ] **Fix the PODUAM attribution** in `data/benchmark_dataset_catalog.csv` and the cached
      `data/benchmark_leaderboards/leaderboard_top10_reference_*.csv`, which credit "Aurisano et al.,
      Nature Communications 2025". The correct citation is von Borries K, Beckwith KV, Goodman JM,
      Chiu WA, Jolliet O, Fantke P, *Nat Commun* 2026;17:647, doi:10.1038/s41467-025-67374-4.
      The manuscript already cites the correct one.
- [ ] **Delete or regenerate the stale `.txt` mirrors** of `run_qsarena_benchmarks.py` and
      `qsar_workflow_core.py`; they have diverged from the `.py` sources and are a trap for readers
      and agents alike.
- [ ] **Add a linter to CI** (`ruff` or `black`) — the last unmet item on the journal's seven-point
      repository checklist.
- [x] **Keep the two manuscript formats in sync.** `manuscript.md` and `submission/body.tex` carry the
      same prose for the OECD reliability section, and `verify_manuscript_numbers.py` now checks the
      §3.13 numbers in both formats.

## Done

- [x] Merge the NSF ACCESS Jetstream2 A100 benchmark run and make it canonical.
- [x] Rebuild the manuscript against it, including the reversed finding that 3D pretrained models do
      win classification tasks when trained to convergence.
- [x] Add the hardware-sensitivity comparison against the consumer-GPU run (§3.11, Table S5).
- [x] LaTeX submission package on the Springer Nature template, compiling with 0 errors.
- [x] Graphical abstract to journal spec (920×300, ≤150 KB), generated from the numbers JSON.
- [x] Abstract to journal spec (≤350 words, with the required Scientific Contribution section).
- [x] Public issue tracker with templates, and MIT license confirmed.
