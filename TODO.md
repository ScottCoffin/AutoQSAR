# QSARena — outstanding work

Live to-do list for the manuscript submission and the tool. Items are ordered by what blocks what.
`publication_recommendations.md` holds the longer-form reviewer-risk analysis; this file is the
actionable checklist.

Last updated: 2026-09-25.

---

## Ensemble OOF repair (2026-09-26) — blocks the manuscript update

- [x] Runner: `--ensemble-member-selection-split oof` (default) builds ensembles from out-of-fold
      member predictions; `train` mode rewarded memorisation (chemprop_fixed ensembles invalid).
- [x] **Run the OOF ensemble repair on the A100** (DONE 2026-10-01/02 on the RTX 4060 instead, with Chemprop OOF (`--ensemble-oof-scope all`) and both 2026-09-29 ensemble fixes) → `benchmark_results/qsarena_benchmark_oof_ensemble`
      ([submission/chemprop_rerun_command.md](submission/chemprop_rerun_command.md) §7). No full model
      is retrained: Uni-Mol reads its saved `cv.data` OOF predictions, and CPU members refit on 5 folds
      (~10 h in total). **Decide on Chemprop**: `--ensemble-oof-scope all` refits it per fold (~65 h,
      or ~40 h with `--ensemble-oof-folds 3`); `cpu` leaves it out of the ensembles, and the paper
      must say so. TabPFN (metered API, credits capped until 2026-10-01) is left out of the
      ensembles unless `--ensemble-oof-allow-api-refits` is passed or local `tabpfn` is installed.
- [x] Then regenerate every asset from that run (DONE 2026-10-02; verifier 96/96), move `verify_manuscript_numbers.py` to it, and
      update the prose (both formats), abstract, graphical abstract and AGENTS.md framing. Take
      selector scaling and dataset wall-clock from the canonical run, not the repair runs.
- [x] Disclose the ensemble history in the paper (done 2026-10-03: Methods §2.13 "Benchmark runs and ensemble
      reconstruction"; the leak was worth at least three ensemble wins, 16 -> 13, despite a larger honest pool).
- [x] **Colab notebook ensemble leak fixed** (block 7A in `build_colab_qsar_tutorial.py`). Members were
      "best per workflow by test RMSE"; the filters, CFA best-per-workflow and the downstream strategy
      used test metrics; weights and stacking used in-sample predictions. Everything now runs on OOF
      predictions (conventional/tuned models refitted on K folds in the cell and cached in `STATE`;
      Uni-Mol from `cv.data`).
- [x] **Manuscript §2.1** says the notebook and runner "share identical code paths". That holds for
      features, splits, selection and the base models, but notebook ensembles draw only on
      conventional, tuned and Uni-Mol members; ChemML, TabPFN, MapLight + GNN and Chemprop have no OOF
      predictions inside a notebook session. Qualify the sentence in both formats.
- [x] **Refactor: one shared ensemble implementation** in `qsar_workflow_core.py` for the runner and
      the notebook (they are two copies today, which is how the notebook's leak outlived the
      runner's fix). Plan, specs and acceptance tests:
      [docs/AGENT_WORK_ORDER_shared_ensemble.md](docs/AGENT_WORK_ORDER_shared_ensemble.md).

## Raised by focused peer review (2026-09-25, second report)

Source: `Peer_Review_Report_QSARena_No_Single_Model_Family_Dominates.docx`; itemised response in
[submission/response_to_focused_review_2026-09-25.md](submission/response_to_focused_review_2026-09-25.md).

- [x] Novelty re-centred on the uniform cross-suite benchmark + the breadth-vs-selection
      decomposition; explicit differentiation vs DeepChem, QSPRpred, OCHEM, ChemSAR, ADMET-AI.
- [x] Estimated ranks demoted to a provisional, bias-flagged secondary analysis in the abstract,
      conclusions, graphical abstract, Limitations and cover letter.
- [x] 45/43/44 dataset reconciliation; data-availability wording with per-dataset identifiers;
      duplicated-sentence artifact; style pass.
- [x] **Article type: Research article** (decided by the author 2026-10-02). Structure converted: Introduction /
      Methods / Results and discussion / Conclusions; the software fields moved into Availability of data and
      materials; LLM use documented in Methods §2.14 as the journal requires. The cover letter still needs updating.
- [x] **Confirm the generative-AI statement** (rewritten 2026-10-02 per the J. Cheminform./Springer Nature policy and moved to Methods §2.14; one [AUTHOR] flag remains: add any assistants other than Claude) (tools and scope) in `declarations.tex` / `manuscript.md`.
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
- [ ] **Reconsider the title** (options drafted 2026-10-02; the author picks). The journal asks for a title that names
      the research design where appropriate. The current title's "Ensembles and Conventional Machine Learning Perform
      Comparably" is now shaky: conventional ML wins 4 of 44 datasets, and the ensembles are built from the other families.
      1. (recommended) No Single Model Family Dominates: A Leakage-Controlled Benchmark of 30 Molecular Property Models
         Across 44 Datasets
      2. Library Breadth, Not Honest Selection, Drives Leaderboard Standing: A Single-Configuration Benchmark of 30 Models
         on 44 Molecular Property Datasets
      3. How Much of a Leaderboard Rank Is Model Selection? Test- Versus Cross-Validation-Selected Performance Across 44
         Molecular Property Benchmarks
      4. QSARena: An Open, Leakage-Controlled Benchmark of 30 Molecular Property Models Across Five Suites Under One Fixed
         Configuration
      If it changes: update `manuscript.md` line 1, `submission/manuscript.tex` (`\title[...]` short title too) and the cover
      letter.

## Chemprop backend failure — root-caused 2026-09-25

- [x] **Fix the Chemprop harness encoding bug (DONE: `_SUBPROCESS_TEXT_KWARGS`; Chemprop valid on 38-42/44 per variant in the chemprop_fixed run) (highest open engineering item).** The A100 run's 86%
      Chemprop failure rate is **not** a model failure: `subprocess.run(..., text=True)` without an
      explicit encoding decoded Chemprop's UTF-8 output as ASCII under the instance's C/POSIX locale,
      raising `UnicodeDecodeError` before any result was read. Windows (cp1252) cannot hit this, which
      is why the RTX run failed differently (genuine `exit=1` training failures, ~50%). Full analysis
      and agent scope in **[CHEMPROP_FIX_PLAN.md](CHEMPROP_FIX_PLAN.md)**.
- [ ] **ADMET-AI parity (partly addressed).** The ADMET-AI-like `Chemprop v2 (D-MPNN + RDKit2D)` variant now runs
      on 38/44 datasets, but there is still no head-to-head reproduction of ADMET-AI; §3.12 says parity is untested.
- [x] **Diagnose the Windows Chemprop failures separately** (DONE 2026-09-28: num-workers 0, the CLI launcher with warn-only determinism, and `cmd /c` redirection; see AGENTS.md) (136 `exit=1`, plus prediction-length
      mismatches that look like a row-alignment bug independent of platform).

## Blocking submission to *Journal of Cheminformatics*

- [ ] **Deposit the Zenodo archive and cite the DOI.** Full instructions in
      [ZENODO.md](ZENODO.md); `.zenodo.json` and `CITATION.cff` are already staged. This is the last
      hard requirement — the journal's reproducibility criteria ask for an external archive
      referenced from the README, not a bare GitHub link, and the paper's *Availability of data and
      materials* section currently carries a placeholder. Needs a Zenodo login, so it cannot be
      automated.
- [x] **Add your ORCID** (DONE 2026-10-02: 0000-0002-7035-1282 in manuscript.tex, proof.tex, manuscript.md, CITATION.cff and .zenodo.json; still enter it in the submission system) to `submission/manuscript.tex`, `CITATION.cff` and the submission system.
- [ ] **Confirm the OEHHA disclaimer wording** in the Acknowledgements.
- [x] **Verify four references** (DONE 2026-10-02 against Crossref: all four match; added the ADDME group author, the Alzheimer's Drug Discovery Foundation) flagged `[VERIFY]` in `submission/references.bib`: the ADDME 2009
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
- [x] **Fix the Chemprop backend.** (DONE: valid on 38-42/44 per variant) Four of five configured variants failed, leaving Chemprop valid
      on only 6 of 44 datasets. This is the largest known threat to the completeness of Figure 2.
- [x] **Restore TabPFN** (TabPFN restored on 31/44 (API daily limit hit the rest); Uni-Mol V2 164M/310M not re-diagnosed) (disabled in the canonical run) and **diagnose Uni-Mol V2 164M/310M**, which
      produced no valid results.
- [ ] **Emit cross-validated metrics from the deep backends.** TabPFN now does. Chemprop, Uni-Mol, MapLight + GNN
      and the fusion methods still do not, so the cross-validation-selected protocol in §3.4 remains a conservative
      lower bound rather than a like-for-like comparison.
- [x] **Blind the ensemble member filter.** (DONE: under the default `--ensemble-member-selection-split oof`, member filters and tie-breaks read out-of-fold predictions only) `exclude_negative_test_r2_members` and the
      correlated-member tie-break consult held-out data; disclosed in §4, but it should be changed.

## Feature-expansion arm (in progress, see docs/FEATURE_EXPANSION_PLAN.md)

- [x] Finish GPU XGBoost on `admetboost+chemeleon` and `chemeleon`; evaluate. (null / worse; see the plan)
- [x] Finish `unimol_repr` featurization; train `admetboost+emb` and `emb`; evaluate. (null / worse; see the plan)
- [x] **Recommendations implemented (2026-10-03):** opt-in `--run-admetboost-xgboost` and `--run-chemprop-chemeleon`;
      §3.12 post-hoc paragraph. See "Implementation of the recommendations" in `docs/FEATURE_EXPANSION_PLAN.md`.
- [x] **Full CheMeleon run: DECLINED by the author (2026-10-03).** The opt-in variant stays in the code; the
      single-dataset pilot (Caco-2 MAE 0.382 vs 0.380 for D-MPNN, ~7x slower) is the only CheMeleon fine-tuning result.
- [ ] **Fold `XGBoost (ADMETboost features)` into the benchmark (approved 2026-10-03; in progress).** The base model
      is training into `qsarena_benchmark_oof_ensemble` (`logs/run_foldin_admetboost.ps1`). PAUSED 2026-10-03 at
      34/44 by the author; relaunch the same script to resume. Then rebuild the ensembles
      (once, after the nested-selection decision), render, and update the paper: 31 models, Table 1, §2.13, and §3.12,
      where the descriptor model becomes a member rather than a post-hoc note.
- [ ] **Fix the feature-selection CV leak throughout (REOPENED 2026-10-03; needs a go-ahead for ~50-55 h of RTX compute).**
      Confirmed causally: `qsarena/feature_expansion/selection_leak.py`. Scope (CV metrics, OOF and ensembles, Chemprop
      selected-descriptor OOF, notebook; CFA's in-sample ranking as a related fix), design (pin the selector method per
      dataset) and cost: `docs/NESTED_SELECTION_CV_PLAN.md`. **Code done 2026-10-04** (runner + notebooks; notebook
      not yet executed end to end). Next: a 1 h pilot after the fold-in, then the run (needs the go-ahead).
- [ ] **Hard-label predictions on 4 binary datasets catalogued as rmse** (fixed in code 2026-10-04; 11 models retrained
      and validated, see AGENTS.md). The paper's §2.13 line "base-model results are unchanged from the source runs" must
      then say these were retrained. Decide TabPFN on cyp1a2/cyp2c19/herg_karim/pampa (API credits vs exclude); then the ensemble
      rebuild picks up the probability OOF; re-render the paper (wins/rankings on those 4 datasets will change).
- [ ] CFA ranks fusion candidates by in-sample training error; move it to OOF predictions (needs the CFA stage
      after the OOF stage).

## Repository hygiene

- [x] **Fix the PODUAM attribution** (done 2026-10-02; registry, catalog, latest leaderboard cache; signature-preserving alias in the runner) in `data/benchmark_dataset_catalog.csv` and the cached
      `data/benchmark_leaderboards/leaderboard_top10_reference_*.csv`, which credit "Aurisano et al.,
      Nature Communications 2025". The correct citation is von Borries K, Beckwith KV, Goodman JM,
      Chiu WA, Jolliet O, Fantke P, *Nat Commun* 2026;17:647, doi:10.1038/s41467-025-67374-4.
      The manuscript already cites the correct one.
- [x] **Delete or regenerate the stale `.txt` mirrors** (deleted 2026-10-02) of `run_qsarena_benchmarks.py` and
      `qsar_workflow_core.py`; they have diverged from the `.py` sources and are a trap for readers
      and agents alike.
- [x] **Add a linter to CI** (already present: `ruff check qsarena tests`; CI's pytest step, red since 2026-09-30 from a pandas 3 MergeError, fixed 2026-10-02) (`ruff` or `black`) — the last unmet item on the journal's seven-point
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
