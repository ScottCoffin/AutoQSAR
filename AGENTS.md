# AGENTS.md — QSARena

Orientation for coding agents. Read this before exploring; most of it took a full session to discover.
Human-facing docs: `README.md` (usage), `CONTAINER.md` / `hpc/README.md` / `js2/README.md` (HPC),
`publication_recommendations.md` (reviewer-risk checklist). Keep this file current when you learn something
that would have saved you time.

## What this repo is

QSARena: SMILES → molecular property QSAR/AutoML workspace plus a 45-dataset benchmark and a manuscript
(`manuscript.md`, target: *Journal of Cheminformatics*). Windows + OneDrive checkout (paths contain spaces:
always quote). Bash (Git Bash) and PowerShell are both available.

## Where to edit (source of truth)

| Want to change | Edit | Notes |
|---|---|---|
| Interactive Colab notebook | `portable_colab_qsar_bundle/build_colab_qsar_tutorial.py` | Regenerates `colab_qsar_tutorial.ipynb`; never hand-edit that notebook. |
| Benchmark runner | `portable_colab_qsar_bundle/run_qsarena_benchmarks.py` | ~10k lines. `run_qsarena_benchmarks.txt` is a **stale mirror**: do not read or grep it as source. |
| Features, splits, CFA | `portable_colab_qsar_bundle/qsar_workflow_core.py` | (`qsar_workflow_core.txt` is a mirror.) |
| Dataset registry / catalog | `benchmark_registry.py`, `data/benchmark_dataset_catalog.csv` | |
| Leaderboard references | `data/benchmark_leaderboards/*.csv` | Current-literature ESOL/Lipophilicity refs live in `ESOL_Lipophilicity_Current_Benchmarks_csv.csv`. |
| Benchmark analysis, manuscript figures and tables | `portable_colab_qsar_bundle/benchmark_results_summary.ipynb` | Hand-maintained (no builder). The last cell (`# MANUSCRIPT_FIGURE_EXPORT`) writes `manuscript_assets/`. |
| Graphical abstract | `portable_colab_qsar_bundle/render_graphical_abstract.py` | Writes `manuscript_assets/figures/graphical_abstract.svg`. It should summarize the paper's decision story, not duplicate Figure 1. |
| Precision, Uni-Mol2, conformers (Update 2) | `qsarena/` package, `run_one.py`, `js2/`, `hpc/` | |
| pip packaging (deps, extras, console scripts) | `pyproject.toml` | See "Packaging" below. |

## Packaging (`pip install qsarena`)

- Both `qsarena/` and `portable_colab_qsar_bundle/` ship as real packages; the bundle got an
  `__init__.py` purely for that. The cross-imports in `run_qsarena_benchmarks.py` already
  used `from portable_colab_qsar_bundle.X import ...` with a `sys.path` fallback, so script-style
  and installed use both work. Don't "simplify" that try/except.
- `workspace_root()` in the runner replaces the old `Path(__file__).resolve().parents[1]` for
  `data/`, `model_cache/` and the leaderboard cache: `QSARENA_HOME` > source checkout (detected by
  `pyproject.toml`/`.git`/`environment-cpu.yml`) > CWD. Without it, an installed copy would write
  into `site-packages`.
- The `data/` tree is **not** bundled in the wheel (tens of MB, mostly regenerable). Missing catalog
  and leaderboard CSVs already degrade to empty dicts / skipped comparisons.
- **PyTDC hard-pins** `numpy==1.26.4`, `pandas==2.1.4`, `scikit-learn==1.2.2`, `rdkit==2023.9.5`
  and `dgl`, so it is in its own `[tdc]` extra and deliberately excluded from `[all]`; it only
  resolves on Python 3.10-3.12. `dgl`/`dgllife` are not in any extra at all (no usable PyPI wheels).
- Build/verify: `python -m build`, then install the wheel in a venv **outside** the repo and with a
  **short** path — a long Windows path makes RDKit fail with
  `ImportError: DLL load failed while importing cDataStructs`.

## Manuscript workflow (the fast path)

1. **Canonical run: `benchmark_results/autoqsar_benchmark_20260623_153839`** — the NSF ACCESS
   Jetstream2 **A100** run (`g3.large`, allocation CIS261142), `full` profile, 28 models incl.
   Uni-Mol V1+V2, committed on origin/main in `bbfb188`. Confirmed A100 by `run_timing.json`
   (`NVIDIA A100-SXM4-40GB`, 32 CPUs), UTC timestamps and zero Windows paths.
   44 datasets analysed (22 reg / 22 cls), 43 fully complete; `tdc_herg_central` abandoned (>24 h),
   `polaris_adme_fang_hppb_1` interrupted but has a full model table.
   The earlier **RTX 4060** run (`benchmark_results/benchmark_name_date`, `cost_optimized`) is kept
   deliberately: §3.11 / Table S5 compare the two. Do not delete it.
2. Regenerate every figure, table and number (~1.5 min; system Python suffices):
   ```bash
   python portable_colab_qsar_bundle/render_manuscript_assets.py   # notebook + figures + tables + numbers JSON + LaTeX tables
   python portable_colab_qsar_bundle/render_graphical_abstract.py  # 920x300 J.Cheminform graphical abstract
   python portable_colab_qsar_bundle/verify_manuscript_numbers.py  # 64 assertions; non-zero exit on drift
   ```
   The first also rewrites the `<!-- TABLE:stem -->` blocks in `manuscript.md` and
   `submission/tables/*.tex`. **Never hand-edit inside those blocks or those .tex files.**
3. **Two manuscript formats must stay in sync**: `manuscript.md` (working doc, checked by the
   verifier) and `submission/body.tex` (the submission). Prose edits must be made in both.
   `verify_manuscript_numbers.py` only checks `manuscript.md`, so a number fixed there but not in
   `body.tex` will pass silently — grep `body.tex` after any numeric change.
4. Numbers come only from `manuscript_assets/manuscript_numbers.json` or
   `manuscript_assets/tables/*.csv`. Never from old notebook outputs, `Manuscript Outline.md` or
   `publication_recommendations.md` (all predate the A100 run).

## Paper's framing (do not weaken these without evidence)

The paper makes three load-bearing claims. Keep them straight when editing:
1. **Leaderboard placement**: top-10 on 35/37 (22/22 on official TDC splits), median rank 3.
2. **Honest model selection costs ~10 placements**: CV-selected falls to 25/37, median rank 8. This
   is the novel methodological contribution; never drop it to make the headline look better.
3. **No effort and no hardware required**: all 44 datasets ran under ONE fixed configuration with no
   per-dataset tuning and GA disabled, and the notebook runs code-free in Colab with no install.
   §3.12 states this and its limits. It is evidenced by the run design, not a marketing line.

Open work is tracked in [TODO.md](TODO.md); the Zenodo deposit is the last blocking submission item.

## Journal requirements already encoded (J. Cheminform., Software article)

- Abstract is capped at **350 words** and must contain a **Scientific Contribution** section
  (max 3 sentences). Both are in place; re-check the word count after any abstract edit.
- Graphical abstract spec: **920x300 px, <=150 KB, white background**. `render_graphical_abstract.py`
  emits exactly that and reads its statistics from `manuscript_numbers.json`.
- Structure: Background / Implementation / Results and discussion / Conclusions /
  **Availability and requirements** (seven fields) / Declarations (seven subsections) / Abbreviations.
  Note `Availability and requirements` and the `Availability of data and materials` declaration are
  two different required sections.
- No "highlights" section is required (that is an Elsevier convention).
- Preprints are permitted and are not prior publication; disclose DOI and license at submission.
- License is **MIT**; the GitHub issue tracker is enabled with templates in `.github/ISSUE_TEMPLATE/`.
  Zenodo archive is **not yet deposited** — see `ZENODO.md`, `.zenodo.json`, `CITATION.cff`.

## Data and analysis traps (all verified; each one changed headline results)

- **`groupby(...).first()` is column-wise first-non-null, not first row.** It grafted `error` text from failed
  retry rows onto valid results and silently dropped them (e.g. MapLight+GNN appeared valid on 7/45 datasets
  instead of 42). The notebook now uses `.nth(0)`; never reintroduce `.first()` for whole-row selection.
- **`elapsed_seconds` in `metrics.csv` is a cumulative per-session clock** that resets on resume. Per-model cost is
  the difference between consecutive rows (see `incremental_model_runtime` in the export cell). Older notebook
  cost cells (cell 12, `PUBLICATION_COST_TABLE`) still use the cumulative value and overstate slow families by orders of
  magnitude (the old "MapLight+GNN ≈ 13 h" was an artifact; the real median is ≈150 s).
- **`analysis_delta_from_best` is an absolute difference**, so it mixes units (clearance RMSE ≈ 40 vs LogS ≈ 0.6).
  Use `relative_gap_to_best` from the export cell for cross-dataset summaries.
- **Test-set leakage:** the per-dataset "best model" is still chosen on the test set (disclose this). Ensemble
  member filtering **was** leaky too and is now **fixed**: `build_ensemble_result` takes
  `member_selection_split`, exposed as `--ensemble-member-selection-split {train,test}` and defaulting to
  `train`. The deposited run predates the flag, so a `run_config.json` with no
  `ensemble_member_selection_split` key means the legacy leaky `test` behaviour — pass `--ensemble-member-selection-split test`
  to reproduce it exactly. **Ensembles must be re-run for the fix to show up in results**; see
  [submission/chemprop_rerun_command.md](submission/chemprop_rerun_command.md). The export cell reports a
  CV-selected sensitivity analysis (only conventional models, TabPFN and ChemML MLP have CV metrics).
- **All `subprocess.run` calls must use `_SUBPROCESS_TEXT_KWARGS`** (`text`/`encoding="utf-8"`/`errors="replace"`),
  never bare `text=True`. Bare `text=True` decodes with the ambient locale; under the C/POSIX locale on
  Jetstream2 that was ASCII, and Chemprop v2's UTF-8 progress output (`0xe2`) made `subprocess.run` itself
  raise `UnicodeDecodeError` before any result was read. That destroyed **86.4% of Chemprop runs at an
  identical rate across all five variants** and was recorded in the same `error` column as genuine training
  failures, which is why it hid for a whole run. Harness I/O errors now say `Chemprop harness error`;
  model failures say `Chemprop training failed`. Regression test: `tests/test_subprocess_encoding.py`
  (needs pytest, which the Windows workstation does not have).
- **Targeted model filters are repeatable exact labels.** Use one `--only-model-names` argument per
  model. Model labels contain commas, so comma-joining labels silently breaks selection. Internal
  TDC multi-seed code stores these filters as a list for the same reason.
- **Chemprop probe runs need at least 3 epochs.** Chemprop v2 defaults to two warmup epochs and rejects
  `--chemprop-epochs 2` before training.
- **Do not seed a repair run with metrics alone.** Ensemble reconstruction needs the ignored
  per-model `predictions.csv` files. Use `prepare_chemprop_repair_run.py`, which copies top-level
  resume artifacts and predictions but no model directories into a new output directory.
- **RDKit 2026.03.1 cannot execute notebook cell 19** for the canonical structures (`bad bond stereo`).
  Use the pinned publication environment before regenerating manuscript assets; a failed execution
  can still overwrite export files, so restore them before interpreting verifier failures.
- **`select_output_dir` now globs `qsarena_benchmark_*`** after the rename, so `--resume` will NOT find the
  canonical `autoqsar_benchmark_20260623_153839`. Do not "fix" the glob to resume into the canonical run;
  always write re-runs to a new `--output-dir`.
- Leaderboard comparisons: use `UPDATED_LEADERBOARD_COMPARISON` (cell 21), not `leaderboard_eval` (cell 9 still scores
  ESOL/Lipophilicity against 2017 MoleculeNet baselines). Only the 22 TDC ADMET Benchmark Group datasets and 5 Polaris
  sets use official splits; tox21/toxcast are single-label subsets; carcinogens, skin_reaction, clintox, hydration-FreeSolv,
  PODUAM and MoleculeNet sets use local splits → "estimated rank", not leaderboard-equivalent.
- **The curated `TDC_ADMET_Benchmark_Performance_by_Model.csv` `TDC_Rank` column is each paper's SELF-REPORTED
  rank at its own publication date, not a leaderboard position.** 64 rows claim rank 1 across 28 datasets;
  `caco2_wang` alone has four. Re-score from `Score_Mean` before quoting any ranking. Doing so turns
  ADMETboost's "first on 18/22" into **0 firsts** (it was true in 2022 and has since been beaten on all 22)
  and MaxQsaring's "19/22 firsts" into **7 firsts / 19 top-3 / median rank 2**. Full audit:
  [LEADERBOARD_PROVENANCE_FINDINGS.md](LEADERBOARD_PROVENANCE_FINDINGS.md).
- **MaxQsaring, ADMETboost, ADMET-AI, DeepAutoQSAR, Auto-ADMET and QW-MTL do not appear in the scraped actual
  TDC top-10s at all.** Any comparison against them is publication-vs-publication, never head-to-head.
  DeepAutoQSAR's "top performer on 20 of 22" is *best-of-three* against ChemProp and DeepPurpose in
  Schrödinger's own white paper — not a leaderboard rank, so it never conflicted with MaxQsaring's claim.
- **Only 24 of 44 datasets carry any leaderboard reference, and only 9 are backed entirely by the actual
  scraped TDC leaderboard** (`caco2_wang`, `clearance_hepatocyte_az`, `clearance_microsome_az`,
  `half_life_obach`, `ld50_zhu`, `lipophilicity_astrazeneca`, `ppbr_az`, `solubility_aqsoldb`,
  `vdss_lombardo`). The notebook reaches 37 by merging curated literature CSVs. Check
  `reference_source` (`tdc` / `literature_static` / unlabelled) before calling anything leaderboard-equivalent.
- **MolGPS (3B) has three wrong-scale rows** in the curated CSV (it reports normalised targets):
  `ppbr_az` MAE 0.679 vs real 7.440, `ld50_zhu` MAE 0.292 vs 0.552, `vdss_lombardo` Spearman 0.942 vs 0.713.
  They produce 3 of its 7 apparent first places. Drop them.
- When ranking our own results against references, **filter to rows whose `primary_metric` matches the
  leaderboard metric first.** Spearman-primary datasets also carry `mae` rows in `primary_metric_value`;
  taking a naive max mixes units and yields absurdities (a "Spearman" of 32.1 on `clearance_hepatocyte_az`).
- The notebook's feature-family labels split MapLight classic into `avalon` + `erg` + `maplight` (descriptor panel). Sum them for "MapLight".
- Catalog metadata is wrong for some tasks (e.g. `tdc_cyp1a2_veith`, `tdc_cyp2c19_veith`, `tdc_herg_karim` list `rmse`/scaffold but
  are binary): the notebook infers classification from strict 0/1 targets, and that inference is what the analysis uses.
- Regression winners are ranked by RMSE even where the TDC leaderboard metric is MAE/Spearman; the leaderboard table
  re-selects the best model per leaderboard metric, so "winner" and "leaderboard model" can differ.
- Backend failures in the canonical run: Chemprop failed on ~half the datasets (all classification tasks) and
  MapLight+GNN hit a DGL `graphbolt` DLL error on first attempts (retries succeeded on 42/45). See `tableS4_model_coverage`.
- `tdc_herg_central` is the one dataset with `status: "running"` and no metrics: it is very large, ran >24 h without
  finishing, and was deliberately abandoned (user, 2026-09-24). Report it as dropped for compute budget, not as a failure.
- **Multi-seed was never run, by choice, not oversight** (user, 2026-09-24): repeating 45 datasets × 25 models five times
  was beyond budget (the single-seed run alone took 155 h). `--run-tdc22-multiseed-best` exists but has no artifacts.
  The manuscript hedges this at length in §4; do not soften that hedging or present margins as established.
- **Host provenance: resolved.** The A100 run is real and is now canonical (see above). The RTX 4060
  evidence applies only to `benchmark_results/benchmark_name_date`, which is retained as the
  hardware-comparison arm. On 37 identically split datasets the two runs differ by a median of
  -0.19%, which is the paper's accessibility result, not a discrepancy to fix.
- **PODUAM is misattributed in repo metadata.** `data/benchmark_dataset_catalog.csv` and the cached
  `data/benchmark_leaderboards/leaderboard_top10_reference_*.csv` credit "Aurisano et al., Nature Communications 2025".
  The actual PODUAM paper is von Borries K, Beckwith KV, Goodman JM, Chiu WA, Jolliet O, Fantke P, *Nat Commun*
  2026;17:647, doi:10.1038/s41467-025-67374-4 (software: github.com/kejbo/PODUAM). The manuscript cites the correct one;
  the repo metadata still needs fixing.
- `predictions.csv` files are gitignored and absent locally, so prediction-diversity panels are skipped (expected).
- No TDC-22 multi-seed artifacts exist for the canonical run (`tdc22_best_model_multiseed*`); results are single-split, single-seed.

## History worth knowing

- The manuscript was first drafted (commit `46bcd3d`) from notebook outputs of an older run,
  `all_benchmarks_no_unimol_20260425_223505`, which that same commit deleted from the repo. Recover it if you need it:
  `git archive 745b820 benchmark_results/all_benchmarks_no_unimol_20260425_223505 | tar -x -C <dir>`.
- Committed notebook outputs before 2026-09-24 came from another machine (`C:\Users\scott\QSARena`) and a mixed state.

## Don'ts (cost savers)

- Don't rerun benchmarks locally to "check" numbers: the canonical run recorded ≈155 h of wall-clock on a GPU box.
  Everything the manuscript needs is recomputable from committed `metrics.csv` / selector / runtime artifacts.
- Don't run the notebook with `benchmark_run_dir = "AUTO"` for manuscript work: it picks the most recently modified run
  directory, and the notebook itself writes into run directories.
- Don't commit without being asked; `publication_recommendations.md` may contain the user's uncommitted edits.
- Ignore `node_modules/`, `tmp/`, `catboost_info/`, `model_cache/`, `logs/`, `gin_supervised_masking_pre_trained.pth`.
