# AGENTS.md — AutoQSAR

Orientation for coding agents. Read this before exploring; most of it took a full session to discover.
Human-facing docs: `README.md` (usage), `CONTAINER.md` / `hpc/README.md` / `js2/README.md` (HPC),
`publication_recommendations.md` (reviewer-risk checklist). Keep this file current when you learn something
that would have saved you time.

## What this repo is

AutoQSAR: SMILES → molecular property QSAR/AutoML workspace plus a 45-dataset benchmark and a manuscript
(`manuscript.md`, target: *Journal of Cheminformatics*). Windows + OneDrive checkout (paths contain spaces:
always quote). Bash (Git Bash) and PowerShell are both available.

## Where to edit (source of truth)

| Want to change | Edit | Notes |
|---|---|---|
| Interactive Colab notebook | `portable_colab_qsar_bundle/build_colab_qsar_tutorial.py` | Regenerates `colab_qsar_tutorial.ipynb`; never hand-edit that notebook. |
| Benchmark runner | `portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py` | ~10k lines. `run_autoqsar_ga_benchmarks.txt` is a **stale mirror**: do not read or grep it as source. |
| Features, splits, CFA | `portable_colab_qsar_bundle/qsar_workflow_core.py` | (`qsar_workflow_core.txt` is a mirror.) |
| Dataset registry / catalog | `benchmark_registry.py`, `data/benchmark_dataset_catalog.csv` | |
| Leaderboard references | `data/benchmark_leaderboards/*.csv` | Current-literature ESOL/Lipophilicity refs live in `ESOL_Lipophilicity_Current_Benchmarks_csv.csv`. |
| Benchmark analysis, manuscript figures and tables | `portable_colab_qsar_bundle/benchmark_results_summary.ipynb` | Hand-maintained (no builder). The last cell (`# MANUSCRIPT_FIGURE_EXPORT`) writes `manuscript_assets/`. |
| Precision, Uni-Mol2, conformers (Update 2) | `autoqsar/` package, `run_one.py`, `js2/`, `hpc/` | |

## Manuscript workflow (the fast path)

1. Canonical run: **`benchmark_results/benchmark_name_date`**: the GPU run from commit `b7cd42c` (2026-05-11),
   `cost_optimized` profile, 45 datasets (23 regression / 22 classification), 25 models incl. Uni-Mol V1.
   The user chose this run (2026-09-24). Do not re-litigate it.
2. Regenerate every figure, table and number (≈1.5 min, no conda env needed; system Python has
   pandas/matplotlib/plotly/scipy/rdkit/nbclient):
   ```bash
   python portable_colab_qsar_bundle/render_manuscript_assets.py
   ```
   Outputs: `manuscript_assets/figures/*.png|pdf`, `manuscript_assets/tables/*.csv|md`,
   `manuscript_assets/manuscript_numbers.json`, plus `publication_*.csv` inside the run directory.
   It re-executes the notebook in place (so its saved outputs match the canonical run) and refreshes every
   `<!-- TABLE:<stem> -->…<!-- /TABLE -->` block in `manuscript.md` from `manuscript_assets/tables/<stem>.md`.
   **Never hand-edit text inside those blocks** — it is overwritten. Edit the export cell instead.
3. Check the prose against the artifacts:
   ```bash
   python portable_colab_qsar_bundle/verify_manuscript_numbers.py   # 69 assertions, exit 1 on drift
   ```
   **Every number in `manuscript.md` must trace to `manuscript_numbers.json` or a `manuscript_assets/tables/*` file.**
   After any rerun this script is the checklist: fix the prose to match the new artifacts, then update the expected
   values in the script. Never copy numbers from old notebook outputs or from `Manuscript Outline.md` /
   `publication_recommendations.md` (both predate the canonical run and the bug fixes below, and their headline
   counts — e.g. "MapLight+GNN mean gap 0.199", "selector slope 1.09", 44 datasets — come from the deleted April run).
4. Figures are static matplotlib written by the notebook's last cell (`# MANUSCRIPT_FIGURE_EXPORT`); Plotly
   `kaleido` is not installed, so don't build figures with Plotly for the manuscript.

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
- **Test-set leakage to disclose (not fixed in runner):** the per-dataset "best model" is chosen on the test set,
  and ensemble member filtering uses test metrics (`ensemble_exclude_negative_test_r2_members`, plus a test-metric
  tie-break when dropping correlated members, `run_autoqsar_ga_benchmarks.py` `build_ensemble_result`). The export
  cell reports a CV-selected sensitivity analysis (only conventional models, TabPFN and ChemML MLP have CV metrics).
- Leaderboard comparisons: use `UPDATED_LEADERBOARD_COMPARISON` (cell 21), not `leaderboard_eval` (cell 9 still scores
  ESOL/Lipophilicity against 2017 MoleculeNet baselines). Only the 22 TDC ADMET Benchmark Group datasets and 5 Polaris
  sets use official splits; tox21/toxcast are single-label subsets; carcinogens, skin_reaction, clintox, hydration-FreeSolv,
  PODUAM and MoleculeNet sets use local splits → "estimated rank", not leaderboard-equivalent.
- `tdc_ppbr_az` reference set contains a row on a different scale (top-1 MAE 0.679 vs top-10 cutoff 7.9); treat its top-1 gap as invalid.
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
- **Host provenance conflict — unresolved, ask before writing hardware claims.** The user states the benchmark ran on an
  NSF ACCESS Jetstream2 A100 (`g3.large`, allocation CIS261142) and that the artifacts merely look local because they
  were downloaded. The artifacts themselves say otherwise, and these strings are written *by the running process*:
  `NVIDIA GeForce RTX 4060 Laptop GPU` (3×), `C:\Users\scott\.conda\envs\autoqsar-py311\...` (761×, 44/45 datasets),
  Windows-only Chemprop exit code 3221226505, and 1,940 timestamps in Pacific Daylight Time rather than UTC. `js2/`
  was also committed 2026-06-18, five weeks after the run commit `b7cd42c` (2026-05-11). Section 2.12 of the manuscript
  carries an `[AUTHOR: hardware statement requires confirmation]` block describing the RTX 4060 workstation. If JS2
  artifacts surface, they must replace `benchmark_results/benchmark_name_date/` wholesale, since every number derives
  from it. The ACCESS/Jetstream2 acknowledgement and citations [41, 42] are in place regardless.
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
- Committed notebook outputs before 2026-09-24 came from another machine (`C:\Users\scott\AutoQSAR`) and a mixed state.

## Don'ts (cost savers)

- Don't rerun benchmarks locally to "check" numbers: the canonical run recorded ≈155 h of wall-clock on a GPU box.
  Everything the manuscript needs is recomputable from committed `metrics.csv` / selector / runtime artifacts.
- Don't run the notebook with `benchmark_run_dir = "AUTO"` for manuscript work: it picks the most recently modified run
  directory, and the notebook itself writes into run directories.
- Don't commit without being asked; `publication_recommendations.md` may contain the user's uncommitted edits.
- Ignore `node_modules/`, `tmp/`, `catboost_info/`, `model_cache/`, `logs/`, `gin_supervised_masking_pre_trained.pth`.
