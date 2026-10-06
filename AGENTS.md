# AGENTS.md — QSARena

Orientation for coding agents. Read this before exploring; most of it took a full session to discover.
Human-facing docs: `README.md` (usage), `CONTAINER.md` / `hpc/README.md` / `js2/README.md` (HPC),
`publication_recommendations.md` (reviewer-risk checklist). Keep this file current when you learn something
that would have saved you time.

## What this repo is

QSARena: SMILES → molecular property QSAR/AutoML workspace plus a 45-dataset benchmark and a manuscript
(`manuscript.md`, target: *Journal of Cheminformatics*). Windows + OneDrive checkout (paths contain spaces:
always quote). Bash (Git Bash) and PowerShell are both available.

## Current status and next steps (updated 2026-10-05) - read this first

**Done**
- **Manuscript run:** `benchmark_results/qsarena_benchmark_oof_ensemble`, 44/44 datasets, **31 models**. Ensembles
  are built from out-of-fold predictions (Chemprop OOF included, `--ensemble-oof-scope all`), with both 2026-09-29
  ensemble fixes. `python tools/verify_oof_ensemble_run.py` passes (its `INTENTIONAL_RETRAINS` lists the
  probability-fix retrains).
- **`XGBoost (ADMETboost features)` folded in** (2026-10-04, author-approved): base model on all 44 datasets with
  nothing else retrained; ensembles and CFA rebuilt once (outer selection); a member of 43 ensembles. Features come
  from the arm's cache (`featurize.find_cached_matrix`, matched by SMILES hash), so the benchmark env needs no
  skfp/mordred.
- **Hard-label bug fixed** (2026-10-04/05): binary datasets catalogued as `rmse` (cyp1a2, cyp2c19, herg_karim, pampa)
  were scored on `.predict()` labels. `predict_values_for_metric` now returns probabilities for classification tasks.
  10 models plus the new one were retrained on those 4 datasets. TabPFN (2026-10-05): pampa retrained with the
  **local** `tabpfn` package (v2.6 weights; `logs/run_tabpfn_proba_fix.ps1` clears the API-key variables; AUROC
  0.578 -> 0.715). On cyp1a2, cyp2c19 and herg_karim its rows are **withdrawn** (status
  `skipped_tabpfn_withdrawn_label_outputs`, metrics blanked): local TabPFN does not fit the 8 GB RTX 4060 at ~10k
  rows x ~1,000 features (OOM, then stalls even at 128-row chunks), and an API rerun is ~110M estimated tokens per
  dataset, not spent without the author. Backups: `benchmark_results/_proba_fix_backup_*`.
- **Manuscript regenerated from the nested-selection run** (2026-10-06): verifier 119/119, abstract at exactly
  **350 words (the cap)**, PDFs build with no errors or undefined references (TeX Live 2026 at
  `C:/texlive/2026/bin/windows`). The render needs the benchmark env
  (`C:/Users/scott/.conda/envs/autoqsar-py311/python.exe`), not system Python (no nbformat). Abstract and graphical
  abstract caption: edit `manuscript.md`, then `python submission/abstract_sync.py` (writes `abstract.tex` and the
  `manuscript.tex` caption, prints the word count). Nested selection is a Methods paragraph (META block
  `methods_nested_selection` in §2.6, from `cv_leak.py` + `selection_leak.csv`); the limitation is now "Residual
  cross-validation optimism".
- **Headline numbers (nested selection):** wins Ensemble 22 (14 reg / 8 cls), Conventional 6, Chemprop 5, Uni-Mol 5,
  TabPFN 3, MapLight+GNN 2, CFA 1; ensembles most consistent (84% within 5%); top-10 35 -> 28 -> 27 (7 + 1);
  test-selected firsts 6; CV pick winner on 4/44, median gap 7.7%; CV-vs-test overstatement 15.9% -> 3.4% (= the
  no-selection arm, 3.4%); hardware comparison now single models only (+0.50%, 22 vs 13). Under outer selection the
  ensembles won 12: the leaky OOF predictions over-weighted selected-feature members. Snapshot of the outer ensembles:
  `ensemble_outer_selection_snapshot.csv` in the run dir (verifier-checked). **The title ("Ensembles and Conventional
  ML Perform Comparably to Pretrained Models") may need revising now that ensembles win half: author item.**
- **Feature-selection CV leak confirmed causally** (`qsarena/feature_expansion/selection_leak.py`; positive in 26/27
  dataset-model pairs). `--cv-selection nested` is implemented (default for the `full` and `cost_optimized`
  profiles; see `docs/NESTED_SELECTION_CV_PLAN.md`). The notebook path has only been syntax-checked.

**Nested-selection run: COMPLETE (2026-10-06 08:52 PDT).** `tools/verify_oof_ensemble_run.py` passes (0 hard, 4 soft
warnings; it now also knows the 3 intentional TabPFN withdrawals). 44/44 datasets have both ensembles; no leaky CV row
remains. Median CV-vs-test overstatement over 344 model-dataset pairs: regression +11.6% -> -1.6%, classification AUROC
+14.1% -> +5.5% (lower in 95.3% of pairs). Repairs on 10-06: Tox21's ensembles were lost when an accidental full relaunch
was stopped mid-rebuild (rebuilt, `-Tag tox21`); tdc_herg and tdc_bioavailability_ma have no saved predictions for
their 12 conventional/ChemML models (pre-existing; never ensemble members), so the nested stage now gives such models
nested CV metrics only (`cv_selection_note`; `-Tag followup`). Rendered and written into the paper 2026-10-06.
Trap: restricted runs (`-Datasets`) rewrite the run-level `leaderboard_top10_reference.csv/.json` with only their
datasets (here: empty, which silently dropped Polaris from the leaderboard analysis) and `dataset_summary.csv`,
`run_config.*` and the report with only theirs. Restore the leaderboard cache from git and regenerate the summary
and report for all datasets before rendering.

**Nested-selection run details (author's go-ahead 2026-10-05)**
- Launcher `logs/run_nested_selection.ps1` (`-Datasets <names> -Tag pilot` for a pilot; no arguments = all 44),
  detached via WMI, log `logs/nested_selection_<tag>_20261005.log`, ends with `LAUNCHER_DONE`. Flags: the OOF run's
  plus `--cv-selection nested --run-admetboost-xgboost --only-model-names Ensemble --rebuild-ensemble
  --ensemble-exclude-model TabPFNRegressor --ensemble-exclude-model TabPFNClassifier`. No base model is retrained.
- TabPFN: the launcher clears the API-key variables, so TabPFN's nested fold refits run locally (no credits). TabPFN
  gets nested CV metrics but stays out of the ensembles, because most of its full-fit predictions came from the
  Prior Labs API, a different backend from the local fold refits. Above `--tabpfn-local-max-cells` (1.5M training
  rows x selected features: tox21, ames, ld50, aqsoldb and the 10k-row CYP/hERG sets) its fold refits are skipped
  and its CV metrics are **withdrawn** (`cv_selection=outer_withdrawn`, old value in `cv_primary_outer`), so it is
  not CV-eligible there. Local TabPFN predicts through `ChunkedTabPFN*` (adaptive chunks, a content-keyed
  prediction cache because the five CV scorers each call predict, and a 0.9 GPU-memory cap so the Windows driver
  cannot silently spill VRAM into system RAM, which made it crawl).
- Pilot (tdc_caco2_wang, 18 min) passed: test metrics unchanged, 15 members nested, outer CV overstatement of
  +3% to +28% became -15% to +1%, ensembles rebuilt (22 members, TabPFN excluded). Full run launched 03:16 PDT
  2026-10-05. **PAUSED by the author at 20:00 PDT 2026-10-05 with 39/44 done** (stopped during `tdc_cyp3a4_veith`;
  CYP1A2, CYP2C19, CYP2D6, hERG-Karim and CYP3A4 remain). **Resumed 22:50 PDT** on only those five
  (`-Datasets tdc_cyp3a4_veith,... -Tag rest`, log `logs/nested_selection_rest_20261005.log`; part-1 log saved as
  `nested_selection_full_20261005_part1.log`). Don't resume with no `-Datasets`: `--rebuild-ensemble` re-passes every
  finished dataset (cheap, cached) but on Tox21 it retrains the two Chemprop variants whose fold refits always drop
  one row (~10 GPU fold trainings, same failed result).
- Checks: no `config signature changed for` except the ensemble family; base rows keep `primary_metric_value`
  and test metrics unchanged; selected-feature members get `cv_selection=nested` and keep `cv_primary_outer`;
  ensembles rebuilt on all 44.

**Next steps**
1. When the nested run finishes: run the checks above and `tools/verify_oof_ensemble_run.py`, then render,
   update the verifier's expected values (never loosen a check), and update the paper: §2.13 (nested CV; remove the
   "TabPFN ... still carries label outputs" sentence), the CV-leak caveat becomes a Methods statement (the
   controlled test is its evidence), and every CV-dependent number (CV pick, gap, decomposition's CV leg, wins).
2. CFA ranks fusion candidates by in-sample training error (related to, but not, the selection leak; TODO.md).
3. Author items: title (options in TODO.md), the `[AUTHOR]` flags (other LLM tools in §2.14, OEHHA disclaimer,
   reviewers, preprint), and the Zenodo deposit (rename the repo first; bundles in `dist/zenodo/`).
4. Anything from the feature-expansion arm that enters the paper is supplementary. New numbers go through the
   render and the verifier, never hand-typed.
5. Optional, each needing approval: Phase 7 learning curve (~18 GPU-h, `docs/HANDOFF_RTX_GPU_WORK.md`) and a 5-seed
   TDC-22 replication. **Declined:** the full CheMeleon run (~116 GPU-h; don't propose it again without new evidence).

**CI** (`.github/workflows/ci.yml`): ruff on `qsarena tests`, then `pytest -m "not gpu and not slow"` on
Python 3.10-3.12 with only `.[dev]` installed and the newest pandas, plus a LaTeX build of `proof.tex`.
Job logs need admin rights; to reproduce a failure locally, build a clean venv at a SHORT path (RDKit DLLs fail
on long Windows paths), e.g. `py -3.13 -m venv C:/Users/scott/qci && C:/Users/scott/qci/Scripts/python -m pip
install -e .[dev]`, then run the same pytest command.

**Running long jobs on the RTX box.** Launch them detached through WMI: write a `.cmd` under `logs/`, then run
`Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c <path>'}`.
The agent session kills its own background tasks after 30 min. Windows EcoQoS throttles WMI-launched processes
(Chemprop child start-up went from 7 s to ~19 s, and conformer generation slowed ~5x). The fix is
`powercfg /overlaysetactive ded574b5-45a0-4f42-8737-46345c09c238`, which resets on reboot.

## A100 task (COMPLETED; historical): regenerate the clean (out-of-fold) ensemble results

**Done.** Steps 4-7 were completed on the RTX 4060 workstation, including the Chemprop OOF that
the 2026-09-26 status below calls "out of reach" (that judgment applied to the A100 allocation only). Keep
this section as the recipe for any future repair run.

**Read this whole section before running anything on the A100.** The goal is
`benchmark_results/qsarena_benchmark_oof_ensemble`: the same base models as
`qsarena_benchmark_chemprop_fixed`, with ensembles rebuilt from out-of-fold (OOF) member
predictions. Background: "Run lineage" and the ensemble trap below, and
[submission/chemprop_rerun_command.md](submission/chemprop_rerun_command.md) §7.

**Hard rules**
- **No full model is ever retrained.** Every base model's train/test predictions already exist in
  the seeded `predictions.csv`. The only training allowed is *fold* models for OOF predictions, and
  in step 5 only CPU ones.
- **No GPU fold refits (Chemprop, `--ensemble-oof-scope all`) without the user's explicit approval**
  in the current conversation. Report the planner's GPU estimate and stop.
- **Never enable `--ensemble-oof-allow-api-refits`.** TabPFN runs on a capped Prior Labs credit
  budget and is left out of the ensembles by default.
- **Never write into `autoqsar_benchmark_20260623_153839` or `qsarena_benchmark_chemprop_fixed`.**
  They are read-only inputs. Everything goes to `qsarena_benchmark_oof_ensemble`.
- **Jetstream2 bills instance uptime, not GPU use.** Don't leave the instance idle between steps.
  Shelving it when work is done is the user's decision: tell them when the run finishes.
- **If any check below fails, stop and report. Do not work around it.**

**Step 0 — code and tests (CPU, ~1 min).**
```bash
git pull origin main
python -m pytest -q tests/unit/test_ensemble_oof.py tests/unit/test_plan_ensemble_oof_repair.py
```
All must pass. `portable_colab_qsar_bundle/plan_ensemble_oof_repair.py` must exist.

**Step 1 — inputs exist (no compute).**
```bash
ls benchmark_results/qsarena_benchmark_chemprop_fixed/*/predictions.csv | wc -l           # expect 44
find benchmark_results/autoqsar_benchmark_20260623_153839 -name cv.data | wc -l          # Uni-Mol OOF files; expect ~61 (44 V1 + 17 V2)
```
If `predictions.csv` is missing, stop: the ensemble cannot be rebuilt without retraining.

**Step 2 — seed the new run (file copy only).**
```bash
python portable_colab_qsar_bundle/prepare_chemprop_repair_run.py \
  benchmark_results/qsarena_benchmark_chemprop_fixed \
  benchmark_results/qsarena_benchmark_oof_ensemble
```

**Step 3 — plan without training (the check that nothing unnecessary is refitted).**
```bash
python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py \
  benchmark_results/qsarena_benchmark_oof_ensemble --scope cpu --folds 5 \
  --source-run benchmark_results/autoqsar_benchmark_20260623_153839
python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py \
  benchmark_results/qsarena_benchmark_oof_ensemble --scope all --folds 5 \
  --source-run benchmark_results/autoqsar_benchmark_20260623_153839   # report only; do NOT run 'all'
```
It writes `ensemble_oof_plan.csv`. Pass criteria for the `cpu` plan:
- exit code 0, and no `missing_inputs` rows;
- zero `gpu_refit` rows;
- every Uni-Mol member has source `unimol_cvdata`. Its OOF predictions are read from disk and it is
  never refitted. List any Uni-Mol member that is `excluded` for the user.
- `cpu_refit` covers only conventional, ChemML and MapLight members. Chemprop, TabPFN and CFA are
  `excluded`.

Re-running the planner after step 5 must show the `cpu_refit` members as `saved_oof`. That confirms
nothing will be refitted twice.

**Step 4 — pilot on the smallest dataset (CPU scope) before the full run.** Take the dataset with
the smallest `n_train` from `ensemble_oof_plan.csv`. Copy its `metrics.csv` first
(`cp .../<ds>/metrics.csv /tmp/<ds>_metrics_before.csv`). Then run the step 5 command with a single
`--dataset-name <ds>`, logging to `logs/oof_pilot.log`. Pass criteria, all required:
```bash
L=logs/oof_pilot.log
# (a) nothing retrained: every model/feature stage must say "(cached)"
grep -E "stage [0-9]+/[0-9]+: (building molecular features|splitting data|conventional model|deep model|GA tuning|CFA)" $L | grep -v "(cached)"   # must print nothing
grep -c "config signature changed" $L          # must be 0
# (b) Uni-Mol read from disk; no GPU fold refits
grep "ensemble-oof" $L | grep -i "uni-mol"      # must say "read saved Uni-Mol internal-fold predictions"
grep "ensemble-oof" $L | grep -iE "(chemprop|uni-mol).*fold [0-9]"   # must print nothing
```
(c) The base-model rows in `metrics.csv` are unchanged. `primary_metric_value` must be identical
for every non-ensemble model in `/tmp/<ds>_metrics_before.csv` and the new file.
(d) The new ensemble rows have `ensemble_member_selection_split == oof`, and
`ensemble_member_filter_notes` mentions no unexpected `refit failed`.

Record the pilot's wall-clock and extrapolate to 44 datasets. Tell the user before starting step 5
if the estimate is far above ~10 h.

**Step 5 — full run, CPU scope.**
```bash
dataset_args=()
for m in benchmark_results/qsarena_benchmark_oof_ensemble/*/metrics.csv; do
  dataset_args+=(--dataset-name "$(basename "$(dirname "$m")")"); done
nohup python portable_colab_qsar_bundle/run_qsarena_benchmarks.py \
  --output-dir benchmark_results/qsarena_benchmark_oof_ensemble --benchmark-profile full \
  "${dataset_args[@]}" --only-model-names 'Ensemble' --run-ensemble --rebuild-ensemble \
  --ensemble-member-selection-split oof --ensemble-oof-folds 5 --ensemble-oof-scope cpu \
  --ensemble-oof-source-run benchmark_results/autoqsar_benchmark_20260623_153839 \
  --run-tabpfn --tabpfn-max-train-rows 11000 --unimol-batch-size 32 --unimol-max-atoms 64 \
  --run-chemprop-mpnn --run-chemprop-dmpnn --run-chemprop-rdkit2d --run-chemprop-cmpnn \
  --run-chemprop-attentivefp --run-chemprop-selected-features \
  --chemprop-epochs 40 --chemprop-ensemble-size 3 --chemprop-random-seed 42 \
  --reuse-persistent-feature-store --reuse-shared-feature-matrix-cache \
  --resume --no-run-tdc22-multiseed-best > logs/oof_ensemble.log 2>&1 &
```
`--only-model-names 'Ensemble'` is what prevents full-model retraining: every other model stage is
filtered out. The `--run-tabpfn` / `--run-chemprop-*` / Uni-Mol flags only tell the runner which
members exist and which settings their fold refits would use. Don't drop them, and don't add a
model name to `--only-model-names`. If the run is interrupted, re-run the identical command: fold
results are cached under `<dataset>/ensemble_oof/` and completed OOF vectors are saved in
`predictions.csv`. Re-apply check (a) to `logs/oof_ensemble.log` periodically.

**Step 6 — Chemprop only if the user approves.** Re-run the step 5 command with
`--ensemble-oof-scope all`. The CPU fold results are reused; only the Chemprop fold trainings run
(the planner's `--scope all` estimate is roughly 65 h). Otherwise the ensembles exclude Chemprop,
and the paper must say so (TODO.md).

**RTX handoff path for approved Chemprop OOF.** If the user wants Chemprop fold OOF on a separate
RTX workstation, do not push the full ignored `predictions.csv` corpus. Use
`docs/HANDOFF_RTX_GPU_WORK.md` and `docs/HANDOFF_RTX_CHEMPROP_OOF.md`: this branch carries
`chemprop_oof_seed/` with only Chemprop train/test seed rows, `tools/run_chemprop_oof_rtx.ps1`
for the GPU box, and importer/exporter helpers for the small returned `chemprop_oof_patch/`.
The meta-analysis implementation spec is CPU-only for phases 1-6; its only GPU item is the
optional, approval-gated Phase 7 learning-curve experiment. If approved, run
`tools/run_meta_phase7_rtx.ps1` and push only `meta_phase7_gpu_patch/`, not
`benchmark_results/qsarena_meta_phase7_gpu/`.

**Step 7 — verify and commit the run.** Run the checks in `chemprop_rerun_command.md` §7 ("Checks
after the run"). Then commit `benchmark_results/qsarena_benchmark_oof_ensemble` (predictions,
`ensemble_oof/` fold caches and model files are gitignored), with the scope used and the planner
summary in the commit message, and push to `main`. Don't regenerate the manuscript on the A100:
that runs on the workstation (`render_manuscript_assets.py --run-dir
benchmark_results/qsarena_benchmark_oof_ensemble`), and it costs no allocation.


### Status as of 2026-09-26 (A100 session ended here)

**Steps 0-3 are done; steps 4-6 were never run.** The instance was shelved before the pilot, so
`qsarena_benchmark_oof_ensemble` contains seeded inputs and `ensemble_oof_plan.csv` only -- no
rebuilt ensembles. Resume at **step 4**.

Planner output, committed as `benchmark_results/qsarena_benchmark_oof_ensemble/ensemble_oof_plan.csv`
(1021 member rows, 44 datasets, folds=5). All `--scope cpu` pass criteria passed: exit 0, zero
`missing_inputs`, zero `gpu_refit`, all 61 Uni-Mol members resolve to `unimol_cvdata` (none
excluded), and `cpu_refit` is only conventional/ChemML/MapLight.

| source | kind | members | fold trainings |
|---|---|---|---|
| `cpu_refit` | conventional 462, chemml 84, maplight_gnn 43 | 589 | 2945 |
| `unimol_cvdata` | unimol 61 | 61 | 0 (read from `cv.data`) |
| `excluded` | chemprop 202, fusion 131, tabpfn 38 | 371 | 0 |

Under `--scope all`, chemprop moves to `gpu_refit`: 1010 GPU fold trainings, **~63 h**.
**That is now out of reach and the question is closed.** The allocation had 1,092 SUs left at
64 SUs/hr (17.1 h); 63 h would have cost 4,032 SUs. CPU is not a substitute either: measured on
32 cores, 523 rows / 40 epochs / ensemble=3 took 344 s, extrapolating to ~480 h on 32 cores
(~7.6x the GPU figure). Shrinking folds/epochs/ensemble-size would bias Chemprop's ensemble
weight downward by making fold models weaker than the deployed ones. Chemprop stays excluded.

**Chemprop cannot be shortcut -- it is refit-or-exclude.** Its only saved artifacts are full-fit
`train_predictions.csv` / `test_predictions.csv`. `splits.json` is a list of length **1**, so all
three ensemble members share one 90/10 split: ~10% of training rows have a held-out prediction and
the other 90% of `train_predictions.csv` is in-sample -- the exact leakage the OOF rebuild removes.
10% coverage cannot fit stacking weights, so there is no cheap Chemprop OOF path.

**What excluding Chemprop costs.** In the old (leaky) ensembles Chemprop was 334 member selections
across 42/44 datasets and 10.7% of total |weight| (vs other 77.4%, Uni-Mol 8.4%, TabPFN 3.5%).
Treat 10.7% as an **upper bound**: the weighted-average method weights by inverse *train* RMSE, and
Chemprop's train predictions are 90% in-sample, so its train RMSE is artificially low and its
weight inflated. Its honest OOF weight is lower.

**Moving to the RTX (2026-09-28): follow [docs/HANDOFF_SSD_TRANSFER.md](docs/HANDOFF_SSD_TRANSFER.md)**;
the required set is now `qsarena_benchmark_oof_ensemble/**/predictions.csv` (~1.1 GB, verified
against `artifact_manifest.csv` by `tools/verify_transferred_predictions.py`).
**Continuing on another machine needs ~0.57 GB of gitignored files**:
`benchmark_results/qsarena_benchmark_chemprop_fixed/*/predictions.csv` (45 files). Without them
the ensembles cannot be rebuilt at all and `prepare_chemprop_repair_run.py` cannot reseed.
The Uni-Mol `cv.data` files (61, 1.25 MB) are **now committed** - they were un-gitignored for this
handoff, so they arrive with a clone. See [docs/HANDOFF_local_cpu_oof.md](docs/HANDOFF_local_cpu_oof.md). The seeded run's 6.11 GB of `stage23_resume_cache.pkl` is regenerable and need not
be copied (omitting it only means features are rebuilt on CPU).

**GPU driver mismatch on this host (not a results problem).** Unattended-upgrades replaced the
NVIDIA driver at 06:59 UTC *during* the 06:23-13:55 benchmark run: kernel module 595.71.05 stayed
resident while userspace moved to 595.84, so `nvidia-smi` fails with
`Driver/library version mismatch`. Already-running processes were unaffected, so that run's results
are sound. Fix is a module reload, **not** a reboot:
`sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia && sudo modprobe nvidia`
(no process held `/dev/nvidia*`; `nvidia_drm` had a refcount of 1, which may need the display
stack stopped first). Only matters for step 6.

## Where to edit (source of truth)

| Want to change | Edit | Notes |
|---|---|---|
| Interactive notebooks | `portable_colab_qsar_bundle/build_colab_qsar_tutorial.py` | Regenerates both `colab_qsar_tutorial.ipynb` (Colab `# @param` controls) and `local_qsar_tutorial.ipynb` (`ipywidgets` controls); never hand-edit generated notebooks. |
| Benchmark runner | `portable_colab_qsar_bundle/run_qsarena_benchmarks.py` | ~10k lines. (The stale `.txt` mirrors were deleted 2026-10-02.) |
| Features, splits, CFA, shared ensemble logic | `portable_colab_qsar_bundle/qsar_workflow_core.py` | Runner orchestration stays in `run_qsarena_benchmarks.py`; notebook member collection stays in builder block 7A. |
| Dataset registry / catalog | `benchmark_registry.py`, `data/benchmark_dataset_catalog.csv` | |
| Leaderboard references | `data/benchmark_leaderboards/*.csv` | Current-literature ESOL/Lipophilicity refs live in `ESOL_Lipophilicity_Current_Benchmarks_csv.csv`. |
| Benchmark analysis, manuscript figures and tables | `portable_colab_qsar_bundle/benchmark_results_summary.ipynb` | Hand-maintained (no builder). The last cell (`# MANUSCRIPT_FIGURE_EXPORT`) writes `manuscript_assets/`. |
| OECD reliability audit / Table S8 | `qsarena/reliability_study.py`, `qsarena/reliability_tables.py` | Writes `results/reliability_tdc22/`, `submission/tables/tableS8_applicability_domain.tex`; not regenerated by `render_manuscript_assets.py`. |
| Graphical abstract | `portable_colab_qsar_bundle/render_graphical_abstract.py` | Writes `manuscript_assets/figures/graphical_abstract.svg`. It should summarize the paper's decision story, not duplicate Figure 1. |
| Precision, Uni-Mol2, conformers (Update 2) | `qsarena/` package, `run_one.py`, `js2/`, `hpc/` | |
| pip packaging (deps, extras, console scripts) | `pyproject.toml` | See "Packaging" below. |
| Any user-facing run option (the 15 decision groups) | `qsarena/config.py` (`RunConfig`) | Then `python -m qsarena.config --write-docs` (regenerates `docs/options_reference.md`, `configs/run.example.yaml`, the tutorial's option tables). Tests fail if they are stale. |
| Batch mode, preflight/dry-run, reports, run.log/events, atomic writes | `qsarena/batch.py`, `preflight.py`, `reporting.py`, `run_events.py`, `artifacts.py` | Wired into the runner's `prepare_args()` / `_run_main()`. |
| Tutorial = Additional file 2 | `docs/tutorial.md` | Every command runs in `tests/docs` (~6 min). PDF: `python submission/build_additional_file_2.py`; never edit the generated `.tex`. Work order: `docs/AGENT_WORK_ORDER_tutorial.md`. |
| Dataset-property meta-analysis (§3.14, Figs 7-9, Tables S9-S10) | `qsarena/meta_analysis/` | Spec `docs/QSARena_spec.md`; methods `docs/meta_analysis/METHODS.md`; inventory `docs/meta_analysis/INVENTORY.md`. Run by `render_manuscript_assets.py` after the notebook (~4 min CPU; `--no-meta-analysis` skips it). Reads the notebook's Fig 6 export (`manuscript_assets/tables/figure6_family_best_models*.csv`) and the committed partitions `data/meta_analysis/dataset_partitions.csv.gz`, which are hash-verified against `split_*_hash`. §3.14 prose lives between `META` markers in `manuscript.md` / `body.tex`, rendered from `meta_numbers.json`; never hand-edit it. Edit the templates in `text.py`. Blocks: `section_3_14`, `limitation_meta`, and `limitation_cvleak` (the selection-leak caveat; numbers from `cv_leak.py`, which measures CV-vs-test overstatement from `metrics.csv` with the feature-expansion arm as the reference). Selector v2: `selection.py` / `selector_features.py`, pre-registered in `docs/meta_analysis/SELECTOR_V2_PLAN.md`; the nested permutation test is cached in `manuscript_assets/tables/meta_selector_v2_permutation.json` by input hash (~1 h when recomputed). Tests: `tests/meta` (the pipeline tests are `slow`). |
| Feature-expansion arm (ADMETboost features, pretrained embeddings, full-feature trees) | `qsarena/feature_expansion/` | **Read `docs/FEATURE_EXPANSION_PLAN.md` first; its checklist is the status tracker, so tick items as you go.** Experimental and separate from the canonical run: it uses the committed partitions and writes `benchmark_results/qsarena_feature_expansion/<feature_set>/metrics.csv`. Feature cache: `.model_cache/feature_expansion/<family>/<dataset>.npy` (gitignored, hash-checked, resumable). CPU featurization runs in **system Python** (scikit-fingerprints, mordredcommunity and gensim are installed there, not in the benchmark env). Mol2Vec model: `.model_cache/mol2vec/model_300dim.pkl` (download URL and sha256 in `featurize.py`). Motivation: honest CV-selected QSARena loses 3-19 to MaxQsaring on the 22 official TDC splits, and ADMETboost, the NIST meta-model and MaxQsaring all feed Mordred-rich, unselected features to XGBoost. Tests: `tests/feature_expansion`. |
| Tutorial example data | `tests/fixtures/tutorial/make_tutorial_data.py` -> `qsarena/examples/data/` | Synthetic targets; shipped in the wheel; `qsarena-examples DIR` copies them out. |

Notebook generation notes:
- Keep the two generated notebooks interface-clean: Colab should not show local-widget control cells, and the local notebook should not expose Colab `# @param` controls, Google Drive setup widgets, or "Upload CSV/XLSX (Colab only)" choices.
- The Colab intro should explain that Colab runtimes are temporary, Drive persistence/downloads are needed for durable outputs, and step `0` must be rerun after Colab intentionally disconnects/restarts to load newly installed compiled packages.
- Keep notebook block `9F` as the self-contained HTML report export. It should work in both generated notebooks, gather whatever dataset/model/prediction/applicability-domain state has already been produced, and avoid mixing Colab-only controls into the local interface or local-widget controls into Colab.
- If setup helper stages are added or removed in `build_colab_qsar_tutorial.py`, update `SETUP_PROGRESS["total"]` before regenerating; the setup log should end with matching counts such as `32/32`.

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

   **Run lineage since 2026-09-26: the manuscript is moving off the canonical run.** Repair runs are
   seeded from the previous run's metrics + predictions (`prepare_chemprop_repair_run.py`) and only
   train what is missing:
   - `benchmark_results/qsarena_benchmark_chemprop_fixed` (A100, committed `ab8d52f`): adds working
     Chemprop (valid on 38-42/44 per variant), TabPFN (31/44; the API daily limit hit the rest) and a
     clean `polaris_adme_fang_hppb_1`; all 44 complete. **Its ensembles are flawed. Do not report
     them.** It used `--ensemble-member-selection-split train`, which rewards memorisation (see the
     trap below). Its selector times and wall-clock totals come from reusing cached features and
     selections, so they are not real costs either. Take selector scaling and dataset wall-clock from
     the canonical run.
   - `benchmark_results/qsarena_benchmark_oof_ensemble`: **the manuscript's run since 2026-10-02.**
     Same base models, ensembles rebuilt on out-of-fold predictions with Chemprop OOF included
     (`--ensemble-oof-scope all`, run on the RTX 4060), no full model retrained, and both
     2026-09-29 ensemble fixes applied. Post-run checks: `python tools/verify_oof_ensemble_run.py`.
   - Regenerate with `render_manuscript_assets.py --run-dir benchmark_results/qsarena_benchmark_oof_ensemble`.
     `verify_manuscript_numbers.py` is pinned to this run (119 checks, including the §2.13 run-history numbers,
     which it recomputes from the committed metrics of all three runs, and the §3.12 post-hoc arm numbers). If the run changes, update
     each expected value; never loosen a check.
2. Regenerate every figure, table and number (~1.5 min; system Python suffices):
   ```bash
   python portable_colab_qsar_bundle/render_manuscript_assets.py   # notebook + figures + tables + numbers JSON + LaTeX tables
   python portable_colab_qsar_bundle/render_graphical_abstract.py  # 920x300 J.Cheminform graphical abstract (SVG+PNG+PDF)
   python portable_colab_qsar_bundle/verify_manuscript_numbers.py  # 119 checks; non-zero exit on drift
   ```
   The first also rewrites the `<!-- TABLE:stem -->` blocks in `manuscript.md` and
   `submission/tables/*.tex`. **Never hand-edit inside those blocks or those .tex files.**
3. **Two manuscript formats must stay in sync**: `manuscript.md` (working doc) and
   `submission/body.tex` (the submission). Prose edits must be made in both. The verifier now checks
   many numbers in both formats, but it is not a complete sync checker — grep `body.tex` after any
   numeric prose edit.
4. Numbers come only from `manuscript_assets/manuscript_numbers.json` or
   `manuscript_assets/tables/*.csv`. Never from old notebook outputs, `Manuscript Outline.md` or
   `publication_recommendations.md` (all predate the A100 run).
   Update after the OECD reliability work: `verify_manuscript_numbers.py` now also checks the 3.13
   reliability numbers in `submission/body.tex`, and the verifier currently runs 119 checks. Still
   grep both manuscript formats after other numeric prose edits.
5. Table S8 / 3.13 comes from the separate reliability study, not the manuscript-assets notebook:
   ```bash
   python -m qsarena.reliability_study --out results/reliability_tdc22 --resummarize
   python -m qsarena.reliability_tables
   ```
   `results/reliability_tdc22/summary.json` is the only source for 3.13 OECD reliability numbers.
   It describes a fixed reference random forest on the 22 official TDC ADMET splits, **not** the
   benchmark-selected model, and it is single-seed only. Table S8 is included by
   `submission/additional_file_1.tex` on a landscape page.

## Paper's framing (do not weaken these without evidence)

The paper makes three load-bearing claims. Keep them straight when editing:
1. **Leaderboard placement**: top-10 on 35/37 (22/22 on official TDC splits), median rank 3.
   **Since the focused review of 2026-09-25 this is a provisional, secondary result**: the
   reference set contains leaked and self-reported entries (see traps below). The abstract,
   conclusions and graphical abstract lead with claim 2 and with the internal CV-vs-test gap
   (nested-selection run: CV selection picks the winner on 4/44, median relative gap 7.7%). Don't move the ranks back
   into the headline. The novelty claim is the uniform cross-suite benchmark plus the
   decomposition; code-free access is described as a usability feature (OCHEM/ChemSAR already
   offer it).
2. **The 35/37 → 27/37 drop is real but has TWO causes, and the paper decomposes them.** A
   matched-candidate-set control (test-selected, restricted to the CV-eligible pool) gives 28/37,
   3 firsts, median rank 6. In the OOF run **7 of the 8 lost placements are the value of the broad
   model library (35→28) and 1 is the cost of honest selection (28→27)**; median rank 3 → 6 → 7;
   first places 5 → 3 → 1. (The canonical run was 35 → 28 → 25 with first places 5 → 3 → 0; TabPFN
   becoming CV-eligible is what changed it.) Never re-attribute the whole gap to selection — that was
   the pre-2026-09-25 error. Never drop the decomposition to make the headline look better either.
   Verified by `chk("gap decomposition 7+1", ...)`.
3. **No effort and no hardware required**: all 44 datasets ran under ONE fixed configuration with no
   per-dataset tuning and GA disabled, and the notebook runs code-free in Colab with no install.
   §3.12 states this and its limits. It is evidenced by the run design, not a marketing line.

Open work is tracked in [TODO.md](TODO.md); the Zenodo deposit is the last blocking submission item.

## Journal requirements already encoded (J. Cheminform., **Research article** since 2026-10-02)

- Abstract is capped at **350 words** and must contain a **Scientific Contribution** section
  (max 3 sentences). Both are in place, and the abstract is at exactly 350 words, so re-check the count after any
  abstract edit. Abstract headings: Background / Methods / Results / Conclusions / Scientific Contribution.
- Graphical abstract spec: **920x300 px, <=150 KB, white background**. `render_graphical_abstract.py`
  emits exactly that and reads its statistics from `manuscript_numbers.json`.
- Structure (Research article, from the journal's submission guidelines): Introduction / Methods / Results and
  discussion / Conclusions / Abbreviations / Declarations. The section labels kept their old LaTeX keys
  (`sec:background`, `sec:implementation`) so cross-references still resolve. The separate `Availability and
  requirements` section is a Software-article requirement: its fields now open the `Availability of data and
  materials` declaration as "Software availability".
- **LLM use must be documented in the Methods** (journal: "Use of an LLM should be properly documented in the
  Methods section"; Springer Nature: LLMs cannot be authors, AI-generated images that are not derived from
  verifiable data are not permitted). It lives in §2.14 "Use of large language models" in both formats, not in
  the Declarations. Keep it accurate if the tools change.
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
- **Test-set leakage:** the per-dataset "best model" is still chosen on the test set (disclose this).
  `--ensemble-member-selection-split` decides what ensembles are selected, weighted and stacked on.
  There are three modes, and only one is correct:
  - `test`: the canonical run. It leaks held-out R² into member selection. A `run_config.json` with no
    `ensemble_member_selection_split` key means this mode.
  - `train`: the chemprop_fixed run. It doesn't leak, but it reads **in-sample** training predictions.
    A 500-tree extra-trees model has a training RMSE near 0, so it wins the correlated-pair tie-break
    and takes ~100% of the stacking weight. On ESOL it pushed MapLight CatBoost out of the ensemble.
    Across that run ensemble regression wins fell from 7 to 1. That was the artefact, not a finding.
  - `oof` (the default): uses only out-of-fold predictions, taken first from what a backend already
    saved and otherwise from refitting on K folds of the training split (same folds as `--cv-folds`).
    Uni-Mol's saved `cv.data` holds its internal 5-fold predictions: they are on the normalised
    target scale, so regression values are inverse-transformed with `target_scaler.ss`. Conventional
    models reuse their CV fold fits in fresh runs. Chemprop (only with `--ensemble-oof-scope all`),
    MapLight+GNN and ChemML are refitted per fold into `<dataset>/ensemble_oof/`. Saved model
    weights do **not** remove the need for this: they give in-sample predictions for training
    molecules. The OOF predictions are saved as `split="oof"` rows in
    `predictions.csv`, so the stage resumes. Members without OOF predictions, and CFA, are excluded
    and noted in `ensemble_member_filter_notes`. Tests: `tests/unit/test_ensemble_oof.py`.
  Downstream code must not assume `predictions.csv` holds only `train`/`test` rows.
  The command-line runner and Colab notebook now call shared ensemble helpers in
  `qsar_workflow_core.py`. Notebook block 7A still owns member collection: conventional and tuned
  models are refitted on K folds inside the cell (cached in `STATE["ensemble_oof_cache"]`), Uni-Mol
  comes from `cv.data`, and other deep workflows are not members there. It must never go back to
  choosing members, filters, CFA inputs or the downstream strategy by *test* metrics, all four of
  which it used to do.
- **Leakage-free CV selection is no longer model-starved.** TabPFN emits CV metrics. In the
  chemprop_fixed run, the CV-selected model picked the per-dataset winner on 3 datasets (it was 0),
  sat a median 8.7% from the best (it was 16.1%), and reached **one** estimated first place (it was
  0). In the nested-selection run: winner on 4/44, median gap 7.7%, one first place, top-10 35 → 28 → 27. The
  "every first place" and "CV never picks the winner" wording is retired; don't reintroduce it.
- **Two OOF-ensemble bugs found 2026-09-29, fixed in `qsar_workflow_core.py`; every ensemble built
  before the fix needs rebuilding (folds are cached, so the rebuild is CPU-only). The OOF run was rebuilt
  with both fixes on 2026-10-01/02.**
  (1) `target_scaler.ss` files are gitignored (`**/*.ss`), so after the SSD/git transfer Uni-Mol's
  regression `cv.data` was used on its *normalised* scale (OOF mean ~0 vs target mean -4). Those
  members then failed the non-positive-R² filter, so Uni-Mol silently dropped out of nearly every
  regression ensemble. `load_unimol_saved_oof` now rebuilds the scaler exactly from the training
  targets (unimol_tools rule: log1p if |skew|>5 or |kurtosis|>20, else StandardScaler) and rejects
  any vector still off the target scale. Provider-backed OOF (Uni-Mol) is now re-read every run
  instead of trusting saved `split="oof"` rows. (2) Members can extrapolate wildly on a few
  molecules: Chemprop RDKit2D/AttentiveFP test predictions reached -1803 on PODUAM (target -11..-1),
  and tabular-NN fold models reached 42,228 on LD50. One such value wrecked whole ensembles
  (PODUAM NC ensemble RMSE 4.1 vs 0.72 for the best member). `build_ensemble` now clips member
  predictions to the training target range (regression; label-free) and lists clipped members in
  the notes. Base-model metrics are unaffected. Tests: `tests/unit/test_ensemble_oof.py`,
  `tests/unit/test_core_ensemble.py`. The pandas 3 `MergeError` in `_ensemble_split_frame` (SMILES
  alignment created a second `row_id__new_obs` column on the third merge) was fixed 2026-10-02: the right-hand
  frame's non-key `row_id`/`SMILES`/`Observed` columns are dropped before merging. It had kept GitHub CI red since
  2026-09-30, because CI installs the latest pandas.
- **The benchmark's model CV leaks through feature selection (found 2026-10-02).** Stage 3 fits the
  ElasticNetCV selector once on ALL training rows, then each model's `cross_validate` (runner ~L5605) runs
  on `X_train_selected`, so every held-out CV fold influenced which features were kept. On predefined/
  scaffold test splits the median CV-vs-test overstatement is 11-44% for benchmark models (ElasticNetCV
  +44%, TabPFNRegressor +27%, SVR +25%, MLPs +23-25%, trees +12-15%), versus ~4% for the
  feature-expansion arm's XGBoost/RF, which use no selection, on the SAME random folds. So the inflation
  comes from selection leakage, not fold geometry (scaffold CV only moved the arm's overstatement from
  ~4% to ~0%). Consequences: the CV-selected ("honest") pick and the CV-vs-test gap in the manuscript
  rest on leaky CV scores, and conventional members' OOF predictions (same folds) carry some of the
  leak into ensemble weighting. Test-set and leaderboard numbers are unaffected. The fix would be to
  refit selection inside each CV fold (costly: ElasticNet times out on large sets). **Causally confirmed on 2026-10-03**
  (`qsarena/feature_expansion/selection_leak.py`: the same features, models and folds, with selection fitted outside
  vs inside the folds; on 9 datasets, the leak is 22.7 points of CV overstatement for ElasticNetCV (9/9 datasets),
  6.1 for random forest (9/9) and 2.5 for SVR (8/9); positive in 26 of 27 pairs, p < 0.001).
  The fix, `--cv-selection nested`, is implemented and its run on the manuscript run started 2026-10-05 (see the status section and `docs/NESTED_SELECTION_CV_PLAN.md`). Data:
  `benchmark_results/qsarena_feature_expansion/admetboost__scaffoldcv/cv_geometry_*.csv`; benchmark
  optimism recomputable from `metrics.csv` (`cv_primary` vs `test_<primary>`).
- **Feature selection is NOT reproducible across machines. Transfer `stage23_resume_cache.pkl`; never let
  a repair run recompute it.** The RTX run had no stage 2/3 caches (they were skipped in the SSD transfer as
  "regenerable"), so it reselected features. 38/44 datasets came out identical, but six did not: Tox21
  (8% overlap, 181 vs 580 features), Ames (16%), LD50 (18%), ESOL (73%), CYP2C9 (93%) and AqSolDB (96%).
  These are mostly datasets where ElasticNetCV hit its 7,200 s limit and the random-forest-importance
  fallback took over. The base models were trained on the ORIGINAL selection (the committed
  `selected_features.csv`, whose sha256 matches `chemprop_selected_descriptor_columns_sha256` in
  `metrics.csv`), so fold refits on a new selection give OOF predictions inconsistent with those models'
  test predictions. Affected member: Chemprop "D-MPNN + Selected descriptors". Repair on 2026-10-01:
  restored the committed selection files, moved the stale Chemprop Selected-descriptor fold caches to
  `benchmark_results/_stale_selection_backup_20261001/`, and copied the original caches over the local
  ones from the A100-era machine. Check with a sha comparison of `selected_features.csv` against git HEAD.
  The cache loader ignores the feature-store path (`_stage23_payload_matches_ignoring_cache_location`),
  so Linux-built caches load on Windows.
- **Tox21 Chemprop OOF: a deterministic one-row drop.** The AttentiveFP and D-MPNN + RDKit2D fold refits
  return 4636 predictions for 4637 training rows (Chemprop cannot featurize one molecule), and
  `_align_chemprop_predictions` refuses to guess, so those two members get no OOF vector and are
  excluded from Tox21's ensembles. The wrapper's retry passes fail the same way. Impact is negligible:
  both base models score test AUROC 0.53-0.60 on Tox21.
- **All `subprocess.run` calls must use `_SUBPROCESS_TEXT_KWARGS`** (`text`/`encoding="utf-8"`/`errors="replace"`),
  never bare `text=True`. Bare `text=True` decodes with the ambient locale; under the C/POSIX locale on
  Jetstream2 that was ASCII, and Chemprop v2's UTF-8 progress output (`0xe2`) made `subprocess.run` itself
  raise `UnicodeDecodeError` before any result was read. That destroyed **86.4% of Chemprop runs at an
  identical rate across all five variants** and was recorded in the same `error` column as genuine training
  failures, which is why it hid for a whole run. Harness I/O errors now say `Chemprop harness error`;
  model failures say `Chemprop training failed`. Regression test: `tests/test_subprocess_encoding.py`
  (needs pytest, which the Windows workstation does not have).
- **Profiles used to overwrite an explicit `--disable-model-families`** (fixed 2026-10-03). The profile code checks
  whether the user set an option by looking for `--<dest>`. That dest is `disabled_model_families`, but the flag is
  `--disable-model-families`, so the user's value was silently replaced (for example, quick re-disabled
  gradient boosting). `_PROFILE_FLAG_ALIASES` maps such dests to their real flag; add an entry if you add a profile
  default whose flag differs from its dest.
- **Binary datasets catalogued as "rmse" saved HARD 0/1 LABELS as test predictions** (found and fixed 2026-10-04 in
  `predict_values_for_metric`). On `tdc_cyp1a2_veith`, `tdc_cyp2c19_veith`, `tdc_herg_karim` and `tdc_pampa_ncats` the
  catalog's metric is rmse, so every model scored through `predict_values_for_metric` (10 conventional and
  gradient-boosting models plus TabPFN) called `.predict()`. Test AUROC/AUPRC were computed on labels (PAMPA AdaBoost
  0.500, LogisticRegression 0.563), and their OOF predictions, so the ensembles, were labels too. Chemprop, Uni-Mol
  and MapLight already returned probabilities. CV AUROC was fine: the scorer calls `predict_proba` itself. The fix
  returns the positive-class score whenever the task is classification. **Repair (2026-10-04):** the 10 models plus
  `XGBoost (ADMETboost features)` were retrained on those 4 datasets (`logs/run_proba_fix.ps1`, backup
  `benchmark_results/_proba_fix_backup_20261004/`). Done and validated on all 4 datasets: no label predictions
  remain, every other model is unchanged, and test AUROC rose by a median of 0.06-0.11 per dataset (PAMPA AdaBoost
  0.500 -> 0.742). `tools/verify_oof_ensemble_run.py` exempts exactly these retrains (`INTENTIONAL_RETRAINS`). TabPFN
  on those 4 datasets still holds label predictions: retraining it needs Prior Labs API credits or a local GPU run that
  already OOMs on them (author decision). Retrained CV values differ by 0.01-0.03 because scaffold `GroupKFold`
  breaks group ties differently across scikit-learn versions (A100 env vs RTX env). Within one machine it is
  deterministic.
- **A filtered resume (`--only-model-names X`) used to DELETE stale rows of models it excludes** (fixed 2026-10-04 in
  `split_stale_metric_rows`). When a cached row's `stage_config_signature` no longer matched, the runner dropped it
  to recompute it, but the filter blocked the recompute, so the row was gone on the next save. It nearly happened in
  the ADMETboost fold-in: on `tdc_cyp2c9_veith`, TabPFNClassifier and both ensemble rows carry signatures that match
  no current flag set (probably written by a late-September TabPFN repair session with different settings), and the
  run was stopped before CYP2C9 was saved. Such rows are now kept, with a `[resume] keeping N row(s)` notice.
  **An UNFILTERED resume would still recompute them**, which would retrain TabPFN on CYP2C9; check before running
  one. Test: `test_filtered_run_keeps_stale_rows_it_will_not_recompute`.
- **Targeted model filters are repeatable exact labels.** Use one `--only-model-names` argument per
  model. Model labels contain commas, so comma-joining labels silently breaks selection. Internal
  TDC multi-seed code stores these filters as a list for the same reason.
- **Chemprop on Windows (RTX box) needs three fixes, all in place since 2026-09-28**
  (details: `docs/HANDOFF_RTX_GPU_WORK.md` status section): `--chemprop-num-workers 0` (each
  DataLoader worker reloads torch/CUDA and exhausts the paging file, `WinError 1455`); the
  runner calls Chemprop through `chemprop_cli_launcher.py`, which makes seeded determinism
  warn-only (torch 2.5.1 has no deterministic CUDA `cumsum`, so every seeded *classification*
  run died in AUROC); and the RTX wrapper redirects via `cmd /c`, because PowerShell 5.1
  `*>` with `ErrorActionPreference=Stop` kills the run on its first stderr line. Intermittent
  Chemprop `exit=3221226505` (0xC0000409) crashes (~2% of folds) are handled by wrapper retry
  passes. The Chemprop env on the RTX is the `qsarena-py311` conda env; system `python` has no torch.
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
- **Multi-seed TDC-22 machinery exists, but no five-seed artifacts exist yet.** Use
  `qsarena-benchmark --tdc22-multiseed --tdc22-multiseed-source-run <existing-run> --output-dir <new-run>`
  to run/resume the work-order C2 stage independently; it writes `results/tdc22_multiseed.csv` by
  default. The older `--run-tdc22-multiseed-best` remains an end-of-run hook. The actual 5-seed TDC-22
  replication has not been run, so the manuscript must keep the single-seed limitation until those
  artifacts exist.
- **OECD / applicability-domain results are reference-model diagnostics, not selected-model evidence.**
  3.13 and Table S8 use `results/reliability_tdc22/summary.json`: a fixed random forest on Morgan
  r=2 fingerprints + RDKit 2D descriptors with 20% of train/val held out for conformal calibration.
  Structural AD is mixed (Roy standardization median out/in error ratio 0.91; kNN Tanimoto 1.15;
  consensus 1.08), while reliability/conformal-width flags are stronger (median out/in error ratio
  3.30; 230.3% higher outside; higher error on all 22 datasets). Do not imply this was run for all
  28 benchmark model families, do not claim multi-seed, and keep the Section 4 limitation.
- **Host provenance: resolved.** The A100 run is real and is now canonical (see above). The RTX 4060
  evidence applies only to `benchmark_results/benchmark_name_date`, which is retained as the
  hardware-comparison arm. On 37 identically split datasets the two runs differ by a median of
  -0.19%, which is the paper's accessibility result, not a discrepancy to fix.
- **PODUAM attribution (fixed 2026-10-02).** The registry, catalog and `leaderboard_top10_reference_latest.csv`
  now credit von Borries K, Beckwith KV, Goodman JM, Chiu WA, Jolliet O, Fantke P, *Nat Commun* 2026;17:647,
  doi:10.1038/s41467-025-67374-4 (software: github.com/kejbo/PODUAM), not "Aurisano et al. 2025". Committed run
  outputs, timestamped leaderboard snapshots and `chemprop_oof_seed/` keep the old string as historical records.
  The `source_label` feeds the stage 2/3 signature (`dataset_source`), which keys cached feature selections and
  every metrics row's `stage_config_signature`; `SIGNATURE_SOURCE_LABEL_ALIASES` in the runner maps the corrected
  labels back so resumes do not retrain. **Any future label correction must add an alias the same way**
  (test: `tests/unit/test_source_label_signature.py`).
- `predictions.csv` files are gitignored and absent locally, so prediction-diversity panels are skipped (expected).
- No TDC-22 multi-seed artifacts exist for the canonical run (`tdc22_best_model_multiseed*`); results are single-split, single-seed.

## History worth knowing

- The manuscript was first drafted (commit `46bcd3d`) from notebook outputs of an older run,
  `all_benchmarks_no_unimol_20260425_223505`, which that same commit deleted from the repo. Recover it if you need it:
  `git archive 745b820 benchmark_results/all_benchmarks_no_unimol_20260425_223505 | tar -x -C <dir>`.
- Committed notebook outputs before 2026-09-24 came from another machine (`C:\Users\scott\QSARena`) and a mixed state.

## Run options, resume and reports (tutorial work order, 2026-09-25)

- **RunConfig defaults must reproduce the old runner behaviour** (user decision). New options (salt
  stripping, dedup, ...) default off; Appendix A's differing "intended defaults" were NOT adopted.
  `tests/fixtures/config/appendix_a_run.yaml` is Appendix A verbatim and must keep loading.
- **Resume is config-signature aware.** Each `metrics.csv` row carries `stage_config_signature`
  (stage 2/3 signature + that model family's args); a changed arg recomputes only that family plus
  fusion/ensembles. Rows and `run_status.json` files from before this change have no signature and are
  reused unvalidated with a printed notice, so resuming older runs never discards work.
- The stage 2/3 cache signature is unchanged for default configs (new keys enter only when non-default),
  so pre-existing `stage23_resume_cache.pkl` files still hit.
- A failing dataset is recorded (`run_status.json` status `failed` + error + remedy) and the run
  continues; `dataset_summary.csv` and `report.html`/`report.md` are written for every mode.
- Test cost: `tests/integration` ~10 min and `tests/docs` ~6 min of CLI runs on the example data.

## Don'ts (cost savers)

- Don't rerun benchmarks locally to "check" numbers: the canonical run recorded ≈155 h of wall-clock on a GPU box.
  Everything the manuscript needs is recomputable from committed `metrics.csv` / selector / runtime artifacts.
- Don't run the notebook with `benchmark_run_dir = "AUTO"` for manuscript work: it picks the most recently modified run
  directory, and the notebook itself writes into run directories.
- Don't commit without being asked; `publication_recommendations.md` may contain the user's uncommitted edits.
- Ignore `node_modules/`, `tmp/`, `catboost_info/`, `model_cache/`, `logs/`, `gin_supervised_masking_pre_trained.pth`.
