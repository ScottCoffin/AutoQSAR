# Handoff: finish the OOF ensemble rebuild on a local CPU machine

You are picking up work that was started on an NSF Jetstream2 A100 instance and stopped so the
instance could be shelved. **Nothing here needs a GPU.** Read this file and the AGENTS.md section
"A100 task: regenerate the clean (out-of-fold) ensemble results" before running anything.

## What the job is

Produce `benchmark_results/qsarena_benchmark_oof_ensemble`: the same base models as
`qsarena_benchmark_chemprop_fixed`, but with ensembles rebuilt from **out-of-fold (OOF)** member
predictions instead of full-fit predictions. The old ensembles are leaky: members were weighted by
inverse *train* RMSE using predictions from models that had seen those rows.

**No full model is ever retrained.** Every base model's train/test predictions already exist. The
only training is small *fold* models for OOF predictions, all CPU.

## Where the work stopped

| Step | State |
|---|---|
| 0 code + tests | done, 14 passed |
| 1 inputs verified | done (44 `predictions.csv`, 61 `cv.data`) |
| 2 seed run dir | done |
| 3 planner, both scopes | done, all `--scope cpu` criteria passed |
| **4 pilot** | **not started - resume here** |
| 5 full CPU run | not started |
| 6 Chemprop GPU | not approved, and deliberately out of scope |

The committed planner output is
`benchmark_results/qsarena_benchmark_oof_ensemble/ensemble_oof_plan.csv` (1021 member rows).

## Before you run anything: check you have all three data sources

The repo does **not** carry the files this job needs. Two came off the A100 as folder downloads and
one is a 1.2 MB set that is easy to miss.

| # | What | Where it must end up | Size | In git? |
|---|---|---|---|---|
| 1 | Base run incl. `predictions.csv` | `benchmark_results/qsarena_benchmark_chemprop_fixed/` | ~7 GB | partly - `predictions.csv` is gitignored |
| 2 | Seeded OOF run | `benchmark_results/qsarena_benchmark_oof_ensemble/` | ~6 GB | only `ensemble_oof_plan.csv` |
| 3 | Uni-Mol `cv.data` | `benchmark_results/autoqsar_benchmark_20260623_153839/**/` | 1.25 MB | **yes - committed, arrives with the clone** |

Sources 1 and 2 must be copied from the A100 (or from wherever you archived them); they are too
large for git. Source 3 now ships **in the repo**: the 61 `cv.data` files (1.25 MB) were
un-gitignored specifically so this handoff needs no side-channel for them. They carry Uni-Mol's
internal-fold predictions, and without them all 61 Uni-Mol members would drop out of the
ensembles. Do **not** fetch the rest of that 89 GB directory - the 68 GB of `model_*.pth` files
are still ignored and are not needed.

Verify all three before starting:

```bash
ls benchmark_results/qsarena_benchmark_chemprop_fixed/*/predictions.csv | wc -l   # expect 44
ls benchmark_results/qsarena_benchmark_oof_ensemble/*/predictions.csv | wc -l     # expect 44
find benchmark_results/autoqsar_benchmark_20260623_153839 -name cv.data | wc -l   # expect 61
```

If any count is short, **stop**. A missing `predictions.csv` means the ensemble cannot be rebuilt
without retraining, which is not allowed.

## If the downloaded folders were renamed

The A100 folders may have been downloaded under names like
`qsarena_benchmark_chemprop_fixed_full_data_` and `qsarena_benchmark_oof_ensemble_full_data_`.
**Rename them to the canonical names above before running anything.** Two reasons:

1. Every command in the runbook, and `--ensemble-oof-source-run`, uses the canonical paths.
2. `select_output_dir` globs `qsarena_benchmark_*`, so a stray `*_full_data_` copy can be picked up
   by directory auto-selection. Keep only one directory per canonical name.

The download is a **superset** of what git has (it adds the gitignored `predictions.csv` and
caches), so overlay it onto the clone rather than replacing the clone. Afterwards run:

```bash
git status --short benchmark_results/qsarena_benchmark_chemprop_fixed/
```

This should print **nothing**. If it lists modified tracked files, the download is older than
`main`; prefer the git version for tracked files (`git checkout -- <path>`) and keep the download
only for gitignored ones.

## Step 4 - pilot on one dataset first

Smallest dataset is `tdc_carcinogens_lagunin` (n_train=223). Snapshot its metrics first:

```bash
cp benchmark_results/qsarena_benchmark_oof_ensemble/tdc_carcinogens_lagunin/metrics.csv \
   /tmp/carcinogens_metrics_before.csv
```

Run the step 5 command from AGENTS.md with a single `--dataset-name tdc_carcinogens_lagunin`,
logging to `logs/oof_pilot.log`. All four checks must pass:

```bash
L=logs/oof_pilot.log
# (a) nothing retrained - must print nothing, and the count must be 0
grep -E "stage [0-9]+/[0-9]+: (building molecular features|splitting data|conventional model|deep model|GA tuning|CFA)" $L | grep -v "(cached)"
grep -c "config signature changed" $L
# (b) Uni-Mol read from disk, no GPU fold refits
grep "ensemble-oof" $L | grep -i "uni-mol"     # must mention reading saved internal-fold predictions
grep "ensemble-oof" $L | grep -iE "(chemprop|uni-mol).*fold [0-9]"   # must print nothing
```

(c) every non-ensemble model's `primary_metric_value` identical between
`/tmp/carcinogens_metrics_before.csv` and the new `metrics.csv`.
(d) new ensemble rows have `ensemble_member_selection_split == oof` and no unexpected
`refit failed` in `ensemble_member_filter_notes`.

**If any check fails, stop and report. Do not work around it.**

## Step 5 - full CPU run

Use the step 5 command exactly as written in AGENTS.md. `--only-model-names 'Ensemble'` is what
prevents full-model retraining - do not add model names to it, and do not drop the
`--run-tabpfn` / `--run-chemprop-*` / Uni-Mol flags, which only declare which members exist.

Expected work: **2945 CPU fold trainings** across 589 members.

| kind | members | fold trainings | median full-fit | est. hours |
|---|---|---|---|---|
| conventional | 462 | 2310 | 7 s | 4.5 |
| chemml | 84 | 420 | 40 s | 4.7 |
| maplight_gnn | 43 | 215 | 137 s | 8.2 |
| **total** | **589** | **2945** | | **~17 h** |

That ~17 h is calibrated on the 32-core A100 host with `--n-jobs 32`. On a 16-core box expect
roughly 35 h and on 8 cores roughly 69 h; it will not scale perfectly. Set `--n-jobs` to your core
count. The run is resumable: re-running the identical command reuses fold caches under
`<dataset>/ensemble_oof/`, so interrupting it is safe.

## Hard rules

- Never retrain a full model. Fold models for OOF only.
- Never use `--ensemble-oof-scope all` (that is the Chemprop GPU path, ~63 h, needs the user's
  explicit approval).
- Never enable `--ensemble-oof-allow-api-refits`. TabPFN is metered; it stays out of the ensembles.
- Never write into `benchmark_results/autoqsar_benchmark_20260623_153839`. Read-only input.
  `qsarena_benchmark_chemprop_fixed` is also read-only for this job.
- Don't regenerate manuscript assets as part of this job; that is a separate, later step.

## Expected outcome, and the caveat to carry into the paper

Ensembles will be rebuilt from OOF predictions for conventional, ChemML and MapLight members, with
Uni-Mol read from `cv.data`. **Chemprop, TabPFN and CFA are excluded** (202 / 38 / 131 members).

Chemprop is excluded because it is refit-or-exclude: its `splits.json` is a list of length 1, so
all three ensemble members share one 90/10 split. Only ~10% of training rows have a held-out
prediction and the other 90% of `train_predictions.csv` is in-sample - the exact leakage being
removed. 10% coverage cannot fit stacking weights.

In the old leaky ensembles Chemprop held 10.7% of total |weight| across 42/44 datasets. Treat that
as an **upper bound** on what is lost: the weighted-average method weights by inverse *train* RMSE,
and Chemprop's train predictions are 90% in-sample, so its weight was inflated by the bug. The
manuscript must state that the clean ensembles exclude Chemprop.

## After the run

Run the "Checks after the run" in `submission/chemprop_rerun_command.md` section 7, then re-run the
planner - the `cpu_refit` members should now report `saved_oof`, confirming nothing is refitted
twice. Commit `benchmark_results/qsarena_benchmark_oof_ensemble` (predictions, `ensemble_oof/` fold
caches and model files are gitignored) with the scope used and the planner summary in the message.
