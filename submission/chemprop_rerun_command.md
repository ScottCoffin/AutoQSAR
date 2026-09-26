# Chemprop repair re-run — Jetstream2 A100

Two runner defects are now fixed in `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`. Both
require a re-run of the Chemprop models and the downstream fusion/ensemble stage to take effect.
This file is the exact procedure.

**Nothing here touches `benchmark_results/autoqsar_benchmark_20260623_153839/`.** That run stays
pristine as the deposited artifact set. The re-run writes to a new directory and is compared against
the old one afterwards.

---

## 1. What was fixed

| | Defect | Fix |
|---|---|---|
| **A** | `subprocess.run(..., text=True)` with no `encoding=` decoded Chemprop's UTF-8 output using the ambient locale. On the Jetstream2 node that was ASCII (C/POSIX locale), so `subprocess.run` raised `UnicodeDecodeError` before any training result was read. This destroyed **86.4 % of Chemprop runs at an identical rate across all five variants** — the signature of a shared I/O fault, not a model problem. | All five `subprocess.run` call sites now use a shared `_SUBPROCESS_TEXT_KWARGS` with `encoding="utf-8", errors="replace"`. Harness I/O failures are now raised as `Chemprop harness error (subprocess I/O)` and are no longer indistinguishable from `Chemprop training failed (exit=N)`. |
| **B** | Ensemble membership was filtered on **held-out** R², and the correlated-pair tie-break used the **held-out** primary metric — in a framework the paper calls leakage-controlled. | New `--ensemble-member-selection-split {train,test}`, defaulting to **`train`**. The old behaviour is still reachable with `test` for exact reproduction of the deposited run. |

Verified locally: 5/5 subprocess regression assertions pass, the 0xe2 decode failure is reproduced byte-for-byte,
`--help` exits 0. Tests live in `tests/test_subprocess_encoding.py` (needs `pytest`; a
pytest-free equivalent was used to verify on the Windows workstation, which has no pytest installed).

---

## 2. Before you start — two things to know

**The deposited run cannot be resumed in place, and should not be.** A separate rename in this
working tree changed `select_output_dir` to glob `qsarena_benchmark_*`, while the deposited run is
named `autoqsar_benchmark_20260623_153839`. `--resume` will therefore not find it. That is fine —
we want a new directory anyway — but do not "fix" the glob and resume into the canonical run.

**Feature caches live outside the run directory** (`model_cache/feature_store_parquet` and
`model_cache/benchmark_feature_matrix_cache`), so a new output directory still reuses the cached
featurisation and the ElasticNetCV selections. This is what keeps the re-run cheap: featurisation and
selection were the dominant cost (median selector time 295 s/dataset, max 1 003 s).

---

## 3. Step 1 — prove the fix on one dataset first

Do not launch 44 datasets before confirming Chemprop trains at all on this host. Pick one of the six
datasets that survived the bug, so you can compare directly against a known-good result:

```bash
export QSARENA_HOME=/path/to/QSARena
cd "$QSARENA_HOME"

python portable_colab_qsar_bundle/run_qsarena_benchmarks.py \
  --output-dir benchmark_results/chemprop_fix_probe \
  --benchmark-profile full \
  --dataset-name tdc_dili \
  --run-chemprop-dmpnn --run-chemprop-rdkit2d \
  --no-run-chemprop-mpnn --no-run-chemprop-cmpnn \
  --no-run-chemprop-attentivefp --no-run-chemprop-selected-features \
  --no-run-unimol-v1 --no-run-tabpfn --no-run-cfa --no-run-ensemble \
  --chemprop-epochs 3 \
  --chemprop-echo-commands
```

Then confirm, in `benchmark_results/chemprop_fix_probe/tdc_dili/metrics.csv`, that:

- no row's `error` column contains `ascii` or `codec`;
- the two Chemprop rows have a non-null `primary_metric_value`.

**Deliberately reproduce the bug once**, to prove the fix is what changed things and not the host:

```bash
LC_ALL=C LANG=C git stash && \
  python portable_colab_qsar_bundle/run_qsarena_benchmarks.py ... # same command
git stash pop
```

Under `LC_ALL=C` on the *unfixed* code this must fail with `'ascii' codec can't decode byte 0xe2`.
On the fixed code the same environment must succeed. Record both outputs — a reviewer asking "how do
you know the backend is repaired rather than the machine changed?" is answered by exactly this pair.

---

## 4. Step 2 — seed and run only missing Chemprop models

Seed a new repair directory with the canonical metrics, predictions and stage-resume files. Model
directories are deliberately not copied. Successful rows are therefore reused from predictions,
while failed Chemprop rows train into the new directory. The canonical run remains untouched.

```bash
export QSARENA_HOME=/path/to/QSARena
cd "$QSARENA_HOME"

python portable_colab_qsar_bundle/prepare_chemprop_repair_run.py \
  benchmark_results/autoqsar_benchmark_20260623_153839 \
  benchmark_results/qsarena_benchmark_chemprop_fixed

# Restrict discovery to the 44 seeded datasets. This deliberately excludes the
# abandoned tdc_herg_central dataset, which has no completed metrics to repair.
dataset_args=()
for metrics_file in benchmark_results/qsarena_benchmark_chemprop_fixed/*/metrics.csv; do
  dataset_args+=(--dataset-name "$(basename "$(dirname "$metrics_file")")")
done

nohup python portable_colab_qsar_bundle/run_qsarena_benchmarks.py \
  --output-dir benchmark_results/qsarena_benchmark_chemprop_fixed \
  --benchmark-profile full \
  "${dataset_args[@]}" \
  --only-model-names 'Chemprop v2 (D-MPNN, ensemble=3)' \
  --only-model-names 'Chemprop v2 (D-MPNN + RDKit2D, ensemble=3)' \
  --only-model-names 'Chemprop v2 (CMPNN, ensemble=3)' \
  --only-model-names 'Chemprop v2 (AttentiveFP, ensemble=3)' \
  --only-model-names 'Chemprop v2 (D-MPNN + Selected descriptors, ensemble=3)' \
  --only-model-names 'Ensemble' \
  --run-chemprop-mpnn --run-chemprop-dmpnn --run-chemprop-rdkit2d \
  --run-chemprop-cmpnn --run-chemprop-attentivefp --run-chemprop-selected-features \
  --no-run-unimol-v1 --no-run-unimol-v2 --no-run-maplight-gnn \
  --no-run-chemml-pytorch --no-run-chemml-tensorflow --no-run-cnn --no-run-tabpfn \
  --no-run-cfa --run-ensemble --rebuild-ensemble \
  --ensemble-member-selection-split train \
  --reuse-persistent-feature-store \
  --reuse-shared-feature-matrix-cache \
  --chemprop-reuse-model-cache \
  --chemprop-epochs 50 --chemprop-ensemble-size 3 --chemprop-random-seed 42 \
  --resume --no-run-tdc22-multiseed-best \
  > logs/chemprop_fixed.log 2>&1 &
```

Notes on the flags that matter:

- Repeated `--only-model-names` values are exact labels. Successful Chemprop labels already present
  in each seeded `metrics.csv` are skipped; only missing/error labels execute.
- Existing conventional, ChemML, Uni-Mol, MapLight, CFA and successful Chemprop predictions are
  retained from the seed and are available to the rebuilt ensemble without retraining.
- `--rebuild-ensemble` is required: without it the fusion stage may reuse cached ensemble rows and
  neither the new Chemprop members nor the leakage-free selection would take effect.
- `--ensemble-member-selection-split train` is the default; it is written out explicitly so the
  choice is visible in `run_config.json` and in the log.
- `tdc_herg_central` remains excluded (>24 h without completing; abandoned for compute budget).
- Re-run `polaris_adme_fang_hppb_1` to clean completion here too — the review flagged submitting with
  an "interrupted but retained" benchmark.

---

## 5. What this will and will not change

From the RTX run — the only evidence we have of Chemprop actually working — a repaired backend is a
**completeness and credibility fix, not a performance fix**. On the 23 RTX datasets where Chemprop
produced valid results it won outright on 1, reached the top 3 on 3, and sat at median rank 9 of 20
models, a median 30 % relative gap behind the best model. Expect Chemprop to add roughly one win on
44 datasets and not to disturb the headline finding.

The reason to do the re-run anyway is the ensembles: on the RTX run Chemprop was admitted into
**42 of 46** ensembles. The ensemble family supplies 16 of 44 wins in the deposited run, and its
member pool is currently missing an entire model family on 38 of 44 datasets. That, plus the fact
that the RTX run only ever configured AttentiveFP — never D-MPNN or D-MPNN+RDKit2D — is why the
ADMET-AI parity question cannot be settled from existing artifacts.

| Will change | Will not change |
|---|---|
| Chemprop coverage (6/44 → expected ≈40+/44) | Uni-Mol V1/V2 results (not re-run) |
| Ensemble and CFA membership on ~38 datasets | Conventional ML results (deterministic, same seed) |
| Ensemble scores, downward, from removing the held-out-R² filter | Feature selection (cached and reused) |
| Table 3's Chemprop row; Figure 2 family wins; §3.9 coverage | The 22 official TDC splits (`split_train_hash`/`split_test_hash` must match) |

---

## 6. After the run — verification, in order

```bash
# 1. splits must be identical to the deposited run, or nothing is comparable
python - <<'PY'
import pandas as pd, pathlib
old = pathlib.Path("benchmark_results/autoqsar_benchmark_20260623_153839")
new = pathlib.Path("benchmark_results/qsarena_benchmark_chemprop_fixed")
bad = []
for d in sorted(new.iterdir()):
    f, g = d / "metrics.csv", old / d.name / "metrics.csv"
    if not (f.exists() and g.exists()):
        continue
    a = pd.read_csv(f, low_memory=False)
    b = pd.read_csv(g, low_memory=False)
    for col in ("split_train_hash", "split_test_hash"):
        if set(a[col].dropna()) != set(b[col].dropna()):
            bad.append((d.name, col))
print("split mismatches:", bad or "none")
PY

# 2. no decode errors survive anywhere
grep -rl "codec can't decode" benchmark_results/qsarena_benchmark_chemprop_fixed/ || echo "clean"

# 3. regenerate every manuscript number against the NEW run, then diff
#    (edit benchmark_run_dir in the notebook's first cell first)
python portable_colab_qsar_bundle/render_manuscript_assets.py
python portable_colab_qsar_bundle/verify_manuscript_numbers.py   # WILL fail: see below
```

`verify_manuscript_numbers.py` asserts 64 values against the deposited run and **is expected to fail
after this re-run** — that is the point. Work through its failures one at a time and update each
assertion to the new value, rather than relaxing the assertions. Every changed number must also be
propagated to **both** `manuscript.md` and `submission/body.tex`; the verifier only reads the former,
so a number fixed in one and not the other passes silently.

Keep the deposited run as the hardware/legacy comparison arm, exactly as
`benchmark_results/benchmark_name_date` is kept today, and report the ensemble score change from
removing the held-out-R² filter explicitly. That delta is a result in its own right: it quantifies
what the leakage was worth, which is a stronger paper than having never leaked.

---

## 7. Ensemble OOF repair (2026-09-26) — run this before regenerating the manuscript

**Why.** The Chemprop repair run (`qsarena_benchmark_chemprop_fixed`) removed the held-out leak
with `--ensemble-member-selection-split train`, but that mode selects, weights and stacks members on
their *in-sample* training predictions, which rewards memorisation. On ESOL the stacking meta-model
put 99.99% of its weight on extra trees (training RMSE 0.026) and dropped MapLight CatBoost; across
the run ensemble regression wins fell from 7 to 1 and OOF stacking sat a median 16% behind the best
single model. Those ensemble results must not be reported.

**What an honest ensemble needs, and what is already on disk.** Stacking and inverse-error
weighting need, for every training molecule, a prediction from a model that did not train on it
(out-of-fold, OOF). The saved full models and their `predictions.csv` are reused as they are; no
base model is retrained. But their predictions for training molecules are in-sample, so rebuilding
the ensemble from saved files alone can only reproduce the memorisation bias (`train`) or the leak
(`test`). What each family needs:

| Members | OOF source | Cost |
|---|---|---|
| Uni-Mol V1 / V2 | **Already saved.** `unimol_tools` trains 5 internal folds and writes each training molecule's out-of-fold prediction to `cv.data` in the model folder. These are the reported model's own fold models. | none (read from disk) |
| Conventional ML, MapLight CatBoost, TabPFN | Their CV fold models were not kept, so they are refitted on 5 folds | CPU; TabPFN uses API tokens |
| ChemML MLPs, MapLight + GNN | Refitted on 5 folds | CPU, minutes per dataset |
| Chemprop (5 variants) | Only a random 10% was held out, for early stopping; there is no full OOF vector | **GPU fold refits: the only expensive part** |

`--ensemble-member-selection-split oof` (the default) does all of this: saved OOF first, then fold
refits, cached per fold under `<dataset>/ensemble_oof/` and saved as `split="oof"` rows in
`predictions.csv`, so the stage resumes. CFA is not an ensemble member. The weighted ensemble is
labelled `Ensemble (Weighted average (inverse OOF error))`. Tests: `tests/unit/test_ensemble_oof.py`.

**Choose the Chemprop option.** Everything else is the same either way.

| `--ensemble-oof-scope` | Chemprop in ensembles? | A100 time (estimate) |
|---|---|---|
| `cpu` | No: left out of the ensembles, with the reason logged | ~10 h of CPU refits (mostly MapLight + GNN) |
| `all` | Yes: 5 variants × 5 folds refitted on the GPU | ~65 h (≈1.5 h per median dataset); `--ensemble-oof-folds 3` cuts that to ~40 h |

If you choose `cpu`, the paper must say that the ensembles exclude Chemprop.

```bash
export QSARENA_HOME=/path/to/QSARena
cd "$QSARENA_HOME"

python portable_colab_qsar_bundle/prepare_chemprop_repair_run.py \
  benchmark_results/qsarena_benchmark_chemprop_fixed \
  benchmark_results/qsarena_benchmark_oof_ensemble

dataset_args=()
for metrics_file in benchmark_results/qsarena_benchmark_oof_ensemble/*/metrics.csv; do
  dataset_args+=(--dataset-name "$(basename "$(dirname "$metrics_file")")")
done

# Uni-Mol folders are found through the path recorded in metrics.csv; --ensemble-oof-source-run is
# a fallback for when the run was moved. The Uni-Mol / Chemprop / TabPFN settings are the ones that
# produced the reported models (canonical run for Uni-Mol, chemprop_fixed for Chemprop and TabPFN),
# so any fold refit matches them.
nohup python portable_colab_qsar_bundle/run_qsarena_benchmarks.py \
  --output-dir benchmark_results/qsarena_benchmark_oof_ensemble \
  --benchmark-profile full \
  "${dataset_args[@]}" \
  --only-model-names 'Ensemble' \
  --run-ensemble --rebuild-ensemble \
  --ensemble-member-selection-split oof --ensemble-oof-folds 5 \
  --ensemble-oof-scope all \
  --ensemble-oof-source-run benchmark_results/autoqsar_benchmark_20260623_153839 \
  --run-tabpfn --tabpfn-max-train-rows 11000 \
  --unimol-batch-size 32 --unimol-max-atoms 64 \
  --run-chemprop-mpnn --run-chemprop-dmpnn --run-chemprop-rdkit2d \
  --run-chemprop-cmpnn --run-chemprop-attentivefp --run-chemprop-selected-features \
  --chemprop-epochs 40 --chemprop-ensemble-size 3 --chemprop-random-seed 42 \
  --reuse-persistent-feature-store --reuse-shared-feature-matrix-cache \
  --resume --no-run-tdc22-multiseed-best \
  > logs/oof_ensemble.log 2>&1 &
# --only-model-names 'Ensemble' keeps every full model from being retrained. The --run-chemprop-*
# flags only list the Chemprop variants whose OOF predictions to build.
```

**TabPFN** runs through the API client, and the repair run already hit its daily token limit. Its
fold refits may hit it again. That is not fatal: that member is left out of that dataset's
ensembles, with the reason in `ensemble_member_filter_notes`. Running TabPFN locally on the GPU
avoids the limit.

**Checks after the run.**

```bash
grep -c "config signature changed" logs/oof_ensemble.log     # must be 0, otherwise stop and investigate
grep "cv.data" logs/oof_ensemble.log | head                  # Uni-Mol should read saved OOF, not refit
python - <<'PY'
import pandas as pd, pathlib
run = pathlib.Path("benchmark_results/qsarena_benchmark_oof_ensemble")
rows = pd.concat([pd.read_csv(p, low_memory=False).assign(dataset=p.parent.name)
                  for p in run.glob("*/metrics.csv")])
ens = rows[rows["model"].astype(str).str.startswith("Ensemble (") & rows["error"].isna()]
print(ens["ensemble_member_selection_split"].value_counts(dropna=False))
print(ens.groupby("model").size())
notes = ens["ensemble_member_filter_notes"].fillna("")
print(notes[notes.str.contains("refit failed|no out-of-fold|unusable", regex=True)].head(20).to_string())
PY
```

Then regenerate from the new run:
`python portable_colab_qsar_bundle/render_manuscript_assets.py --run-dir benchmark_results/qsarena_benchmark_oof_ensemble`.
