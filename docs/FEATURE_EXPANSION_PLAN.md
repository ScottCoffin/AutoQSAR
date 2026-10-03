# Feature-expansion arm: ADMETboost features + pretrained embeddings

Status tracker and design for improvement items 2 and 3 from the 2026-09-29 review of MaxQsaring,
ADMETboost and the NIST meta-model. **Update the checklist at the bottom whenever you finish a step.**

## Why

On the 22 official TDC ADMET splits, QSARena's *honest* (CV-selected) pick loses head-to-head to
MaxQsaring (3-19), MolE, ADMETboost and the NIST meta-model (see `docs/meta_analysis/` and the
2026-09-29 notes in AGENTS.md). All three published methods share two ingredients that QSARena lacks:

1. **Descriptor-rich features, all concatenated.** The ADMETboost set (MACCS, ECFP4, Mol2Vec, PubChem,
   Mordred 2D, RDKit descriptors) is fed to XGBoost with **no feature selection**; Mordred was ADMETboost's
   most important feature group on every task. QSARena has no Mordred, Mol2Vec or PubChem, and its tree
   models only see the subset that ElasticNetCV (a linear filter) selected.
2. **Pretrained molecular representations as plain features** (MaxQsaring: GROVER, kBERT, Chemprop and GIN
   representations, fed to XGBoost).

Per-task hyperparameter tuning, their third ingredient, is deliberately **out of scope**: it is likely
over 12 h on the RTX laptop and would break the paper's "one fixed configuration" claim.

## Design (an experimental arm, not a change to the canonical run)

- **Same partitions:** every dataset uses the committed, hash-verified partitions
  (`data/meta_analysis/dataset_partitions.csv.gz`), so results are directly comparable with the
  benchmark and the leaderboards. The benchmark runner and its results are not modified.
- **Code:** `qsarena/feature_expansion/`
  - `featurize.py`: per-dataset feature matrices for the new families, cached as `.npy` in
    `.model_cache/feature_expansion/<family>/<dataset>.npy`, row order = partition order
    (train rows by row_index, then test rows). Gitignored; regenerable.
  - `train.py`: fixed-configuration XGBoost / CatBoost / random forest on the full concatenation (no
    selection). 5-fold CV on the training partition for an honest CV score, then one full fit and one
    test evaluation.
  - `evaluate.py`: (a) new models versus the benchmark's models per dataset; (b) the honest CV-selected
    pick with the new models added to the pool; (c) the 22-dataset TDC head-to-head against MaxQsaring,
    MolE, ADMETboost and the NIST meta-model.
- **Outputs:** `benchmark_results/qsarena_feature_expansion/` (metrics CSVs committed; features and models
  not).
- **Families**

  | family | source | dim | notes |
  |---|---|---|---|
  | `maccs` | skfp MACCSFingerprint | 166 | |
  | `ecfp4` | skfp ECFPFingerprint (r=2, 2048 bits) | 2048 | as DeepChem CircularFingerprint |
  | `rdkit2d` | skfp RDKit2DDescriptorsFingerprint | ~200 | |
  | `pubchem` | skfp PubChemFingerprint | 881 | local RDKit implementation (DeepChem's queries the PubChem server) |
  | `mordred` | skfp MordredFingerprint (2D, mordredcommunity) | 1613 | about 21 ms per molecule |
  | `mol2vec` | own implementation of Jaeger et al. 2018 with the authors' `model_300dim.pkl` | 300 | model at `.model_cache/mol2vec/model_300dim.pkl`, sha256 `62934b4e...c2331f`, from github.com/samoturk/mol2vec |
  | `unimol_repr` | unimol_tools UniMolRepr (pretrained V1, no fine-tuning) CLS embedding | 512 | GPU; label-free |
  | `chemeleon` | Chemprop 2.2 CheMeleon foundation message passing, mean-pooled | 2048 | GPU; label-free |

  Only label-free embeddings are used. Supervised fine-tuned embeddings (as in MaxQsaring) would leak
  training labels into the tree model's inputs unless they were produced out of fold.
- **Feature sets evaluated:** `admetboost` (the first six families), `admetboost+emb` (all eight), and
  `emb` (the two embeddings only).
- **Models (fixed config, no tuning):** XGBoost (`n_estimators=1000, learning_rate=0.05, max_depth=6,
  subsample=0.8, colsample_bytree=0.5, min_child_weight=1, tree_method=hist`), CatBoost (defaults,
  1000 iterations), random forest (500 trees, `max_features="sqrt"`). NaN/inf values in descriptors are
  set to 0, as ADMETboost does.
- **Pre-specified primary comparison:** honest CV-selected pick, old pool versus old pool plus the new
  models, on the 22 official TDC splits (head-to-head wins against each full-coverage method, and
  mean rank). The per-dataset test-selected numbers are secondary.

## Compute

| step | where | estimate |
|---|---|---|
| featurize maccs/ecfp4/rdkit2d/pubchem/mordred/mol2vec | CPU, system Python, 2 low-priority workers | ~40 min |
| featurize unimol_repr | conformers on CPU, then GPU | ~1.5 h (after the GPU run) |
| featurize chemeleon | GPU | ~15 min (after the GPU run) |
| train 3 models x 3 feature sets x 44 datasets x (5 CV + 1 full fit) | CPU (XGBoost can use the GPU) | ~4-8 h CPU; much less with GPU XGBoost |

Run CPU steps at **below-normal priority** with capped threads while the Chemprop OOF run is live. If
Chemprop fold times rise more than ~20%, pause and resume after the run.

## How to run

```bash
python -m qsarena.feature_expansion.featurize --families maccs ecfp4 rdkit2d pubchem mordred mol2vec --workers 2
python -m qsarena.feature_expansion.train --feature-set admetboost
python -m qsarena.feature_expansion.evaluate
```

(Embedding families need the benchmark conda env with torch, Chemprop 2.2 and unimol_tools; on the RTX box
its path is in AGENTS.md, Chemprop-on-Windows trap.)

## Checklist

- [x] Scope and dependencies (2026-09-29). System Python has scikit-fingerprints, mordredcommunity and
      gensim installed. The Mol2Vec model is downloaded. Nothing was installed into the benchmark env
      (the GPU run was live).
- [x] `featurize.py` for CPU families plus tests (Mol2Vec vocabulary hit rate 99.9%; Mordred ~9.7% non-finite -> 0)
- [x] CPU featurization run: all 6 families x 44 datasets done 2026-09-29 15:15 (3.1 GB cache; Mol2Vec hit rate >= 97.7% per dataset; Mordred non-finite share <= 15.7%, set to 0)
- [x] `train.py` plus tests (smoke on 3 cheap families: hERG RF test AUROC 0.859 vs QSARena CV pick 0.732; Caco-2 XGB MAE 0.284 vs 0.302)
- [x] train `admetboost` set: done 2026-09-29 23:14 (44 datasets x XGBoost + RF; ~8 CPU-h at below-normal).
      Fixed afterwards without retraining: `cv_primary` now uses the benchmark's recorded primary metric even
      on binary tasks catalogued as rmse (18 rows recomputed from the saved OOF), and `evaluate.py` treats
      `mse` (Polaris) as lower-is-better.
- [x] **Result, 22 official TDC splits** (`benchmark_results/qsarena_feature_expansion/tdc22_*.csv`):
      | honest entry | vs MaxQsaring | vs ADMETboost | vs NIST | vs MolE | mean rank |
      |---|---|---|---|---|---|
      | CV pick, benchmark pool | 3-19 | 9-13 | 11-11 | 9-13 | 4.77 |
      | CV pick, benchmark + arm (PRIMARY, pre-specified) | 3-19 | 10-12 | 11-11 | 9-13 | 4.64 |
      | fixed: XGBoost [admetboost] (secondary, pre-specified) | 5-17 | **13-9** | **13-9** | 11-11 | **3.91** |
      | fixed: RF [admetboost] | 1-21 | 2-20 | 4-18 | 4-18 | 6.32 |
      MaxQsaring's mean rank is 1.96. Reading: the arm's XGBoost is the best honest QSARena entry (second only to
      MaxQsaring), but CV selection almost never picks it (1 of 22), so the primary comparison barely
      moves. The bottleneck is selection: for predefined splits the benchmark's CV uses RANDOM folds while
      TDC's test split is SCAFFOLD; MaxQsaring selects with scaffold 5-fold CV.
- [ ] NEXT (new, not pre-specified; label it as such): honest selection on **scaffold-fold** CV/OOF scores
      for the arm plus the benchmark's conventional models. CORRECTION 2026-09-30: the benchmark's saved OOF uses the same RANDOM folds, so scaffold scores need refits: arm and conventional models on CPU (~7+ CPU-h); Chemprop and Uni-Mol would need the GPU. Part 1 (arm only) APPROVED and STARTED 2026-10-01 ~19:50: `python -m qsarena.feature_expansion.train --feature-set admetboost --cv-strategy scaffold --cv-only` -> `benchmark_results/qsarena_feature_expansion/admetboost__scaffoldcv/`; log `logs/feature_expansion_scaffoldcv.log`; below-normal, n_jobs 2; pause if Chemprop folds slow >10% (CYP2C9 D-MPNN baseline ~450 s). Part 2 (benchmark conventional models on scaffold folds) not approved yet.
- [x] **Scaffold-CV part 1 (exploratory), done 2026-10-02** (`qsarena/feature_expansion/cv_geometry.py`;
      outputs in `admetboost__scaffoldcv/`). Arm only: scaffold CV picks the better of XGB/RF on 36/44
      (vs 34/44 with the benchmark's folds), and is calibrated on TDC predefined splits (median
      overstatement 0% vs ~4%). **Bigger finding:** the benchmark's own models overstate CV by 11-44% on
      the same folds, from feature selection fitted outside the CV loop (see the AGENTS.md trap). That,
      more than fold geometry, is why honest CV never picks the arm's models. DECIDED 2026-10-02 (user, option 3): (a) a caveat in the paper NOW, with the measured
      inflation computed in the render pipeline (not hand-typed); (b) nested-selection CV for the benchmark's
      conventional models was scoped in `docs/NESTED_SELECTION_CV_PLAN.md` (full ~18 h on an A100 box, >60 h here)
      and then DECLINED by the user (2026-10-02, not worth the compute). The caveat is the final treatment.
- [x] `evaluate.py` (port of the TDC head-to-head analysis) plus tests. Note: it selects on `cv_primary` (the dataset's primary metric) from the live run files, which is slightly kinder than Table 4's notebook CV pick (benchmark pool vs MaxQsaring 3-19 either way; vs ADMETboost 9-13 here vs 6-16 in Table 4). Old-pool and new-pool comparisons inside evaluate.py use identical logic.
- [ ] embedding families: `embeddings.py` written and smoke-tested 2026-10-02 (CheMeleon 2048-d from Zenodo 15460715,
      cached at ~/.chemprop/chemeleon_mp.pt; Uni-Mol V1 CLS 512-d).
      - [x] CheMeleon featurized, 44/44 (`.model_cache/feature_expansion/chemeleon/`).
      - [x] `unimol_repr` featurized 44/44 (done 2026-10-02 ~19:10; `.model_cache/feature_expansion/unimol_repr/`).
            Embedding families now run one dataset per child process (`max_tasks_per_child=1`): unimol_tools leaked
            ~3 GB per large dataset (15.7 GB committed after five) until the box started paging.
- [x] GPU XGBoost on `admetboost+chemeleon`: 44/44, done 2026-10-02 ~14:45. **Result: CheMeleon adds nothing
      measurable to the ADMETboost features.** Paired test metric vs `XGBoost [admetboost]` on 35 datasets with a
      comparable test metric: better on 19, worse on 15, median +0.18% (mean -0.87%), Wilcoxon p = 0.54. Biggest
      losses: half-life Spearman -23%, hepatocyte/microsome clearance -7%; biggest gains: CYP2C9-substrate AUPRC +13%.
      TDC-22 head-to-head is unchanged within noise (vs ADMETboost 12-10 vs 13-9; vs NIST 14-8 vs 13-9; vs MaxQsaring
      4-18 vs 5-17). Note that mean ranks in `tdc22_mean_rank.csv` shift whenever entries are added, since arm
      entries compete with each other; compare head-to-head records instead.
- [x] `chemeleon` alone: 44/44, done 2026-10-02 ~16:10. **Result: label-free CheMeleon embeddings alone are
      significantly worse than the ADMETboost features** (paired, 35 datasets: better on 12, worse on 23, median
      -1.78%, Wilcoxon p = 0.010). They are still competitive with published TDC models (vs ADMETboost 11-11;
      vs NIST 10-12; vs MolE 8-14; vs MaxQsaring 3-19). Together with the null `admetboost+chemeleon` result:
      frozen CheMeleon embeddings carry little information beyond the descriptor set. Fine-tuning (option B) is
      the only route left for CheMeleon, and it is unscoped.
- [x] `evaluate.py` fix (2026-10-02): it now skips `<set>__<variant>` dirs. `admetboost__scaffoldcv` repeats the model
      names with scaffold-CV scores and no test metrics, which duplicated CV-pick candidates and blanked the fixed
      models' test values. Test: `test_load_candidates_skips_cv_protocol_variant_dirs`.
- [x] `python -m qsarena.feature_expansion.evaluate` run on all five feature sets (2026-10-02).
- [x] `admetboost+emb` 44/44 (done 2026-10-02 ~21:40). **Result: null.** Paired vs `XGBoost [admetboost]`, 35 datasets:
      better on 16, worse on 19, median -0.04% (mean -1.59%), Wilcoxon p = 0.30; half-life again the worst (-35%).
      Vs `admetboost+chemeleon`: better on 12, worse on 23, p = 0.058, so Uni-Mol CLS embeddings add nothing on top.
- [x] `emb` alone 44/44 (done 2026-10-02 ~22:40). **Result: worst XGBoost set.** Paired vs `XGBoost [admetboost]`:
      better on 7, worse on 28, median -2.42%, p < 0.001. Vs `chemeleon` alone: better on 9, worse on 26, p = 0.027.
- [x] write-up: see "Results and recommendation (2026-10-02)" below. The integration decision is the author's.

## Results and recommendation (2026-10-02)

Fixed-configuration XGBoost, committed partitions, test metrics on the dataset's primary metric. Paired comparisons
use the 35 datasets with a comparable test metric; TDC-22 records are head-to-head against published values.

| feature set | features | vs `admetboost` (paired) | vs ADMETboost | vs NIST | vs MolE | vs MaxQsaring |
|---|---|---|---|---|---|---|
| `admetboost` | ~5.2k descriptors + fingerprints | — | **13-9** | **13-9** | 11-11 | 5-17 |
| `admetboost+chemeleon` | + CheMeleon 2048-d | 19-15, median +0.18%, p = 0.54 | 12-10 | 14-8 | 10-12 | 4-18 |
| `admetboost+emb` | + CheMeleon + Uni-Mol 512-d | 16-19, median -0.04%, p = 0.30 | 11-11 | 15-7 | 10-12 | 4-18 |
| `chemeleon` | CheMeleon only | 12-23, median -1.78%, p = 0.010 | 11-11 | 10-12 | 8-14 | 3-19 |
| `emb` | CheMeleon + Uni-Mol only | 7-28, median -2.42%, p < 0.001 | 6-16 | 9-13 | 9-13 | 2-20 |

(Mean ranks in `tdc22_mean_rank.csv` move whenever an entry is added, because arm entries compete with each other;
use the head-to-head records above.)

**Reading.** Frozen, label-free embeddings (CheMeleon, Uni-Mol V1 CLS) add nothing measurable to the ADMETboost
descriptor set, and on their own are significantly worse than it. The arm's one positive result is unchanged: a fixed
XGBoost on the full, unselected ADMETboost features is the strongest honest QSARena entry on the 22 official TDC
splits, beating the published ADMETboost and NIST meta-model 13-9. MaxQsaring is still clearly ahead (5-17). CV
selection rarely picks the arm's model because the benchmark's own CV scores are inflated by feature selection
(`limitation_cvleak`).

**Recommendation (for the author to decide).**
1. Do **not** add frozen embeddings to the runner. If CheMeleon is pursued, only fine-tuning (option B, a Chemprop
   variant starting from the foundation checkpoint) remains untested; it is unscoped.
2. Consider adding `XGBoost [ADMETboost features, unselected]` to the runner's model library as an opt-in model
   (it needs scikit-fingerprints, mordredcommunity and gensim plus the Mol2Vec model, so not in the default install).
   That would change the canonical model set, so it belongs in a future run, not this paper's numbers.
3. For the paper: at most one supplementary sentence or table row (the TDC-22 head-to-head of the fixed
   `admetboost` XGBoost), clearly labelled as a post-hoc arm outside the benchmark's fixed configuration. Nothing in
   the arm changes the paper's main claims.

## Implementation of the recommendations (2026-10-03, author-approved)

1. **No frozen embeddings in the runner** (decision recorded). Option B, CheMeleon *fine-tuning*, is now
   implemented as an opt-in Chemprop variant: `--run-chemprop-chemeleon` (RunConfig `deep.chemprop.variants:
   [chemeleon]`; never a profile default). The spec is `chemeleon_variant_spec()` in `qsar_workflow_core.py`; it
   passes `chemprop train --from-foundation CHEMELEON` (Chemprop >= 2.2; the checkpoint caches at
   `~/.chemprop/chemeleon_mp.pt`). Label: `Chemprop v2 (CheMeleon fine-tuned, ensemble=N)`, family `graph_nn`.
   Verified end to end on the example data. **Pilot** (Caco-2, 728 train rows, 40 epochs, one model, RTX 4060):
   162 s vs 22 s for D-MPNN (9.3 M vs 318 K parameters); test MAE 0.382 vs 0.380, so no gain on this single
   dataset. **Cost of a full benchmark run:** ~23 GPU-h for the base model (ensemble=3) and ~93 GPU-h more for 5-fold
   OOF ensemble membership (~116 h total). That exceeds the 12 h approval threshold, so it has **not been run**.
2. **Opt-in `XGBoost (ADMETboost features)`** in the runner: `--run-admetboost-xgboost` (RunConfig
   `models.admetboost_xgboost`). The arm's fixed XGBoost (`XGB_PARAMS` in `train.py`, the single source of truth)
   on the full, unselected ADMETboost features (`featurize.admetboost_matrix`), computed label-free on train+test
   SMILES. It joins the conventional loop with its own feature matrix (like MapLight CatBoost), so it gets CV
   metrics without the selection leak, train/test/OOF predictions and ensemble membership, and the ensemble OOF stage
   refits it on the same features. It needs `pip install qsarena[features]` plus the Mol2Vec model; if they are
   missing, the row records the error and a later resume retries it. Family `gradient_boosting`, so the quick
   profile disables it unless `--disable-model-families` is given. That exposed a bug, now fixed: profiles overwrote
   an explicit `--disable-model-families`. Neither flag enters any family's resume signature (tested), so existing
   runs resume unchanged. It belongs in a future run, not this paper's numbers.
3. **Paper:** a "post-hoc check with unselected descriptors" paragraph in §3.12 (13-9 vs ADMETboost and NIST, 5-17
   vs MaxQsaring; embeddings null), labelled as outside the fixed configuration and naming both opt-in models; the
   arm's output directory is listed under Availability of data and materials. Numbers are checked by
   `verify_manuscript_numbers.py` from `tdc22_head_to_head.csv` and `evaluation_summary.json`
   (`paired_vs_admetboost_xgboost`, now written by `evaluate.py`).

Tests: `tests/unit/test_opt_in_models.py` and `tests/feature_expansion/test_evaluate.py`.

