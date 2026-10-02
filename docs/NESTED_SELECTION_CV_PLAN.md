# Nested-selection CV: scope (follow-up to the 2026-10-02 CV-leak decision)

**Status: DECLINED by the user on 2026-10-02 (not worth the compute). Kept for the record.** The paper
keeps the auto-rendered caveat (`limitation_cvleak`) instead. Don't start this without a new decision.

## Problem

Stage 3 fits the ElasticNetCV feature selector once on all training rows. Each model's
`cross_validate` then runs on `X_train_selected`, so every CV validation fold helped choose the
features. The manuscript's caveat (`limitation_cvleak` META block, `qsarena/meta_analysis/cv_leak.py`)
measures the effect: median CV-vs-test overstatement is 15.9% for benchmark models (per model
4.7-43.8%), versus 3.4% for the unselected feature-expansion models on the same folds.

The leak affects three things:
1. CV-selected ("honest") picks and the 35 -> 28 -> 27 decomposition.
2. The CV-vs-test gap numbers.
3. Conventional members' OOF predictions, which use the same folds, so ensemble weighting.

Test-set metrics and leaderboard ranks are unaffected.

## Fix

Refit the selector inside each CV fold, keeping the same folds (`--cv-folds`, the same splitter):

- For each fold: fit the selector on fold-train, transform fold-train and fold-val, fit the model,
  then score fold-val. That replaces `cross_validate(model, X_train_selected, ...)` (runner ~L5605)
  with a manual loop. The deployed model is still the full-train selector plus a full-train fit, so
  **test metrics do not change**.
- Cache the per-fold selections as `<dataset>/nested_selection/fold_<k>.pkl`, keyed by the
  stage 2/3 signature plus fold index, so the step resumes.
- Write the fold predictions as the new `split="oof"` rows for conventional members. Ensembles are
  then rebuilt CPU-only; the Chemprop and Uni-Mol OOF already exist.
- Add a `--cv-selection {outer,nested}` flag that defaults to `outer`, so old runs reproduce.

## Costs

| Item | Estimate | Basis |
|---|---|---|
| Selector refits, 5 folds x 44 | ~18 h on 32 cores (A100 box) | observed median 295 s per full fit (canonical run), x5 folds, x0.84 for 80% fold size |
| The same on this RTX box (16 cores) | **>60 h** | the six largest sets (hERG-Karim, CYP x5) already hit the 7,200 s ElasticNetCV timeout here and fall back to RF; 6 x 5 x 2 h |
| Conventional model fold fits | a few CPU-h | same work as today's CV, minus caching |
| TabPFN CV re-run | **Prior Labs API credits** | the runner prefers `tabpfn_client` (capped API) when `PRIORLABS_API_KEY` is set; exclude it, or ask first |
| Ensemble rebuild + render | ~2-3 CPU-h | folds cached; same as the 2026-10-01 refresh |

A secondary problem: the fallback after a timeout makes nested selection non-reproducible across
machines (the trap already seen with `stage23_resume_cache.pkl`). Running it on a machine where
ElasticNetCV never times out (the A100: max 1,003 s) avoids that.

## Options

1. **Pilot (recommended first, about 2-4 h on this box, no API use):** nested CV for the
   conventional models on the 25 datasets with fewer than 1,500 training rows. It reports how much
   the CV overstatement shrinks and whether any CV-selected pick changes. It decides whether the full
   run is worth it. If the overstatement falls to the arm's ~3-4% and no picks change, the caveat
   plus the pilot result may suffice for the paper.
2. **Full run on an A100 / 32-core box (~18 h + ~3 h rebuild):** replaces the caveat with corrected
   numbers. TabPFN stays on the outer protocol unless credits are approved, and the paper says so.
3. **Cheaper selector inside folds** (for example a fixed-alpha ElasticNet): fast, but the CV would
   then measure a different pipeline from the one deployed. Not recommended.

## Acceptance checks

- Test metrics for every base model are bit-identical before and after (`tools/verify_oof_ensemble_run.py`).
- On the feature-expansion arm's folds, nested CV overstatement for the conventional models should
  approach the arm's ~3-4%; if it stays near 15%, the leak was not the main cause, so stop and report.
- `verify_manuscript_numbers.py` values move only via the render. Never loosen a check.
