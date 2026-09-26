# Work-order progress (handoff) - 2026-09-25

Implementing `submission/QSARena_IDE_Agent_Work_Order_Manuscript_Code.docx`. Nothing is committed yet
(user has not asked). Full detail of what changed is in `CHANGELOG.md`.

## Completed before pickup
- New modules: `qsarena/{applicability_domain,uncertainty,interpretability,qmrf,provenance,
  reliability_study,reliability_tables,admet_ai_parity}.py`.
- AD CLI (`simple_applicability_domain.py`) includes Roy standardization; `main(argv)`.
- Runner writes `environment_manifest.json` after `run_config.json` (non-fatal).
- Tests in `tests/unit`, `tests/integration`, fixture `tests/fixtures/tdc_tiny`; pytest/ruff config +
  `[dev]` extra in `pyproject.toml`; `.github/workflows/ci.yml` builds `proof.tex`.
- C8 rename sweep + guard test `tests/unit/test_rename_residue.py`; Colab badge/core URL now points at
  `main` and `QSARena`, so the GitHub repo rename is required for those links.
- `results/admet_ai_parity.csv`: ADMET-AI-equivalent variant valid on 2/22 TDC, 0 comparable; C4 blocked on C1.
- Manuscript positioning and references: MaxQsaring authors fixed, Uni-Mol2 and Chemprop v2 cited,
  MetaQSAR paragraph + Table 6 row added, refs 57-60 added, stale title quote fixed, proof banner
  removed, declarations runner filename fixed, Additional file 1 title fixed.

## Completed in pickup
- Resummarized the completed reliability study with the current `MIN_GROUP=5` rule:
  `python -m qsarena.reliability_study --out results/reliability_tdc22 --resummarize`.
- Regenerated Table S8:
  `python -m qsarena.reliability_tables`.
- Added Section 3.13 "Regulatory alignment with the OECD (Q)SAR principles" in both
  `manuscript.md` and `submission/body.tex`, using only `results/reliability_tdc22/summary.json`
  for the reliability numbers.
- Revised the Section 4 applicability-domain/calibration limitation in both manuscript formats.
- Added Table S8 to `submission/additional_file_1.tex`, made it landscape in the supplementary PDF,
  and appended the Markdown Table S8 copy under `## Supplementary tables`.
- Added verifier checks for the Section 3.13 numbers in both `manuscript.md` and `submission/body.tex`.
- Updated `CHANGELOG.md`, `TODO.md`, `submission/README.md`, and the supplementary-file description.

## Completed after updated v4 work order
- Implemented the independent/resumable TDC-22 multi-seed runner path:
  `qsarena-benchmark --tdc22-multiseed --tdc22-multiseed-source-run <existing-run> --output-dir <new-run>`.
  It reads selected models from an existing source run, reuses completed per-seed `metrics.csv` files
  when `--resume` is on, and writes a compact mean/SD CSV to `results/tdc22_multiseed.csv` by default.
- Refactored the end-of-run TDC-22 multi-seed summary into
  `summarize_tdc22_multiseed_metrics`, adding seed count, mean/SD/min/max, single-seed value,
  within-1-SD flag, metric direction, timing, selected family/workflow, and selection-note columns.
- Added targeted tests for the summary table, 2-seed kill/resume behavior, and independent CSV export:
  `tests/unit/test_tdc22_multiseed.py`.
- Added thin integration guard tests for team-owned work-order items: Chemprop still enumerates all
  five configured variants, and ensemble member filtering uses the train split when requested:
  `tests/unit/test_work_order_guards.py`.
- Updated `CHANGELOG.md`, `TODO.md`, `AGENTS.md`, and this handoff note for the v4 work order.

## Current validation
- `python portable_colab_qsar_bundle/verify_manuscript_numbers.py` -> PASS 78 checks.
- `cd submission && pdflatex proof && bibtex proof && pdflatex proof && pdflatex proof` -> no undefined
  refs/citations in `proof.log`.
- `pdflatex additional_file_1` -> builds `additional_file_1.pdf`; remaining overfull warnings are from
  existing dense supplementary tables, not the new S8 landscape page.
- `python -m pytest -q -m "not gpu and not slow"` -> 56 passing.
- `ruff check qsarena tests portable_colab_qsar_bundle/verify_manuscript_numbers.py` -> clean.
- `python -m pytest -q tests/unit/test_tdc22_multiseed.py tests/unit/test_work_order_guards.py` -> 6 passing.

## Remaining
- Blocked or not locally runnable: C1 (team Chemprop fix/re-run), C5/M4 propagation after Chemprop
  and ensemble re-runs, actual 5-seed TDC-22 compute run, Zenodo DOI, GitHub repo rename, ORCID and
  acknowledgement placeholders.
