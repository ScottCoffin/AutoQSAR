# Changelog

All notable changes to QSARena. Entries are grouped by the work-order item they implement
(`submission/QSARena_IDE_Agent_Work_Order_Manuscript_Code.docx`, v4 / build 26037).

## [Unreleased]

### Added
- **Applicability domain (C2).** `qsarena.applicability_domain`: the Roy-Kar-Ambure (2015)
  descriptor standardization approach and a kNN Tanimoto-distance domain with a training-only
  leave-one-out threshold. The `qsarena-applicability-domain` CLI now reports the standardization
  flag and counts it in its consensus label; `main()` accepts `argv` so it can be tested in-process.
- **Uncertainty and calibration (C5).** `qsarena.uncertainty`: split-conformal regression intervals
  (absolute or difficulty-normalised), LAC conformal prediction sets for classifiers, the
  probability-confidence reliability flag, ECE, Brier score and reliability-diagram bins.
- **Reliability study on the 22 official TDC splits (C2/C5/C9).** `python -m qsarena.reliability_study`
  fits one fixed reference model (random forest on Morgan + RDKit 2D) per dataset and writes
  `results/reliability_tdc22/`: `applicability_domain.csv`, `calibration.csv`,
  `per_molecule_predictions.csv` (with AD flags and conformal intervals per molecule),
  `feature_importance_top10.csv`, `split_hashes.csv`, reliability diagrams, a sample QMRF-style
  report and an environment manifest. Deterministic for a fixed seed.
- **Per-model interpretability (C9).** `qsarena.interpretability`: normalised native or permutation
  importances, and an explicit "no per-feature attribution" status for graph/3D models.
- **QMRF-style report export (C6).** `qsarena.qmrf`: five OECD-principle sections populated from a
  run; raises `QMRFInputError` naming every missing input instead of writing "N/A".
- **Environment manifest (C7).** `qsarena.provenance.write_environment_manifest`: key package
  versions, full installed-distribution list, git commit/dirty flag and hardware. The benchmark
  runner now writes `environment_manifest.json` next to `run_config.json` (non-fatal on failure).
- **ADMET-AI parity table (C4, tooling only).** `python -m qsarena.admet_ai_parity` builds
  `results/admet_ai_parity.csv`, one row per TDC dataset with explicit NA reasons. Against the
  canonical run the ADMET-AI-equivalent variant is valid on 2 of 22 TDC datasets and overlaps the
  captured Chemprop-RDKit leaderboard values on none, so parity remains untested until the
  team-owned Chemprop fix (C1) is re-run.
- **Independent TDC-22 multi-seed stage (C2, tooling only).** `qsarena-benchmark --tdc22-multiseed`
  now reads selected models from an existing source run, writes all seed artifacts under a new
  `--output-dir`, reuses completed per-seed `metrics.csv` files on resume, and writes the compact
  mean/SD table requested by the work order to `results/tdc22_multiseed.csv` by default. Unit tests
  cover the summary table, resume path, and independent entry point; the expensive 5-seed x 22-run
  replication itself has not been run.
- **Thin work-order guard tests (C1/C4).** Added tests that the full Chemprop configuration still
  enumerates all five variants, including the ADMET-AI-equivalent `D-MPNN + RDKit2D`, and that
  ensemble member filtering uses the train split when `member_selection_split="train"`.
- **Tests and CI.** `tests/unit` and `tests/integration` (pytest; markers `gpu`, `slow`), a
  48-molecule fixture from the official TDC splits, and `.github/workflows/ci.yml` (pytest on
  Python 3.10-3.12, ruff, and a `proof.tex` LaTeX build that fails on undefined references).
  `pip install -e .[dev]` installs the tooling.

### Changed
- **Rename completeness (C8).** Removed remaining AutoQSAR residue from user-facing code: the Colab
  notebook builder (title strings, kernel name, Drive output folder, core-download URL), conda
  environment names (`qsarena-py311*`, matching the already-updated docs), issue templates,
  ignore files and the workflow-map title. `containers/autoqsar.def` → `containers/qsarena.def`
  (CONTAINER.md already referred to the new name). `tests/unit/test_rename_residue.py` fails on any
  new occurrence outside a documented allowlist (Schrödinger citations, the canonical run's
  directory name, historical documents).
- The notebook's Colab badge and raw workflow-core URL pointed at a `master` branch that does not
  exist; both now point at `main`.
- `.gitignore` covers the runner's current `.qsarena_tmp/` directory.
- Ruff import-order and unused-import fixes in `qsarena/`.

### Manuscript
- **M3.** Corrected MaxQsaring author given names in `references.bib` (verified against Crossref);
  Uni-Mol2 and Chemprop v2 are now actually cited where those versions are named; added MetaQSAR,
  the OECD (Q)SAR validation principles, the OECD (Q)SAR Assessment Framework and Roy et al. (2015).
- **M2.** MetaQSAR paragraph in §3.12 and a Table 6 row, from the chapter itself (the work order's
  suggested algorithm list described Schrödinger AutoQSAR, not MetaQSAR).
- **M1.** New §3.13 mapping QSARena to the five OECD principles, with applicability-domain and
  calibration numbers taken from `results/reliability_tdc22/summary.json`; added Table S8 to
  Additional file 1 and verifier checks for both manuscript formats.
- **M5.** Removed the proof-build banner from `proof.tex`; fixed the runner filename in the
  data-availability declaration.

### Not done (blocked or out of scope)
- C1/C3 (team-owned Chemprop and ensemble fixes) and everything that depends on their re-runs:
  C4 parity numbers, M4 coverage/ensemble propagation.
- Multi-seed TDC replication: the independent/resumable runner path is implemented, but the full
  5-seed TDC-22 job has not been run; the manuscript's single-seed limitation stands.
- Zenodo DOI (C7/M3): needs a Zenodo login; placeholder remains.
