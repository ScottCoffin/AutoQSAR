# PFAS auxiliary QSAR run artifacts

This folder keeps the lightweight summary tables and diagnostics needed to inspect the PFAS auxiliary QSAR run.

The per-row `predictions.csv` export is intentionally not tracked. It contained 566,327 rows and was about 89 MB, which is below GitHub's hard 100 MB limit but large enough to create repository bloat and GitHub large-file warnings. Regenerate it from the run pipeline when row-level predictions are needed.
