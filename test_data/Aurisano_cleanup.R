# Exact extraction of modeled PODUAM STD datasets used by run_cv_BNN.py:
# - run_cv_BNN.py loads: ../PODUAM/features/xy_pod_{effect}-std_{feat}.csv
# - run_feature_calculation.py defines POD_logmol for STD as:
#     y_std = df_std["y"] where df_std comes from data_pod_{effect}-std.csv
# Therefore, the authoritative SMILES + target pair is:
#   Canonical_QSARr + y (renamed to POD_logmol) from data_pod_{effect}-std.csv

library(tidyverse)

base_raw <- "https://raw.githubusercontent.com/kejbo/PODUAM/main"

build_modeled_subset <- function(effect) {
    data_url <- paste0(base_raw, "/data/data_pod_", effect, "-std.csv")
    feat_url <- paste0(base_raw, "/features/xy_pod_", effect, "-std_rdkit.csv")

    data_std <- read_csv(data_url, show_col_types = FALSE) |>
        select(ID, QSAR_READY_SMILES = Canonical_QSARr, POD_logmol = y)

    # Sanity-check against the exact ID set used by run_cv_BNN.py
    feat_std <- read_csv(feat_url, show_col_types = FALSE) |>
        select(ID, POD_logmol_feat = POD_logmol)

    out <- data_std |>
        inner_join(feat_std, by = "ID") |>
        mutate(pod_delta = abs(POD_logmol - POD_logmol_feat))

    if (nrow(out) != nrow(feat_std)) {
        stop("ID mismatch with modeled feature file for effect: ", effect)
    }
    if (any(out$pod_delta > 1e-12, na.rm = TRUE)) {
        stop(
            "POD_logmol mismatch between data_pod and xy_pod files for effect: ",
            effect
        )
    }

    out |>
        select(QSAR_READY_SMILES, POD_logmol)
}

merged_nc <- build_modeled_subset("nc")
merged_rd <- build_modeled_subset("rd")

# Optional exports for benchmark ingestion
write_csv(merged_nc, "test_data/poduam_nc_std_qsar_smiles_pod_logmol.csv")
write_csv(merged_rd, "test_data/poduam_rd_std_qsar_smiles_pod_logmol.csv")

message("Rows (nc): ", nrow(merged_nc))
message("Rows (rd): ", nrow(merged_rd))
