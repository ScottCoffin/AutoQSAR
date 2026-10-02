"""Assert that every number quoted in manuscript.md matches manuscript_assets/manuscript_numbers.json.

Run from the repository root, after `render_manuscript_assets.py`:

    python portable_colab_qsar_bundle/verify_manuscript_numbers.py

Exits non-zero if any claim drifts from the artifacts, if a referenced figure is missing, or if a
<!-- TABLE:stem --> block in manuscript.md was never filled. When a benchmark is rerun, expect failures:
fix the manuscript prose to match the new artifacts, then update the expected values here.

Expected values below correspond to the out-of-fold ensemble rebuild,
benchmark_results/qsarena_benchmark_oof_ensemble. Timing values remain anchored to the
canonical full A100 run because the OOF repair run reused seeded/cached base-model artifacts.
"""

import csv
import io
import json
import pathlib
import re
import sys

d = json.load(open("manuscript_assets/manuscript_numbers.json"))
rel = json.loads(pathlib.Path("results/reliability_tdc22/summary.json").read_text(encoding="utf-8"))
t = pathlib.Path("manuscript.md").read_text(encoding="utf-8")
body = pathlib.Path("submission/body.tex").read_text(encoding="utf-8")
ok, bad = [], []


def chk(label, cond, detail=""):
    (ok if cond else bad).append(f"{label} {detail}")


def norm_text(s):
    return re.sub(r"\s+", " ", s.replace("\\%", "%").replace("\\,", "").replace("~", " "))


L, W, F, C, FF = d["leaderboard"], d["wins_by_family"], d["family_consistency"], d["cost"], d["feature_families"]

# ---- coverage -------------------------------------------------------------------------------
chk("run is OOF rebuild", d["run_dir"] == "qsarena_benchmark_oof_ensemble" and d["benchmark_profile"] == "full")
chk("coverage 44/44/44", (d["datasets_with_run_status"], d["datasets_completed"], d["datasets_analyzed"]) == (44, 44, 44))
chk("tasks 22/22", d["datasets_by_task"] == {"regression": 22, "classification": 22})
chk("suites", d["datasets_by_suite"] == {"TDC": 32, "Polaris": 5, "MoleculeNet": 3, "ChemML": 2, "PODUAM": 2})
chk("models 30 rows 1047", (d["models_with_valid_results"], d["valid_model_dataset_rows"]) == (30, 1047))
chk("size 280/1605/13445/156052",
    (d["dataset_size"]["min"], round(d["dataset_size"]["median"]), d["dataset_size"]["max"], d["dataset_size"]["total"]) == (280, 1604, 13445, 156052))
chk("splits", d["split_counts"] == {"predefined": 27, "scaffold": 12, "target_quartiles": 4, "random": 1})
chk("incomplete datasets", set(d["incomplete_datasets"]) == set())
chk("no multiseed", d["multiseed_artifacts_present"] is False)

# ---- wins -----------------------------------------------------------------------------------
for fam, tot, r, c in [
    ("Ensemble (stacking / averaging)", 13, 6, 7),
    ("Chemprop v2 GNN", 8, 3, 5),
    ("Uni-Mol (3D pretrained)", 8, 4, 4),
    ("Conventional ML", 4, 1, 3),
    ("TabPFN (tabular foundation)", 5, 4, 1),
    ("MapLight + GNN", 4, 4, 0),
    ("CFA combinatorial fusion", 2, 0, 2),
    ("Deep tabular NN (ChemML MLP)", 0, 0, 0),
]:
    chk(f"wins {fam}", (W[fam]["total"], W[fam]["regression"], W[fam]["classification"]) == (tot, r, c), str(W.get(fam)))
chk("wins sum to 44", sum(v["total"] for v in W.values()) == 44)
chk("largest share 30%", round(100 * max(v["total"] for v in W.values()) / 44) == 30)
chk("3D wins 4 classification", W["Uni-Mol (3D pretrained)"]["classification"] == 4)

# ---- consistency ----------------------------------------------------------------------------
for fam, pct, rank in [
    ("Ensemble (stacking / averaging)", 61, 3.0),
    ("Uni-Mol (3D pretrained)", 57, 4.0),
    ("Conventional ML", 59, 4.0),
    ("Chemprop v2 GNN", 55, 6.0),
    ("CFA combinatorial fusion", 42, 6.0),
    ("TabPFN (tabular foundation)", 30, 10.0),
    ("MapLight + GNN", 27, 12.0),
    ("Deep tabular NN (ChemML MLP)", 18, 16.0),
]:
    v = F[fam]
    chk(f"consistency {fam}",
        round(v["Within 5% of best (% of datasets)"]) == pct and v["Median rank of family-best model"] == rank,
        f"{v['Within 5% of best (% of datasets)']:.1f} r{v['Median rank of family-best model']}")
chk("chemprop 42 datasets", F["Chemprop v2 GNN"]["Datasets with valid results"] == 42)

# ---- leaderboard ----------------------------------------------------------------------------
chk("lb 37/430/35/5/med3",
    (L["datasets_compared"], L["reference_rows"], L["top10_test_selected"], L["rank1_test_selected"], L["median_rank_test_selected"]) == (37, 430, 35, 5, 3.0))
chk("cv 27/1/med7", (L["top10_cv_selected"], L["rank1_cv_selected"], L["median_rank_cv_selected"]) == (27, 1, 7.0))
# Matched-candidate-set control: holds the pool at the CV-eligible models and selects on test.
# It decomposes the 35->27 drop into library breadth (35->28) and honest selection (28->27).
chk("matched pool 28/3/med6", (L["top10_matched_pool"], L["rank1_matched_pool"], L["median_rank_matched_pool"]) == (28, 3, 6.0))
chk("gap decomposition 7+1", (L["top10_test_selected"] - L["top10_matched_pool"],
                             L["top10_matched_pool"] - L["top10_cv_selected"]) == (7, 1))
chk("below top10", set(L["below_top10_datasets"]) == {"tdc_skin_reaction", "tdc_tox21"})
chk("rank1 names", set(L["rank1_datasets"]) == {
    "lipophilicity", "tdc_bioavailability_ma", "tdc_carcinogens_lagunin", "tdc_clearance_microsome_az",
    "tdc_toxcast"})
chk("95% top-10", round(100 * L["top10_test_selected"] / L["datasets_compared"]) == 95)
B = d["leaderboard_by_comparability"]
o = B["tdc_admet_group_official"]
chk("tdc official 22/22, 2 firsts, cv 16",
    (o["datasets"], o["top10_test_selected"], o["median_rank_test_selected"], len(o["rank1_test_selected"]), o["top10_cv_selected"], o["median_rank_cv_selected"]) == (22, 22, 3.0, 2, 16, 7.0))
chk("polaris 5/5", (B["polaris_official"]["datasets"], B["polaris_official"]["top10_test_selected"]) == (5, 5))
chk("local 10, 8 top10", (B["local_split"]["datasets"], B["local_split"]["top10_test_selected"]) == (10, 8))
chk("cv all: 5 winners, 7.9%",
    (d["cv_selected_all_datasets"]["is_overall_winner"], round(d["cv_selected_all_datasets"]["median_relative_gap_to_test_best_pct"], 1)) == (5, 7.9))

# ---- fusion / ablation ----------------------------------------------------------------------
fv = d["fusion_vs_best_single"]
chk("fusion cls 9/22 +2.0", (fv["classification"]["datasets"], fv["classification"]["fusion_better"], round(fv["classification"]["median_rel_improvement_when_better_pct"], 1)) == (22, 9, 2.0))
chk("fusion reg 6/22 +1.8/-2.6", (fv["regression"]["datasets"], fv["regression"]["fusion_better"], round(fv["regression"]["median_rel_improvement_when_better_pct"], 1), round(fv["regression"]["median_rel_improvement_all_pct"], 1)) == (22, 6, 1.8, -2.6))
chk("ablation 10/31/3/13", [r["datasets_improved_vs_previous"] for r in d["component_ablation"]] == [0, 10, 31, 3, 13])

# ---- features --------------------------------------------------------------------------------
chk("features 12137/46.2/22.9", (FF["selected_features_total"], round(FF["maplight_classic_share_selected_pct"], 1), round(FF["maplight_classic_share_available_pct"], 1)) == (12137, 46.2, 22.9))
pf = FF["per_family"]
for fam, e in [("rdkit", 5.16), ("erg", 2.86), ("avalon", 2.54), ("maccs", 1.18), ("fcfp6", 0.73)]:
    chk(f"enrichment {fam}", round(pf[fam]["enrichment_vs_uniform"], 2) == e, f"{pf[fam]['enrichment_vs_uniform']:.3f}")
chk("avalon share 29.7", round(pf["avalon"]["share_selected_pct"], 1) == 29.7)

# ---- cost ------------------------------------------------------------------------------------
chk("cost totals", (round(C["total_recorded_wall_clock_hours"], 1), round(C["median_dataset_wall_clock_hours"], 2), round(C["max_dataset_wall_clock_hours"], 1), C["max_dataset"]) == (111.8, 1.89, 11.2, "tdc_herg_karim"))
m = C["per_family_median_own_seconds"]
for fam, v, nd in [("CFA combinatorial fusion", 0.3, 1), ("Ensemble (stacking / averaging)", 0.6, 1),
                   ("Conventional ML", 6.8, 1), ("Deep tabular NN (ChemML MLP)", 39.8, 1),
                   ("MapLight + GNN", 137, 0), ("Chemprop v2 GNN", 269, 0), ("Uni-Mol (3D pretrained)", 374, 0)]:
    chk(f"cost {fam}", round(m[fam], nd) == v, f"{m[fam]:.2f}")
chk("unimol 55x conventional", round(m["Uni-Mol (3D pretrained)"] / m["Conventional ML"]) == 55)
S = d["selector_scaling"]
chk("selector 0.77/0.58/295/1003", (round(S["log10_slope"], 2), round(S["pearson_r"], 2), round(S["median_selector_seconds"]), round(S["max_selector_seconds"]), S["max_selector_dataset"]) == (0.77, 0.58, 295, 1003, "tdc_herg_karim"))

# ---- hardware comparison ---------------------------------------------------------------------
R = d["run_comparison_vs_rtx"]
chk("run comparison 44/37", (R["datasets_compared"], R["datasets_same_split"]) == (44, 37))
chk("run comparison median 0.30", round(R["median_change_pct_same_split"], 2) == 0.30)
chk("run comparison 21 vs 15", (R["a100_better_same_split"], R["rtx_better_same_split"]) == (21, 15))
chk("7 datasets re-split", len(R["datasets_respilt_to_scaffold"]) == 7)

# ---- provenance --------------------------------------------------------------------------------
chk("repro commit bbfb188", d["reproducibility"]["run_artifacts_committed_in"] == "bbfb188" and d["reproducibility"]["random_seed"] == 13)

# ---- regulatory reliability study --------------------------------------------------------------
Sstd, Sknn, Scons, Srel, Sconf = (rel["standardization"], rel["knn_tanimoto"], rel["consensus"],
                                  rel["reliability"], rel["conformal"])
chk("reliability study 22 official",
    (rel["n_datasets"], rel["n_regression"], rel["n_classification"], rel["seed"]) == (22, 9, 13, 0))
chk("reliability reference model",
    rel["reference_model"] == "RandomForest (300 trees) on Morgan r=2 2048-bit + RDKit 2D descriptors")
chk("reliability structural summary",
    (round(100 * Sstd["median_coverage"], 1), round(100 * Sstd["min_coverage"], 1),
     round(100 * Sstd["max_coverage"], 1), Sstd["n_datasets_out_error_higher"],
     Sstd["n_datasets_with_both_groups"], round(Sstd["median_pct_higher_error_out"], 1),
     round(Sstd["median_pct_higher_error_out_regression"], 1)) == (95.7, 92.6, 99.5, 8, 17, -8.6, 34.8))
chk("reliability knn consensus summary",
    (round(100 * Sknn["median_coverage"], 1), Sknn["n_datasets_out_error_higher"],
     Sknn["n_datasets_with_both_groups"], round(Sknn["median_pct_higher_error_out"], 1),
     round(100 * Scons["median_coverage"], 1), Scons["n_datasets_out_error_higher"],
     round(Scons["median_pct_higher_error_out"], 1)) == (89.7, 13, 21, 15.3, 87.0, 14, 7.9))
chk("reliability confidence summary",
    (round(100 * Srel["median_coverage"], 1), Srel["n_datasets_out_error_higher"],
     round(Srel["median_error_ratio_out_in"], 2), round(Srel["median_pct_higher_error_out"], 1),
     round(Srel["median_pct_higher_error_out_regression"], 1),
     round(Srel["median_pct_higher_error_out_classification"], 1)) == (51.2, 22, 3.30, 230.3, 108.0, 382.1))
chk("reliability conformal summary",
    (round(100 * Sconf["regression_median_coverage"], 1), round(100 * Sconf["classification_median_coverage"], 1),
     round(Sconf["classification_median_ece"], 3), round(Sconf["classification_median_brier"], 3)) == (93.2, 90.9, 0.065, 0.137))
for label, text in [("md", t), ("body", body)]:
    nt = norm_text(text)
    chk(f"sec oecd {label}", ("3.13 Regulatory alignment with the OECD" in nt if label == "md" else "label{sec:oecd}" in text))
    chk(f"oecd numbers {label}",
        all(s in nt for s in ["95.7%", "8 of 17", "89.7%", "13 of 21", "87.0%", "14 of 22",
                              "51.2%", "230.3%", "93.2%", "90.9%", "0.065", "0.137"]))
    chk(f"reference model caveat {label}", "not the per-dataset selected QSARena model" in nt)

# ---- dataset-property meta-analysis (qsarena.meta_analysis; section 3.14) ---------------------
# The 3.14 prose and the meta limitation paragraph are rendered from meta_numbers.json into
# META blocks in both formats; re-render them and fail on any drift in either direction.
meta_path = pathlib.Path("manuscript_assets/meta_numbers.json")
chk("meta_numbers.json present", meta_path.exists())
if meta_path.exists():
    sys.path.insert(0, str(pathlib.Path(".").resolve()))
    from qsarena.meta_analysis.text import BLOCKS, check_manuscript, unresolved

    M = json.loads(meta_path.read_text(encoding="utf-8"))
    chk("meta run matches manuscript run", M["run_dir"] == d["run_dir"])
    chk("meta 44 datasets", M["n_datasets"] == 44 == d["datasets_analyzed"])
    chk("meta macros resolve", all(not unresolved(tpl, M["macros"]) for tpl in BLOCKS.values()))
    for problem in check_manuscript(M):
        chk("meta block", False, problem)
    # Cross-check the meta-analysis against the notebook's own family statistics (Table 3, Fig 2).
    for fam, pct in M["within5_pct_by_family"].items():
        if fam in F:
            chk(f"meta within-5% {fam}", abs(pct - F[fam]["Within 5% of best (% of datasets)"]) < 0.05, f"{pct:.1f}")
    chk("meta chemprop valid == table3",
        M["chemprop_valid_datasets"] == F["Chemprop v2 GNN"]["Datasets with valid results"])
    chk("meta winner groups sum to 44", sum(M["winner_groups"].values()) == 44)
    chk("meta figures 7-9 referenced",
        all(f"manuscript_assets/figures/{s}.png" in t
            for s in ("figureM1_size_crossover", "figureM2_shift_difficulty", "figureM3_recommender")))

# ---- tables agree with the JSON ----------------------------------------------------------------
t2 = list(csv.DictReader(io.StringIO(pathlib.Path("manuscript_assets/tables/table2_dataset_catalog.csv").read_text(encoding="utf-8"))))
chk("table2 has leaderboard columns", {"Est. rank", "Best published", "Leaderboard metric", "Best published model"} <= set(t2[0].keys()))
chk("table2 rows == datasets", len(t2) == d["datasets_analyzed"])
chk("table2 ranks populated", sum(1 for r in t2 if str(r["Est. rank"]).strip()) == L["datasets_compared"])
t6 = {r["Model family"]: r for r in csv.DictReader(io.StringIO(pathlib.Path("manuscript_assets/tables/table6_cost.csv").read_text(encoding="utf-8")))}
chk("table6 has published comparators", any("published" in k for k in t6))

print(f"PASS {len(ok)} checks")
for b in bad:
    print("FAIL", b)
figs = re.findall(r"\]\((manuscript_assets/figures/[^)]+)\)", t)
missing_figs = [f for f in figs if not pathlib.Path(f).exists()]
print("figures referenced:", len(figs), "missing:", missing_figs)
unfilled = re.findall(r"<!-- TABLE:(\w+) -->\s*<!-- /TABLE -->", t)
print("unfilled table blocks:", unfilled)
sys.exit(1 if (bad or missing_figs or unfilled) else 0)
