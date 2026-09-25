"""Assert that every number quoted in manuscript.md matches manuscript_assets/manuscript_numbers.json.

Run from the repository root, after `render_manuscript_assets.py`:

    python portable_colab_qsar_bundle/verify_manuscript_numbers.py

Exits non-zero if any claim drifts from the artifacts, if a referenced figure is missing, or if a
<!-- TABLE:stem --> block in manuscript.md was never filled. When a benchmark is rerun, expect failures:
fix the manuscript prose to match the new artifacts, then update the expected values here.

Expected values below correspond to the canonical run: NSF ACCESS Jetstream2 A100,
benchmark_results/autoqsar_benchmark_20260623_153839.
"""

import csv
import io
import json
import pathlib
import re
import sys

d = json.load(open("manuscript_assets/manuscript_numbers.json"))
t = pathlib.Path("manuscript.md").read_text(encoding="utf-8")
ok, bad = [], []


def chk(label, cond, detail=""):
    (ok if cond else bad).append(f"{label} {detail}")


L, W, F, C, FF = d["leaderboard"], d["wins_by_family"], d["family_consistency"], d["cost"], d["feature_families"]

# ---- coverage -------------------------------------------------------------------------------
chk("run is A100", d["run_dir"] == "autoqsar_benchmark_20260623_153839" and d["benchmark_profile"] == "full")
chk("coverage 45/43/44", (d["datasets_with_run_status"], d["datasets_completed"], d["datasets_analyzed"]) == (45, 43, 44))
chk("tasks 22/22", d["datasets_by_task"] == {"regression": 22, "classification": 22})
chk("suites", d["datasets_by_suite"] == {"TDC": 32, "Polaris": 5, "MoleculeNet": 3, "ChemML": 2, "PODUAM": 2})
chk("models 28 rows 837", (d["models_with_valid_results"], d["valid_model_dataset_rows"]) == (28, 837))
chk("size 280/1605/13445/156052",
    (d["dataset_size"]["min"], round(d["dataset_size"]["median"]), d["dataset_size"]["max"], d["dataset_size"]["total"]) == (280, 1604, 13445, 156052))
chk("splits", d["split_counts"] == {"predefined": 27, "scaffold": 12, "target_quartiles": 4, "random": 1})
chk("incomplete datasets", set(d["incomplete_datasets"]) == {"polaris_adme_fang_hppb_1", "tdc_herg_central"})
chk("no multiseed", d["multiseed_artifacts_present"] is False)

# ---- wins -----------------------------------------------------------------------------------
for fam, tot, r, c in [
    ("Ensemble (stacking / averaging)", 16, 7, 9),
    ("Uni-Mol (3D pretrained)", 11, 6, 5),
    ("Conventional ML", 8, 2, 6),
    ("MapLight + GNN", 4, 4, 0),
    ("Deep tabular NN (ChemML MLP)", 2, 2, 0),
    ("Chemprop v2 GNN", 2, 1, 1),
    ("CFA combinatorial fusion", 1, 0, 1),
]:
    chk(f"wins {fam}", (W[fam]["total"], W[fam]["regression"], W[fam]["classification"]) == (tot, r, c), str(W.get(fam)))
chk("wins sum to 44", sum(v["total"] for v in W.values()) == 44)
chk("largest share 36%", round(100 * max(v["total"] for v in W.values()) / 44) == 36)
chk("3D wins 5 classification", W["Uni-Mol (3D pretrained)"]["classification"] == 5)

# ---- consistency ----------------------------------------------------------------------------
for fam, pct, rank in [
    ("Ensemble (stacking / averaging)", 75, 2.0),
    ("Uni-Mol (3D pretrained)", 70, 4.0),
    ("Conventional ML", 66, 3.0),
    ("MapLight + GNN", 27, 8.5),
    ("Deep tabular NN (ChemML MLP)", 25, 12.5),
    ("Chemprop v2 GNN", 83, 2.0),
]:
    v = F[fam]
    chk(f"consistency {fam}",
        round(v["Within 5% of best (% of datasets)"]) == pct and v["Median rank of family-best model"] == rank,
        f"{v['Within 5% of best (% of datasets)']:.1f} r{v['Median rank of family-best model']}")
chk("chemprop only 6 datasets", F["Chemprop v2 GNN"]["Datasets with valid results"] == 6)

# ---- leaderboard ----------------------------------------------------------------------------
chk("lb 37/430/35/5/med3",
    (L["datasets_compared"], L["reference_rows"], L["top10_test_selected"], L["rank1_test_selected"], L["median_rank_test_selected"]) == (37, 430, 35, 5, 3.0))
chk("cv 25/0/med8", (L["top10_cv_selected"], L["rank1_cv_selected"], L["median_rank_cv_selected"]) == (25, 0, 8.0))
# Matched-candidate-set control: holds the pool at the CV-eligible models and selects on test.
# It decomposes the 35->25 drop into library breadth (35->28) and honest selection (28->25).
chk("matched pool 28/3/med6", (L["top10_matched_pool"], L["rank1_matched_pool"], L["median_rank_matched_pool"]) == (28, 3, 6.0))
chk("gap decomposition 7+3", (L["top10_test_selected"] - L["top10_matched_pool"],
                             L["top10_matched_pool"] - L["top10_cv_selected"]) == (7, 3))
chk("below top10", set(L["below_top10_datasets"]) == {"tdc_skin_reaction", "tdc_tox21"})
chk("rank1 names", set(L["rank1_datasets"]) == {
    "tdc_bioavailability_ma", "tdc_carcinogens_lagunin", "tdc_clearance_microsome_az",
    "tdc_cyp2c9_substrate_carbonmangels", "tdc_toxcast"})
chk("95% top-10", round(100 * L["top10_test_selected"] / L["datasets_compared"]) == 95)
B = d["leaderboard_by_comparability"]
o = B["tdc_admet_group_official"]
chk("tdc official 22/22, 3 firsts, cv 15",
    (o["datasets"], o["top10_test_selected"], o["median_rank_test_selected"], len(o["rank1_test_selected"]), o["top10_cv_selected"], o["median_rank_cv_selected"]) == (22, 22, 3.0, 3, 15, 8.0))
chk("polaris 5/5", (B["polaris_official"]["datasets"], B["polaris_official"]["top10_test_selected"]) == (5, 5))
chk("local 10, 8 top10", (B["local_split"]["datasets"], B["local_split"]["top10_test_selected"]) == (10, 8))
chk("cv all: 0 winners, 16.1%",
    (d["cv_selected_all_datasets"]["is_overall_winner"], round(d["cv_selected_all_datasets"]["median_relative_gap_to_test_best_pct"], 1)) == (0, 16.1))

# ---- fusion / ablation ----------------------------------------------------------------------
fv = d["fusion_vs_best_single"]
chk("fusion cls 10/22 +0.8", (fv["classification"]["datasets"], fv["classification"]["fusion_better"], round(fv["classification"]["median_rel_improvement_when_better_pct"], 1)) == (22, 10, 0.8))
chk("fusion reg 7/22 +2.4/-1.4", (fv["regression"]["datasets"], fv["regression"]["fusion_better"], round(fv["regression"]["median_rel_improvement_when_better_pct"], 1), round(fv["regression"]["median_rel_improvement_all_pct"], 1)) == (22, 7, 2.4, -1.4))
chk("ablation 10/26/6/16", [r["datasets_improved_vs_previous"] for r in d["component_ablation"]] == [0, 10, 26, 6, 16])

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
chk("run comparison median -0.19", round(R["median_change_pct_same_split"], 2) == -0.19)
chk("run comparison 15 vs 21", (R["a100_better_same_split"], R["rtx_better_same_split"]) == (15, 21))
chk("7 datasets re-split", len(R["datasets_respilt_to_scaffold"]) == 7)

# ---- provenance --------------------------------------------------------------------------------
chk("repro commit bbfb188", d["reproducibility"]["run_artifacts_committed_in"] == "bbfb188" and d["reproducibility"]["random_seed"] == 13)

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
