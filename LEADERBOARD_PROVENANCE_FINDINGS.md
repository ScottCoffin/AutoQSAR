# Leaderboard reference provenance — audit findings

All numbers below were computed from committed artifacts on 2026-09-25 and are reproducible from
`data/benchmark_leaderboards/TDC_ADMET_Benchmark_Performance_by_Model.csv` plus the
`leaderboard_top10_json` column of the canonical run's `metrics.csv` files.

**Bottom line: the comparison table the manuscript leans on mixes verified leaderboard positions with
publication self-reports, and reports the self-reports as if they were ranks. Three of them are
mutually impossible, and three values are on the wrong scale.**

---

## 1. Why "MaxQsaring 19/22" and "DeepAutoQSAR 20/22" looked incompatible

They are not incompatible, because they are not the same kind of claim. Neither is a leaderboard rank.

| Claim | What it actually is |
|---|---|
| MaxQsaring "first on 19 of 22" | Its **own 2025 paper's** rank at time of publication. |
| ADMETboost "first on 18 of 22" | Its **own 2022 paper's** rank at time of publication. |
| DeepAutoQSAR "top performer on 20 of 22" | **Best of three** in Schrödinger's own white paper, which compares DeepAutoQSAR against only ChemProp and DeepPurpose. Not a leaderboard position at all. |
| ADMET-AI "highest average rank" | A self-reported **aggregate** across endpoints, not per-endpoint supremacy. |

So MaxQsaring and DeepAutoQSAR were never in competition in those two figures, and the apparent
arithmetic conflict is an artifact of the manuscript presenting all four in one sentence as though
they were commensurable.

The deeper problem is that the curated table **encodes these self-reports in a column named
`TDC_Rank`**, which reads as a leaderboard position:

- **64 rows carry `TDC_Rank = 1` across 28 datasets.** That is impossible for a single leaderboard.
- `caco2_wang` alone has **four** models each recorded as rank 1 (CaliciBoost 0.256, MaxQsaring 0.275,
  MapLight+GNN 0.276, ADMETboost 0.288).

## 2. Re-scoring from the reported values changes the picture completely

Ranking each dataset by `Score_Mean` in the honest direction, on the 22 official TDC tasks:

| Model | Self-reported #1 | **Score-derived #1** | Score-derived top-3 | Median rank |
|---|---|---|---|---|
| MaxQsaring | 19 | **7** | **19** | **2** |
| MolGPS (3B) | 0 | 7 → **4 after scale fix** | 10 | — |
| MiniMol (GINE) | 6 | 3 | 6 | — |
| MolE | 10 | 2 | 7 | — |
| **ADMETboost (XGBoost)** | **18** | **0** | **6** | — |

ADMETboost's 18 first places evaporate entirely. It was genuinely first in 2022 and has since been
overtaken on **every one of the 22 datasets**. This is precisely the staleness problem: the
manuscript currently cites that 18/22 as a live comparison.

**The defensible statement about MaxQsaring is: top-3 on 19 of 22, median rank 2 — the strongest
model in the curated reference set — not "first on 19 of 22."**

## 3. Three curated values are on the wrong scale

Checked against the actual TDC leaderboard values we scraped, these beat the real #1 by margins that
are not physically plausible (MolGPS reports normalised targets):

| Dataset | Metric | Curated MolGPS (3B) | Actual TDC #1 | Implied improvement |
|---|---|---|---|---|
| `ppbr_az` | MAE | 0.679 | 7.440 (Gradient Boost) | 90.9 % |
| `ld50_zhu` | MAE | 0.292 | 0.552 | 47.1 % |
| `vdss_lombardo` | Spearman | 0.942 | 0.713 | 32.1 % |

`AGENTS.md` already noted the `ppbr_az` case. The other two are new. All three should be dropped or
rescaled; they account for 3 of MolGPS's 7 apparent first places.

## 4. Reference coverage is much thinner than the manuscript implies

Of the 44 analysed datasets:

- **24** have any leaderboard reference at all in the run artifacts (not 37 — the notebook reaches 37
  by merging in the curated literature CSVs).
- **9** have a top-10 drawn *entirely* from the actual scraped TDC leaderboard:
  `caco2_wang`, `clearance_hepatocyte_az`, `clearance_microsome_az`, `half_life_obach`, `ld50_zhu`,
  `lipophilicity_astrazeneca`, `ppbr_az`, `solubility_aqsoldb`, `vdss_lombardo`.
- **15** have no actual-TDC rows at all — including `tox21`, `toxcast`, `clintox`,
  `carcinogens_lagunin`, `skin_reaction`, all five Polaris sets and both PODUAM sets.
- The remaining `tdc_*` datasets in the 22-task set are backed by `literature_static` /
  curated rows only.

**None of MaxQsaring, ADMETboost, ADMET-AI, DeepAutoQSAR, Auto-ADMET or QW-MTL appears anywhere in
the scraped actual TDC top-10s.** Every comparison against them rests on their own publications.

## 5. Does the CV re-run change the rankings? Yes — decisively

On the 22 official TDC datasets (from `manuscript_numbers.json`, already correctly metric-matched):

| Protocol | Top-10 | First places | Median rank |
|---|---|---|---|
| Test-selected | 22/22 | 3 | **3** |
| CV-selected | 15/22 | **0** | **8** |

This is the paper's most defensible quantitative finding and it is currently under-sold. Framed
against the reference set: under test-selection QSARena's median rank of 3 sits alongside
MaxQsaring's 2; under honest CV selection it falls to 8, clearly behind. **The model-selection
protocol moves a system by about five leaderboard positions — a larger effect than most of the
differences between the published methods being compared.** That reframes the contribution from
"we are competitive" to "leaderboard position is substantially an artifact of selection protocol,"
which is both more honest and more interesting.

## 6. Required manuscript changes

1. Rename "estimated leaderboard rank" → **"rank among curated published reference values"**, and add
   a provenance flag per dataset (actual TDC scrape vs publication self-report).
2. Report headline placement on the **9 actual-TDC** and **27 benchmark-equivalent** subsets
   separately from the 37-dataset merge. Exclude `tox21`/`toxcast` from headline counts.
3. Fix MaxQsaring to **top-3 on 19/22, median rank 2**; add it to Table 7 with those values.
4. Delete or heavily caveat the ADMETboost "18 of 22" claim, and state that re-scoring gives 0.
5. Label DeepAutoQSAR's 20/22 as **best-of-three in a three-way vendor comparison**, not a rank.
6. Drop the three MolGPS scale-error rows from the reference table.
7. Promote §5 above into the abstract and Conclusion as the headline methodological result.
