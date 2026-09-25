"""Assert that every number quoted in manuscript.md matches manuscript_assets/manuscript_numbers.json.

Run from the repository root, after `render_manuscript_assets.py`:

    python portable_colab_qsar_bundle/verify_manuscript_numbers.py

Exits non-zero if any claim drifts from the artifacts, if a referenced figure is missing, or if a
<!-- TABLE:stem --> block in manuscript.md was never filled. When a benchmark is rerun, expect failures:
fix the manuscript prose to match the new artifacts, then update the expected values here.
"""

import sys

import json, re, pathlib

d = json.load(open('manuscript_assets/manuscript_numbers.json'))
t = pathlib.Path('manuscript.md').read_text(encoding='utf-8')
ok, bad = [], []


def chk(label, cond, detail=''):
    (ok if cond else bad).append(f"{label} {detail}")


L, W, F, C, FF = d['leaderboard'], d['wins_by_family'], d['family_consistency'], d['cost'], d['feature_families']
chk('coverage 46/45', d['datasets_with_run_status'] == 46 and d['datasets_completed'] == 45)
chk('tasks 23/22', d['datasets_by_task'] == {'regression': 23, 'classification': 22})
chk('suites', d['datasets_by_suite'] == {'TDC': 32, 'Polaris': 5, 'ChemML': 3, 'MoleculeNet': 3, 'PODUAM': 2})
chk('models 25 rows 828', d['models_with_valid_results'] == 25 and d['valid_model_dataset_rows'] == 828)
chk('size', (d['dataset_size']['min'], d['dataset_size']['median'], d['dataset_size']['max'], d['dataset_size']['total']) == (38, 1478.0, 13445, 156090))
chk('splits', d['split_counts'] == {'predefined': 27, 'random': 6, 'scaffold': 6, 'target_quartiles': 6})
for fam, tot, r, c in [('Ensemble (stacking / averaging)', 15, 5, 10), ('Conventional ML', 11, 1, 10),
                       ('Uni-Mol V1 (3D pretrained)', 7, 7, 0), ('MapLight + GNN', 4, 4, 0),
                       ('CFA combinatorial fusion', 4, 2, 2), ('TabPFN (tabular foundation)', 3, 3, 0),
                       ('Chemprop v2 GNN', 1, 1, 0), ('Deep tabular NN (ChemML MLP)', 0, 0, 0)]:
    chk(f'wins {fam}', (W[fam]['total'], W[fam]['regression'], W[fam]['classification']) == (tot, r, c), str(W[fam]))
chk('cls 20 of 22', W['Ensemble (stacking / averaging)']['classification'] + W['Conventional ML']['classification'] == 20)
chk('reg sums 23', sum(v['regression'] for v in W.values()) == 23)
for fam, pct, rank in [('Ensemble (stacking / averaging)', 76, 2.0), ('Conventional ML', 69, 2.0),
                       ('Uni-Mol V1 (3D pretrained)', 62, 6.0), ('MapLight + GNN', 26, 11.5),
                       ('Chemprop v2 GNN', 17, 9.0), ('Deep tabular NN (ChemML MLP)', 16, 14.0)]:
    v = F[fam]
    chk(f'consistency {fam}', round(v['Within 5% of best (% of datasets)']) == pct and v['Median rank of family-best model'] == rank,
        f"{v['Within 5% of best (% of datasets)']:.1f} r{v['Median rank of family-best model']}")
chk('ens cls gap 0.03', round(F['Ensemble (stacking / averaging)']['Median gap to best, classification (%)'], 2) == 0.03)
chk('lb 37/430/35/6/3', (L['datasets_compared'], L['reference_rows'], L['top10_test_selected'], L['rank1_test_selected'], L['median_rank_test_selected']) == (37, 430, 35, 6, 3.0))
chk('cv 26/1/7', (L['top10_cv_selected'], L['rank1_cv_selected'], L['median_rank_cv_selected']) == (26, 1, 7.0))
chk('below top10', L['below_top10_datasets'] == ['polaris_adme_fang_solu_1', 'tdc_tox21'])
chk('rank1 names', set(L['rank1_datasets']) == {'tdc_bioavailability_ma', 'tdc_carcinogens_lagunin', 'tdc_cyp2c9_substrate_carbonmangels', 'tdc_hydrationfreeenergy_freesolv', 'tdc_skin_reaction', 'tdc_toxcast'})
B = d['leaderboard_by_comparability']
o = B['tdc_admet_group_official']
chk('tdc official', (o['datasets'], o['top10_test_selected'], o['median_rank_test_selected'], len(o['rank1_test_selected']), o['top10_cv_selected'], o['median_rank_cv_selected']) == (22, 22, 3.0, 2, 15, 8.0))
chk('polaris 4/5', (B['polaris_official']['datasets'], B['polaris_official']['top10_test_selected']) == (5, 4))
chk('local 10, 4 rank1', B['local_split']['datasets'] == 10 and len(B['local_split']['rank1_test_selected']) == 4)
chk('cv all', (d['cv_selected_all_datasets']['is_overall_winner'], round(d['cv_selected_all_datasets']['median_relative_gap_to_test_best_pct'], 1)) == (2, 10.4))
fv = d['fusion_vs_best_single']
chk('fusion cls', (fv['classification']['datasets'], fv['classification']['fusion_better'], round(fv['classification']['median_rel_improvement_when_better_pct'], 2)) == (22, 12, 0.48))
chk('fusion reg', (fv['regression']['datasets'], fv['regression']['fusion_better'], round(fv['regression']['median_rel_improvement_when_better_pct'], 1), round(fv['regression']['median_rel_improvement_all_pct'], 1)) == (23, 7, 3.3, -2.4))
chk('ablation', [r['datasets_improved_vs_previous'] for r in d['component_ablation']] == [0, 7, 26, 8, 15])
chk('features', (FF['selected_features_total'], round(FF['maplight_classic_share_selected_pct'], 1), round(FF['maplight_classic_share_available_pct'], 1)) == (16863, 37.8, 22.9))
pf = FF['per_family']
for fam, e in [('rdkit', 3.90), ('erg', 2.24), ('avalon', 1.99), ('maccs', 1.14), ('morgan', 0.52)]:
    chk(f'enrichment {fam}', round(pf[fam]['enrichment_vs_uniform'], 2) == e, f"{pf[fam]['enrichment_vs_uniform']:.3f}")
chk('avalon share 23', round(pf['avalon']['share_selected_pct']) == 23)
chk('cost totals', (round(C['total_recorded_wall_clock_hours'], 1), round(C['median_dataset_wall_clock_hours'], 2), round(C['max_dataset_wall_clock_hours'], 1), C['max_dataset']) == (155.0, 1.11, 23.6, 'tdc_herg_karim'))
m = C['per_family_median_own_seconds']
for fam, v in [('CFA combinatorial fusion', 0.2), ('Ensemble (stacking / averaging)', 0.5), ('Conventional ML', 17),
               ('Deep tabular NN (ChemML MLP)', 18), ('Chemprop v2 GNN', 98), ('TabPFN (tabular foundation)', 125),
               ('MapLight + GNN', 151), ('Uni-Mol V1 (3D pretrained)', 1964)]:
    chk(f'cost {fam}', round(m[fam], 1 if v < 1 else 0) == v, f"{m[fam]:.1f}")
chk('unimol 115x', round(m['Uni-Mol V1 (3D pretrained)'] / m['Conventional ML']) == 115)
S = d['selector_scaling']
chk('selector', (round(S['log10_slope'], 2), round(S['pearson_r'], 2), round(S['median_selector_seconds']), round(S['max_selector_seconds']), S['max_selector_dataset']) == (1.36, 0.67, 99, 10654, 'tdc_cyp2d6_veith'))
chk('selector 3.0h', round(S['max_selector_seconds'] / 3600, 1) == 3.0)
chk('repro', d['reproducibility']['run_artifacts_committed_in'] == 'b7cd42c' and d['reproducibility']['random_seed'] == 13 and d['reproducibility']['chemprop_seed'] == 42)
chk('abandoned', d['incomplete_datasets'] == ['tdc_herg_central'] and d['multiseed_artifacts_present'] is False)
mc = d['model_coverage']
chk('chemprop 21/23', {mc['Chemprop v2 (AttentiveFP, ensemble=1)']['datasets_valid'], mc['Chemprop v2 (AttentiveFP + Selected descriptors, ensemble=1)']['datasets_valid']} == {21, 23})
chk('maplightgnn 42', mc['MapLight + GNN (CatBoost, Strict Parity)']['datasets_valid'] == 42)
chk('tabpfn 10/35', (mc['TabPFNClassifier']['datasets_valid'], mc['TabPFNRegressor']['datasets_valid']) == (10, 35))

t5 = pathlib.Path('manuscript_assets/tables/table5_ensemble_value_add.csv').read_text(encoding='utf-8')
import csv, io
rows = list(csv.DictReader(io.StringIO(t5)))
g = {(r['Fusion method'], r['Task']): r for r in rows}
chk('t5 cls invrmse 8, oof 6', g[('Inverse-RMSE weighted average', 'classification')]['Beats best base'] == '8' and g[('OOF stacking', 'classification')]['Beats best base'] == '6')
chk('t5 cls ranks 2.5/3.5', float(g[('Inverse-RMSE weighted average', 'classification')]['Median rank']) == 2.5 and float(g[('OOF stacking', 'classification')]['Median rank']) == 3.5)
chk('t5 reg oof 1/23, cfa 3/22', g[('OOF stacking', 'regression')]['Beats best base'] == '1' and g[('CFA fusion', 'regression')]['Beats best base'] == '3' and g[('CFA fusion', 'regression')]['Datasets'] == '22')

t4 = list(csv.DictReader(io.StringIO(pathlib.Path('manuscript_assets/tables/table4_leaderboard_comparison.csv').read_text(encoding='utf-8'))))
r4 = {r['Dataset']: r for r in t4}
chk('esol 0.592 rank4 n30', round(float(r4['esol_delaney']['AutoQSAR value']), 3) == 0.592 and r4['esol_delaney']['Est. rank'] == '4' and r4['esol_delaney']['References (n)'] == '30')
chk('lipo 0.582 rank6 n27', round(float(r4['lipophilicity']['AutoQSAR value']), 3) == 0.582 and r4['lipophilicity']['Est. rank'] == '6' and r4['lipophilicity']['References (n)'] == '27')
t2 = list(csv.DictReader(io.StringIO(pathlib.Path('manuscript_assets/tables/table2_dataset_catalog.csv').read_text(encoding='utf-8'))))
r2 = {r['Dataset']: r for r in t2}
chk('herg_karim 13445', r2['tdc_herg_karim']['Molecules'] == '13445')
chk('xyz 38 mols TabPFN', r2['chemml_xyz_polarizability']['Molecules'] == '38' and 'TabPFN' in r2['chemml_xyz_polarizability']['Best model'])
chk('freesolv chemprop', 'Chemprop' in r2['freesolv_sampl']['Best model'])
unimol_wins = sorted(r['Dataset'] for r in t2 if 'Uni-Mol' in r['Best model'])
chk('unimol 7 wins', len(unimol_wins) == 7, str(unimol_wins))
t6 = list(csv.DictReader(io.StringIO(pathlib.Path('manuscript_assets/tables/table6_cost.csv').read_text(encoding='utf-8'))))
r6 = {r['Architecture family']: r for r in t6}
chk('params 47.3M/395618/324097', round(float(r6['Uni-Mol V1 (3D pretrained)']['Median trainable parameters']) / 1e6, 1) == 47.3 and float(r6['Chemprop v2 GNN']['Median trainable parameters']) == 395618 and float(r6['Deep tabular NN (ChemML MLP)']['Median trainable parameters']) == 324097)
chk('pool 3147/3047', round(float(r6['CFA combinatorial fusion']['Median cost incl. base pool (s)'])) == 3147 and round(float(r6['Ensemble (stacking / averaging)']['Median cost incl. base pool (s)'])) == 3047)

print(f"PASS {len(ok)} checks")
for b in bad:
    print("FAIL", b)
figs = re.findall(r'\]\((manuscript_assets/figures/[^)]+)\)', t)
print("figures referenced:", len(figs), "missing:", [f for f in figs if not pathlib.Path(f).exists()])
print("unfilled table blocks:", re.findall(r'<!-- TABLE:(\w+) -->\s*<!-- /TABLE -->', t))

missing_figs = [f for f in figs if not pathlib.Path(f).exists()]
unfilled = re.findall(r'<!-- TABLE:(\w+) -->\s*<!-- /TABLE -->', t)
sys.exit(1 if (bad or missing_figs or unfilled) else 0)
