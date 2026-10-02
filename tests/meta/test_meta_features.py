"""Phase 1: meta-features on a hand-checkable toy case and on the real partitions."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from portable_colab_qsar_bundle.qsar_workflow_core import make_morgan_matrix
from qsarena.meta_analysis import io
from qsarena.meta_analysis import meta_features as mf

TRAIN = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C(=O)O", "C1CCCCC1O", "c1ccc2ccccc2c1", "CCO"]
TRAIN_Y = [1, 0, 0, 0, 0, 1]
TEST = ["c1ccccc1Cl", "CCCO", "c1ccncc1"]


def _rdkit_bitvects(smiles):
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


@pytest.fixture(scope="module")
def toy():
    return mf.compute_meta_features(TRAIN, TRAIN_Y, TEST)


def test_toy_scaffolds_and_labels(toy):
    # Scaffolds: benzene x3, cyclohexane, naphthalene, and the acyclic CCO (its own key in the core).
    assert toy["n_bemis_murcko_scaffolds"] == 4
    assert toy["scaffolds_per_molecule"] == pytest.approx(4 / 6)
    assert toy["singleton_scaffold_frac"] == pytest.approx(3 / 4)
    assert toy["task"] == "classification"
    assert toy["pos_prevalence"] == pytest.approx(1 / 3)
    assert toy["imbalance_ratio"] == pytest.approx(2.0)
    assert np.isnan(toy["target_skew"])


def test_toy_similarity_matches_rdkit(toy):
    train_fp, test_fp = _rdkit_bitvects(TRAIN), _rdkit_bitvects(TEST)
    snn = [max(DataStructs.BulkTanimotoSimilarity(fp, train_fp)) for fp in test_fp]
    assert toy["mean_snn"] == pytest.approx(np.mean(snn), abs=1e-6)
    assert toy["median_snn"] == pytest.approx(np.median(snn), abs=1e-6)
    assert toy["ood_fraction"] == pytest.approx(np.mean(np.array(snn) < 0.40), abs=1e-12)
    pairs = [DataStructs.TanimotoSimilarity(train_fp[i], train_fp[j]) for i in range(6) for j in range(i + 1, 6)]
    assert toy["internal_diversity"] == pytest.approx(1 - np.mean(pairs), abs=1e-6)
    assert toy["internal_diversity_method"] == "exact"


def test_regression_label_features():
    features = mf.compute_meta_features(TRAIN, [0.1, 0.5, 2.0, 3.5, 1.0, 9.0], TEST)
    assert features["task"] == "regression"
    assert features["target_range"] == pytest.approx(8.9)
    assert np.isnan(features["pos_prevalence"])


def test_fingerprints_bit_identical_to_core():
    smiles = TRAIN + TEST
    core = make_morgan_matrix(smiles, radius=2, n_bits=2048).to_numpy() > 0
    assert np.array_equal(mf.fingerprints(smiles, 2, 2048), core)


@pytest.fixture(scope="module")
def partitions():
    return io.load_partitions()


@pytest.fixture(scope="module")
def summary():
    return io.load_dataset_summary()


def test_sampled_diversity_is_deterministic(partitions):
    train = partitions[(partitions["dataset"] == "tdc_ames") & (partitions["split"] == "train")]
    fp = mf.fingerprints(train["smiles"])
    assert len(fp) > mf.EXACT_DIVERSITY_MAX_N
    first, method = mf.internal_diversity(fp, seed=0)
    second, _ = mf.internal_diversity(fp, seed=0)
    assert first == second and method.startswith("sampled")
    assert 0 < first < 1


@pytest.mark.parametrize("dataset", ["tdc_herg", "esol_delaney", "tdc_cyp2d6_veith", "polaris_adme_fang_solu_1"])
def test_partition_matches_recorded_split_hash(partitions, summary, dataset):
    sub = partitions[partitions["dataset"] == dataset]
    assert io.verify_partition_hashes(sub, summary) == []


def test_all_partitions_match_split_hashes(partitions, summary):
    assert io.verify_partition_hashes(partitions, summary) == []


def test_catalog_ranges_and_coverage(partitions, summary):
    small = partitions[partitions["dataset"].isin(["tdc_herg", "esol_delaney", "tdc_caco2_wang"])]
    catalog = mf.compute_meta_feature_catalog(small, summary)
    assert len(catalog) == 3
    assert catalog[["n_train", "n_test", "task", "split_strategy"]].notna().all().all()
    for column in ("mean_snn", "median_snn", "ood_fraction", "internal_diversity"):
        assert catalog[column].between(0, 1).all()
    assert (catalog["n_bemis_murcko_scaffolds"] >= 0).all()
    prevalence = catalog["pos_prevalence"].dropna()
    assert ((prevalence > 0) & (prevalence < 1)).all()
