from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdReducedGraphs, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split

try:
    import pyarrow  # noqa: F401

    FEATURE_STORE_SHARD_FORMAT = "parquet"
except Exception:
    try:
        import fastparquet  # noqa: F401

        FEATURE_STORE_SHARD_FORMAT = "parquet"
    except Exception:
        FEATURE_STORE_SHARD_FORMAT = "csv"


FEATURE_FAMILY_LABELS = {
    "morgan": "Morgan fingerprints",
    "ecfp6": "ECFP6 fingerprints",
    "fcfp6": "FCFP6 fingerprints",
    "maccs": "MACCS keys",
    "rdkit": "RDKit descriptors",
    "maplight": "MapLight classic (Morgan + Avalon + ErG + chosen descriptors)",
}

MAPLIGHT_DESCRIPTOR_NAMES = [
    "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v",
    "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2",
    "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9",
    "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "FractionCSP3", "HallKierAlpha",
    "HeavyAtomCount", "HeavyAtomMolWt", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "MaxAbsEStateIndex",
    "MaxAbsPartialCharge", "MaxEStateIndex", "MaxPartialCharge", "MinAbsEStateIndex", "MinAbsPartialCharge",
    "MinEStateIndex", "MinPartialCharge", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles",
    "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NumRadicalElectrons", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
    "NumSaturatedRings", "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12",
    "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
    "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "RingCount", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3",
    "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10",
    "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
    "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "TPSA", "VSA_EState1", "VSA_EState10", "VSA_EState2",
    "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8",
    "VSA_EState9", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N",
    "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN",
    "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2",
    "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid",
    "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur",
    "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide",
    "fr_ester", "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone",
    "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss",
    "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom",
    "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol",
    "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide",
    "fr_prisulfonamd", "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene",
    "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea", "qed",
]


def murcko_scaffold_key(smiles_text):
    smiles_text = str(smiles_text)
    mol = Chem.MolFromSmiles(smiles_text)
    if mol is None:
        return f"INVALID::{smiles_text}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or f"NO_SCAFFOLD::{smiles_text}"


def scaffold_train_test_split(X, y, smiles, test_size=0.2, random_state=42):
    smiles_series = pd.Series(smiles, dtype=str).reset_index(drop=True)
    y_series = pd.Series(y, dtype=float).reset_index(drop=True)
    X_frame = pd.DataFrame(X).reset_index(drop=True).copy()

    n_total = len(smiles_series)
    if n_total < 3:
        raise ValueError("Scaffold splitting requires at least 3 molecules.")

    desired_test = int(round(float(test_size) * n_total))
    desired_test = min(max(desired_test, 1), n_total - 1)

    scaffold_to_indices = {}
    for idx, smiles_value in enumerate(smiles_series):
        scaffold_to_indices.setdefault(murcko_scaffold_key(smiles_value), []).append(idx)

    rng = np.random.RandomState(int(random_state))
    grouped = list(scaffold_to_indices.items())
    rng.shuffle(grouped)
    grouped.sort(key=lambda item: (-len(item[1]), item[0]))

    train_indices = []
    test_indices = []
    for _scaffold_key, indices in grouped:
        current_test = len(test_indices)
        proposed_test = current_test + len(indices)
        assign_to_test = current_test < desired_test and (
            abs(proposed_test - desired_test) <= abs(current_test - desired_test) or current_test == 0
        )
        if assign_to_test:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if len(test_indices) == 0 and len(train_indices) > 1:
        test_indices.append(train_indices.pop())
    if len(train_indices) == 0 and len(test_indices) > 1:
        train_indices.append(test_indices.pop())

    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)

    return (
        X_frame.iloc[train_indices].reset_index(drop=True),
        X_frame.iloc[test_indices].reset_index(drop=True),
        y_series.iloc[train_indices].reset_index(drop=True),
        y_series.iloc[test_indices].reset_index(drop=True),
        smiles_series.iloc[train_indices].reset_index(drop=True),
        smiles_series.iloc[test_indices].reset_index(drop=True),
    )


def target_quartile_labels(y, q=4):
    y_series = pd.Series(y, dtype=float).reset_index(drop=True)
    if len(y_series) < int(q) * 2:
        raise ValueError(f"Target-quartile splitting needs at least {int(q) * 2} rows; found {len(y_series)}.")
    labels = pd.qcut(y_series, q=int(q), labels=False, duplicates="drop")
    labels = labels.astype("Int64")
    if labels.isna().any() or labels.nunique(dropna=True) < 2:
        raise ValueError(
            "Target-quartile splitting could not create at least two non-empty target bins. "
            "Use the random split for very small or low-variance target data."
        )
    label_counts = labels.value_counts(dropna=True)
    if int(label_counts.min()) < 2:
        raise ValueError(
            "Target-quartile splitting requires at least two samples per target bin. "
            "Use a smaller test fraction or the random split for this dataset."
        )
    return labels.astype(int).astype(str)


class TargetQuartileStratifiedKFold:
    def __init__(self, n_splits=5, random_state=42, q=4):
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.q = int(q)

    def get_n_splits(self, X=None, y=None, groups=None):
        return int(self.n_splits)

    def split(self, X, y, groups=None):
        X_frame = pd.DataFrame(X).reset_index(drop=True)
        y_series = pd.Series(y, dtype=float).reset_index(drop=True)
        labels = target_quartile_labels(y_series, q=self.q)
        max_supported_folds = int(labels.value_counts(dropna=True).min())
        effective_folds = min(int(self.n_splits), max_supported_folds)
        if effective_folds < 2:
            raise ValueError("Target-quartile CV requires at least two samples per quartile bin in the training data.")
        splitter = StratifiedKFold(n_splits=int(effective_folds), shuffle=True, random_state=int(self.random_state))
        yield from splitter.split(X_frame, labels)


def make_qsar_cv_splitter(X, y, smiles, split_strategy="random", cv_folds=5, random_seed=42):
    X_frame = pd.DataFrame(X).reset_index(drop=True)
    y_series = pd.Series(y, dtype=float).reset_index(drop=True)
    smiles_series = pd.Series(smiles, dtype=str).reset_index(drop=True)
    requested_folds = min(int(cv_folds), len(X_frame))
    if requested_folds < 2:
        raise ValueError("At least 2 CV folds are required.")

    split_strategy = str(split_strategy).strip().lower()
    if split_strategy == "random":
        splitter = KFold(n_splits=requested_folds, shuffle=True, random_state=int(random_seed))
        return splitter, int(requested_folds), "random"

    if split_strategy == "target_quartiles":
        labels = target_quartile_labels(y_series, q=4)
        max_supported_folds = int(labels.value_counts(dropna=True).min())
        effective_folds = min(requested_folds, max_supported_folds)
        if effective_folds < 2:
            raise ValueError("Target-quartile CV requires at least two samples per quartile bin in the training data.")
        splitter = StratifiedKFold(n_splits=int(effective_folds), shuffle=True, random_state=int(random_seed))
        return list(splitter.split(X_frame, labels)), int(effective_folds), "target_quartiles"

    if split_strategy == "scaffold":
        groups = smiles_series.map(murcko_scaffold_key)
        unique_groups = int(pd.Series(groups).nunique(dropna=False))
        effective_folds = min(requested_folds, unique_groups)
        if effective_folds < 2:
            raise ValueError("Scaffold CV requires at least two distinct Murcko scaffold groups in the training data.")
        splitter = GroupKFold(n_splits=int(effective_folds))
        return list(splitter.split(X_frame, y_series, groups)), int(effective_folds), "scaffold"

    raise ValueError("CV split strategy must be 'random', 'target_quartiles', or 'scaffold'.")


def make_reusable_inner_cv_splitter(split_strategy="random", cv_folds=5, random_seed=42):
    requested_folds = int(cv_folds)
    if requested_folds < 2:
        raise ValueError("At least 2 CV folds are required.")

    split_strategy = str(split_strategy).strip().lower()
    if split_strategy == "random":
        return KFold(n_splits=requested_folds, shuffle=True, random_state=int(random_seed)), requested_folds, "random"
    if split_strategy == "target_quartiles":
        return TargetQuartileStratifiedKFold(n_splits=requested_folds, random_state=int(random_seed), q=4), requested_folds, "target_quartiles"
    if split_strategy == "scaffold":
        fallback = KFold(n_splits=requested_folds, shuffle=True, random_state=int(random_seed))
        return fallback, requested_folds, "random_inner_fallback_for_scaffold"
    raise ValueError("CV split strategy must be 'random', 'target_quartiles', or 'scaffold'.")


def make_morgan_matrix(smiles_list, radius=2, n_bits=1024):
    generator = AllChem.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"morgan_bit_{i:04d}" for i in range(n_bits)])


def make_circular_fingerprint_matrix(smiles_list, radius=3, n_bits=1024, use_features=False, prefix="ecfp6"):
    generator = AllChem.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
        atomInvariantsGenerator=(AllChem.GetMorganFeatureAtomInvGen() if bool(use_features) else None),
    )
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"{prefix}_bit_{i:04d}" for i in range(n_bits)])


def make_ecfp6_matrix(smiles_list, n_bits=1024):
    return make_circular_fingerprint_matrix(smiles_list, radius=3, n_bits=int(n_bits), use_features=False, prefix="ecfp6")


def make_fcfp6_matrix(smiles_list, n_bits=1024):
    return make_circular_fingerprint_matrix(smiles_list, radius=3, n_bits=int(n_bits), use_features=True, prefix="fcfp6")


def make_maccs_matrix(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"maccs_bit_{i:03d}" for i in range(fp.GetNumBits())])


def make_rdkit_descriptor_matrix(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rows.append({f"rdkit_{name}": func(mol) for name, func in Descriptors._descList})
    return pd.DataFrame(rows)


def make_maplight_morgan_count_matrix(smiles_list, radius=2, n_bits=1024):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros((int(n_bits),), dtype=np.float32))
            continue
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=int(radius), nBits=int(n_bits))
        arr = np.zeros((int(n_bits),), dtype=np.int32)
        for bit_id, count in fp.GetNonzeroElements().items():
            bit_id = int(bit_id)
            if 0 <= bit_id < int(n_bits):
                arr[bit_id] = int(count)
        rows.append(arr.astype(np.float32))
    return pd.DataFrame(rows, columns=[f"maplight_morgan_{i:04d}" for i in range(int(n_bits))])


def make_avalon_count_matrix(smiles_list, n_bits=1024):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros((int(n_bits),), dtype=np.float32))
            continue
        fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=int(n_bits))
        arr = np.zeros((int(n_bits),), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr.astype(np.float32))
    return pd.DataFrame(rows, columns=[f"avalon_count_{i:04d}" for i in range(int(n_bits))])


def make_erg_matrix(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        erg = rdReducedGraphs.GetErGFingerprint(mol)
        rows.append(np.asarray(erg, dtype=np.float32))
    arr = np.vstack(rows)
    return pd.DataFrame(arr, columns=[f"erg_{i:03d}" for i in range(arr.shape[1])])


def make_maplight_descriptor_matrix(smiles_list):
    calculator = MolecularDescriptorCalculator(MAPLIGHT_DESCRIPTOR_NAMES)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rows.append(np.asarray(calculator.CalcDescriptors(mol), dtype=np.float32))
    arr = np.vstack(rows)
    return pd.DataFrame(arr, columns=[f"maplight_desc_{name}" for name in MAPLIGHT_DESCRIPTOR_NAMES])


def make_maplight_classic_matrix(smiles_list, radius=2, n_bits=1024):
    parts = [
        make_maplight_morgan_count_matrix(smiles_list, radius=radius, n_bits=n_bits),
        make_avalon_count_matrix(smiles_list, n_bits=n_bits),
        make_erg_matrix(smiles_list),
        make_maplight_descriptor_matrix(smiles_list),
    ]
    return pd.concat(parts, axis=1)


def finalize_feature_matrix(feature_df):
    feature_df = feature_df.copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    float32_limit = np.finfo(np.float32).max
    feature_df = feature_df.mask(feature_df.abs() > float32_limit, np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_df = feature_df.fillna(0.0)
    return feature_df.astype(np.float32)


def align_feature_matrix_to_training_columns(feature_df, expected_columns):
    aligned = feature_df.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned = aligned.loc[:, list(expected_columns)].copy()
    return finalize_feature_matrix(aligned)


def normalize_selected_feature_families(selected_feature_families=None, **feature_flags):
    selected = []
    if selected_feature_families is not None:
        for family in selected_feature_families:
            family_key = str(family).strip().lower()
            if family_key in FEATURE_FAMILY_LABELS and family_key not in selected:
                selected.append(family_key)
    flag_mapping = {
        "use_morgan_features": "morgan",
        "use_ecfp6_features": "ecfp6",
        "use_fcfp6_features": "fcfp6",
        "use_maccs_keys": "maccs",
        "use_rdkit_descriptors": "rdkit",
        "use_maplight_classic": "maplight",
    }
    for flag_name, family_key in flag_mapping.items():
        if bool(feature_flags.get(flag_name)) and family_key not in selected:
            selected.append(family_key)
    return selected


def feature_family_label(family_key, radius=2, n_bits=1024):
    family_key = str(family_key).strip().lower()
    if family_key == "morgan":
        return f"Morgan(radius={int(radius)}, bits={int(n_bits)})"
    if family_key == "ecfp6":
        return f"ECFP6(bits={int(n_bits)})"
    if family_key == "fcfp6":
        return f"FCFP6(bits={int(n_bits)})"
    if family_key == "maplight":
        return f"MapLight classic (Morgan+Avalon+ErG, bits={int(n_bits)})"
    return FEATURE_FAMILY_LABELS[family_key]


def default_feature_store_path():
    return Path("./model_cache/feature_store_parquet")


def resolve_feature_store_path(feature_store_path="AUTO"):
    path_text = str(feature_store_path).strip()
    if not path_text or path_text.upper() == "AUTO":
        return default_feature_store_path()
    return Path(path_text)


def feature_store_representation_payload(selected_families, radius=2, n_bits=1024):
    selected_families = [str(family).strip().lower() for family in selected_families]
    return {
        "families": selected_families,
        "morgan_radius": int(radius) if any(family in {"morgan", "maplight"} for family in selected_families) else None,
        "fingerprint_bits": int(n_bits) if any(family in {"morgan", "ecfp6", "fcfp6", "maplight"} for family in selected_families) else None,
    }


def feature_store_representation_key(selected_families, radius=2, n_bits=1024):
    payload = feature_store_representation_payload(selected_families, radius=radius, n_bits=n_bits)
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16]
    family_prefix = "_".join(payload["families"]) or "none"
    return f"{family_prefix}__{payload_hash}"


def ensure_feature_store(feature_store_path):
    feature_store_path = Path(feature_store_path)
    feature_store_path.mkdir(parents=True, exist_ok=True)
    return feature_store_path


def feature_store_representation_dir(feature_store_path, representation_key):
    return Path(feature_store_path) / representation_key


def feature_store_representation_schema_path(feature_store_path, representation_key):
    return feature_store_representation_dir(feature_store_path, representation_key) / "_schema.json"


def feature_store_representation_shard_dir(feature_store_path, representation_key):
    return feature_store_representation_dir(feature_store_path, representation_key) / "shards"


def write_feature_store_schema(feature_store_path, representation_key, representation_payload, columns):
    representation_dir = feature_store_representation_dir(feature_store_path, representation_key)
    representation_dir.mkdir(parents=True, exist_ok=True)
    feature_store_representation_shard_dir(feature_store_path, representation_key).mkdir(parents=True, exist_ok=True)
    schema_path = feature_store_representation_schema_path(feature_store_path, representation_key)
    schema_payload = {
        "representation_key": representation_key,
        "representation_payload": representation_payload,
        "columns": list(columns),
    }
    if schema_path.exists():
        existing = json.loads(schema_path.read_text(encoding="utf-8"))
        if list(existing.get("columns", [])) != list(columns) or dict(existing.get("representation_payload", {})) != dict(representation_payload):
            raise RuntimeError(
                "The molecular feature store already has a different schema for this representation. "
                "Use a new feature-store path if you need to rebuild that representation definition."
            )
        return
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")


def feature_store_shard_paths(feature_store_path, representation_key):
    shard_dir = feature_store_representation_shard_dir(feature_store_path, representation_key)
    if not shard_dir.exists():
        return []
    pattern = "*.parquet" if FEATURE_STORE_SHARD_FORMAT == "parquet" else "*.csv.gz"
    return sorted(shard_dir.glob(pattern))


def read_feature_store_shard(shard_path):
    if str(shard_path).lower().endswith(".parquet"):
        return pd.read_parquet(shard_path)
    return pd.read_csv(shard_path)


def write_feature_store_shard(shard_df, shard_path):
    if str(shard_path).lower().endswith(".parquet"):
        shard_df.to_parquet(shard_path, index=False)
    else:
        shard_df.to_csv(shard_path, index=False, compression="gzip")


def load_feature_store_rows(feature_store_path, representation_key, smiles_list, expected_columns):
    smiles_list = [str(smiles) for smiles in smiles_list]
    if not smiles_list:
        return pd.DataFrame(columns=list(expected_columns))
    shard_paths = feature_store_shard_paths(feature_store_path, representation_key)
    if not shard_paths:
        return pd.DataFrame(columns=list(expected_columns))
    requested_smiles = set(smiles_list)
    loaded_frames = []
    for shard_path in shard_paths:
        shard_df = read_feature_store_shard(shard_path)
        if "canonical_smiles" not in shard_df.columns:
            continue
        shard_df["canonical_smiles"] = shard_df["canonical_smiles"].astype(str)
        shard_df = shard_df.loc[shard_df["canonical_smiles"].isin(requested_smiles)].copy()
        if shard_df.empty:
            continue
        for column in expected_columns:
            if column not in shard_df.columns:
                shard_df[column] = np.nan
        loaded_frames.append(shard_df.loc[:, ["canonical_smiles", *list(expected_columns)]])
    if not loaded_frames:
        return pd.DataFrame(columns=list(expected_columns))
    combined = pd.concat(loaded_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["canonical_smiles"], keep="first")
    combined = combined.set_index("canonical_smiles")
    combined = combined.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    ordered_rows = [combined.loc[str(smiles)].to_numpy(dtype=np.float32) for smiles in smiles_list if str(smiles) in combined.index]
    ordered_index = [str(smiles) for smiles in smiles_list if str(smiles) in combined.index]
    return pd.DataFrame(ordered_rows, columns=list(expected_columns), index=ordered_index, dtype=np.float32)


def append_feature_store_rows(feature_store_path, representation_key, feature_df, smiles_list):
    feature_df = feature_df.copy()
    feature_df = feature_df.loc[:, list(feature_df.columns)].astype(np.float32)
    smiles_list = [str(smiles) for smiles in smiles_list]
    if feature_df.empty or not smiles_list:
        return 0
    shard_dir = feature_store_representation_shard_dir(feature_store_path, representation_key)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_df = feature_df.reset_index(drop=True).copy()
    shard_df.insert(0, "canonical_smiles", smiles_list)
    suffix = ".parquet" if FEATURE_STORE_SHARD_FORMAT == "parquet" else ".csv.gz"
    shard_path = shard_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    write_feature_store_shard(shard_df, shard_path)
    return int(len(shard_df))


def build_feature_family_frames(smiles_list, selected_families, radius=2, n_bits=1024):
    frames = []
    labels = []
    built_families = []
    skipped_families = []
    warnings_list = []
    for family_key in selected_families:
        if family_key == "morgan":
            frames.append(make_morgan_matrix(smiles_list, radius=int(radius), n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "maplight":
            frames.append(make_maplight_classic_matrix(smiles_list, radius=int(radius), n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "ecfp6":
            frames.append(make_ecfp6_matrix(smiles_list, n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "fcfp6":
            frames.append(make_fcfp6_matrix(smiles_list, n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "maccs":
            frames.append(make_maccs_matrix(smiles_list))
            labels.append(feature_family_label(family_key))
            built_families.append(family_key)
        elif family_key == "rdkit":
            frames.append(make_rdkit_descriptor_matrix(smiles_list))
            labels.append(feature_family_label(family_key))
            built_families.append(family_key)
        else:
            raise ValueError(f"Unsupported feature family: {family_key}")
    return frames, labels, built_families, skipped_families, warnings_list


def build_feature_matrix_from_smiles(
    smiles_list,
    selected_feature_families=None,
    radius=2,
    n_bits=1024,
    enable_persistent_feature_store=True,
    reuse_persistent_feature_store=True,
    persistent_feature_store_path="AUTO",
    **feature_flags,
):
    selected_families = normalize_selected_feature_families(
        selected_feature_families=selected_feature_families,
        **feature_flags,
    )
    if not selected_families and bool(feature_flags.get("use_maplight_classic")):
        selected_families = ["maplight"]
    if not selected_families:
        raise ValueError("Please select at least one molecular feature family.")
    smiles_list = [str(smiles) for smiles in smiles_list]
    representation_payload = feature_store_representation_payload(selected_families, radius=radius, n_bits=n_bits)
    representation_key = feature_store_representation_key(selected_families, radius=radius, n_bits=n_bits)

    frames, labels, built_families, skipped_families, warnings_list = build_feature_family_frames(
        smiles_list[:1],
        selected_families,
        radius=radius,
        n_bits=n_bits,
    )
    schema_feature_df = finalize_feature_matrix(pd.concat(frames, axis=1))
    expected_columns = list(schema_feature_df.columns)

    cached_feature_df = pd.DataFrame(columns=expected_columns)
    missing_smiles = list(smiles_list)
    feature_store_path_resolved = None
    if bool(enable_persistent_feature_store):
        feature_store_path_resolved = ensure_feature_store(resolve_feature_store_path(persistent_feature_store_path))
        write_feature_store_schema(
            feature_store_path_resolved,
            representation_key,
            representation_payload,
            expected_columns,
        )
        if bool(reuse_persistent_feature_store):
            cached_feature_df = load_feature_store_rows(
                feature_store_path_resolved,
                representation_key,
                smiles_list,
                expected_columns,
            )
            missing_smiles = [smiles for smiles in smiles_list if smiles not in set(cached_feature_df.index.astype(str))]

    generated_feature_df = pd.DataFrame(columns=expected_columns)
    if missing_smiles:
        frames, labels, built_families, skipped_families, warnings_list = build_feature_family_frames(
            missing_smiles,
            selected_families,
            radius=radius,
            n_bits=n_bits,
        )
        if not frames:
            raise RuntimeError("None of the requested molecular feature families could be built.")
        generated_feature_df = finalize_feature_matrix(pd.concat(frames, axis=1))
        generated_feature_df = generated_feature_df.loc[:, expected_columns].copy()
        generated_feature_df.index = pd.Index(missing_smiles, dtype=str)
        if feature_store_path_resolved is not None:
            append_feature_store_rows(
                feature_store_path_resolved,
                representation_key,
                generated_feature_df,
                missing_smiles,
            )

    cached_row_map = (
        {
            str(index): np.asarray(row.to_numpy(dtype=np.float32), dtype=np.float32)
            for index, row in cached_feature_df.groupby(level=0).first().iterrows()
        }
        if not cached_feature_df.empty
        else {}
    )
    generated_row_map = (
        {
            str(index): np.asarray(row.to_numpy(dtype=np.float32), dtype=np.float32)
            for index, row in generated_feature_df.groupby(level=0).first().iterrows()
        }
        if not generated_feature_df.empty
        else {}
    )
    combined_rows = []
    for smiles_text in smiles_list:
        if smiles_text in cached_row_map:
            combined_rows.append(cached_row_map[smiles_text])
        elif smiles_text in generated_row_map:
            combined_rows.append(generated_row_map[smiles_text])
        else:
            raise RuntimeError(f"Feature generation did not return a row for canonical SMILES: {smiles_text}")
    combined = pd.DataFrame(combined_rows, columns=expected_columns, dtype=np.float32)
    build_info = {
        "selected_feature_families": list(selected_families),
        "built_feature_families": list(built_families),
        "skipped_feature_families": list(skipped_families),
        "warnings": list(warnings_list),
        "representation_label": " + ".join(labels),
        "representation_key": representation_key,
        "feature_store_path": str(feature_store_path_resolved) if feature_store_path_resolved is not None else "",
        "feature_store_shard_format": FEATURE_STORE_SHARD_FORMAT,
        "cached_rows_loaded": int(len(smiles_list) - len(missing_smiles)),
        "generated_rows_added": int(len(missing_smiles)),
    }
    return combined, build_info
