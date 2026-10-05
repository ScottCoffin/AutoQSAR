"""Per-dataset feature matrices for the feature-expansion arm (docs/FEATURE_EXPANSION_PLAN.md).

Each family is cached as ``<cache>/<family>/<dataset>.npy`` (float32). Rows follow the committed
partition order: training rows by ``row_index``, then test rows by ``row_index``
(:func:`dataset_smiles`). A ``<dataset>.json`` sidecar records the row count and the SMILES hash,
so a stale cache is detected and rebuilt, not silently reused.

CPU families use scikit-fingerprints (``pip install scikit-fingerprints mordredcommunity gensim``).
Mol2Vec follows Jaeger et al. (2018) with the authors' pretrained ``model_300dim.pkl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qsarena.meta_analysis import io

CACHE_DIR = io.REPO_ROOT / ".model_cache" / "feature_expansion"
MOL2VEC_MODEL = io.REPO_ROOT / ".model_cache" / "mol2vec" / "model_300dim.pkl"
MOL2VEC_SHA256 = "62934b4ec245716c1e97ca95483434e0b66bfe2efc499f5123e17e6eb0c2331f"
CPU_FAMILIES = ["maccs", "ecfp4", "rdkit2d", "pubchem", "mordred", "mol2vec"]
EMBEDDING_FAMILIES = ["unimol_repr", "chemeleon"]
FEATURE_SETS = {
    "admetboost": ["maccs", "ecfp4", "mol2vec", "pubchem", "mordred", "rdkit2d"],
    "admetboost+emb": ["maccs", "ecfp4", "mol2vec", "pubchem", "mordred", "rdkit2d", "unimol_repr", "chemeleon"],
    "emb": ["unimol_repr", "chemeleon"],
    "admetboost+chemeleon": ["maccs", "ecfp4", "mol2vec", "pubchem", "mordred", "rdkit2d", "chemeleon"],
    "chemeleon": ["chemeleon"],
}


def dataset_smiles(partitions: pd.DataFrame, dataset: str) -> tuple[list[str], int]:
    """SMILES in cache row order (train by row_index, then test by row_index) and the train count."""
    sub = partitions[partitions["dataset"] == dataset]
    train = sub[sub["split"] == "train"].sort_values("row_index")
    test = sub[sub["split"] == "test"].sort_values("row_index")
    return list(train["smiles"]) + list(test["smiles"]), len(train)


def _smiles_sha(smiles: list[str]) -> str:
    return hashlib.sha256("\n".join(smiles).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# Family implementations
# ---------------------------------------------------------------------------------------------
def _skfp(name: str):
    import skfp.fingerprints as fps

    return {
        "maccs": lambda: fps.MACCSFingerprint(n_jobs=1),
        "ecfp4": lambda: fps.ECFPFingerprint(fp_size=2048, radius=2, n_jobs=1),
        "rdkit2d": lambda: fps.RDKit2DDescriptorsFingerprint(n_jobs=1),
        "pubchem": lambda: fps.PubChemFingerprint(n_jobs=1),
        "mordred": lambda: fps.MordredFingerprint(n_jobs=1),
    }[name]()


_MOL2VEC_KV = None


def _mol2vec_vectors():
    global _MOL2VEC_KV
    if _MOL2VEC_KV is None:
        from gensim.models import word2vec

        if not MOL2VEC_MODEL.exists():
            raise FileNotFoundError(
                f"{MOL2VEC_MODEL} missing: download github.com/samoturk/mol2vec/raw/master/examples/models/"
                f"model_300dim.pkl (sha256 {MOL2VEC_SHA256})"
            )
        _MOL2VEC_KV = word2vec.Word2Vec.load(str(MOL2VEC_MODEL)).wv
    return _MOL2VEC_KV


def mol2vec_sentence(mol, radius: int = 1) -> list[str]:
    """Mol2Vec "alternating sentence": per atom (in index order), its Morgan identifiers for r = 0..radius."""
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    generator.GetSparseCountFingerprint(mol, additionalOutput=output)
    per_atom: dict[int, dict[int, int]] = {}
    for identifier, hits in output.GetBitInfoMap().items():
        for atom_idx, r in hits:
            per_atom.setdefault(atom_idx, {})[r] = identifier
    words = []
    for atom_idx in sorted(per_atom):
        for r in range(radius + 1):
            if r in per_atom[atom_idx]:
                words.append(str(per_atom[atom_idx][r]))
    return words


def mol2vec_matrix(smiles: list[str]) -> tuple[np.ndarray, float]:
    """Sum of pretrained word vectors per molecule; unknown words map to 'UNK'. Returns (X, vocab hit rate)."""
    from rdkit import Chem

    kv = _mol2vec_vectors()
    out = np.zeros((len(smiles), kv.vector_size), dtype=np.float32)
    hits = total = 0
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        for word in mol2vec_sentence(mol):
            total += 1
            if word in kv.key_to_index:
                hits += 1
                out[i] += kv[word]
            else:
                out[i] += kv["UNK"]
    return out, hits / max(total, 1)


def compute_family(family: str, smiles: list[str]) -> tuple[np.ndarray, dict]:
    if family == "mol2vec":
        X, hit_rate = mol2vec_matrix(smiles)
        return X, {"vocab_hit_rate": hit_rate}
    if family in EMBEDDING_FAMILIES:
        from qsarena.feature_expansion import embeddings

        return embeddings.compute(family, smiles), {}
    X = np.asarray(_skfp(family).transform(smiles), dtype=np.float64)
    extra = {"nonfinite_fraction": float(np.mean(~np.isfinite(X)))}
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), extra


def find_cached_matrix(families: list[str], smiles: list[str], cache_dir: Path = CACHE_DIR) -> np.ndarray | None:
    """The cached matrix for exactly these SMILES (same order) if every family has it, else None.

    The arm caches by dataset name; the runner knows only the SMILES, so match on the SMILES hash instead.
    """
    sha = _smiles_sha(list(smiles))
    first = Path(cache_dir) / families[0]
    for meta in sorted(first.glob("*.json")) if first.is_dir() else []:
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info.get("smiles_sha256") != sha or info.get("n_rows") != len(smiles):
            continue
        dataset = meta.stem
        if all(is_cached(family, dataset, list(smiles), cache_dir) for family in families):
            return np.hstack([np.load(cache_paths(family, dataset, cache_dir)[0]) for family in families])
    return None


def admetboost_matrix(smiles: list[str]) -> np.ndarray:
    """The full ``admetboost`` feature set for arbitrary SMILES, columns in family order.

    Used by the benchmark runner's ``XGBoost (ADMETboost features)`` model. Reuses the feature-expansion cache when
    it holds exactly these SMILES, otherwise computes every family. Every family is label-free (nothing is fitted to
    targets), so computing train and test SMILES together cannot leak.
    """
    families = FEATURE_SETS["admetboost"]
    cached = find_cached_matrix(families, list(smiles))
    if cached is not None:
        return cached
    return np.hstack([compute_family(family, list(smiles))[0] for family in families])


# ---------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------
def cache_paths(family: str, dataset: str, cache_dir: Path = CACHE_DIR) -> tuple[Path, Path]:
    base = Path(cache_dir) / family
    return base / f"{dataset}.npy", base / f"{dataset}.json"


def is_cached(family: str, dataset: str, smiles: list[str], cache_dir: Path = CACHE_DIR) -> bool:
    npy, meta = cache_paths(family, dataset, cache_dir)
    if not (npy.exists() and meta.exists()):
        return False
    info = json.loads(meta.read_text(encoding="utf-8"))
    return info.get("smiles_sha256") == _smiles_sha(smiles) and info.get("n_rows") == len(smiles)


def featurize_dataset(family: str, dataset: str, smiles: list[str], cache_dir: Path = CACHE_DIR) -> str:
    if is_cached(family, dataset, smiles, cache_dir):
        return f"{family}/{dataset}: cached"
    import time

    started = time.perf_counter()
    X, extra = compute_family(family, smiles)
    npy, meta = cache_paths(family, dataset, cache_dir)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, X)
    meta.write_text(
        json.dumps(
            {
                "family": family,
                "dataset": dataset,
                "n_rows": len(smiles),
                "n_features": int(X.shape[1]),
                "smiles_sha256": _smiles_sha(smiles),
                "seconds": round(time.perf_counter() - started, 1),
                **extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return f"{family}/{dataset}: {X.shape} in {time.perf_counter() - started:.0f}s"


def load_features(families: list[str], dataset: str, partitions: pd.DataFrame, cache_dir: Path = CACHE_DIR):
    """Concatenated feature matrix (cache row order) plus the number of training rows."""
    smiles, n_train = dataset_smiles(partitions, dataset)
    blocks = []
    for family in families:
        if not is_cached(family, dataset, smiles, cache_dir):
            raise FileNotFoundError(f"{family}/{dataset} not featurized (or stale): run featurize.py first")
        blocks.append(np.load(cache_paths(family, dataset, cache_dir)[0]))
    return np.hstack(blocks), n_train


def _lower_priority() -> None:
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
    except Exception:
        pass


def _job(args) -> str:
    family, dataset, smiles, cache_dir = args
    _lower_priority()
    return featurize_dataset(family, dataset, smiles, Path(cache_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", default=CPU_FAMILIES)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = parser.parse_args(argv)
    _lower_priority()
    partitions = io.load_partitions()
    datasets = args.datasets or sorted(partitions["dataset"].unique())
    # Largest datasets first so the tail of the run is short jobs.
    sizes = partitions.groupby("dataset").size()
    datasets = sorted(datasets, key=lambda d: -sizes[d])
    jobs = [(f, d, dataset_smiles(partitions, d)[0], args.cache_dir) for f in args.families for d in datasets]
    if any(f in EMBEDDING_FAMILIES for f in args.families):
        # unimol_tools does not release conformer/model memory between calls (15 GB committed after five
        # 12k-molecule datasets), so each embedding job runs in a fresh child process.
        with ProcessPoolExecutor(max_workers=max(1, args.workers), max_tasks_per_child=1) as pool:
            for line in pool.map(_job, jobs):
                print(line, flush=True)
    elif args.workers <= 1:
        for job in jobs:
            print(_job(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for line in pool.map(_job, jobs):
                print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
