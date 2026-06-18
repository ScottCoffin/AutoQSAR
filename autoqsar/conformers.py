"""
autoqsar.conformers — RDKit ETKDGv3+MMFF 3D conformer generation with
persistent SQLite cache.

Design:
  - One conformer per molecule, keyed by (canonical_smiles, conformer_seed).
  - Cache is a SQLite database at <cache_dir>/conformer_cache.db.
  - All conformer generation is CPU work; no GPU required.
  - Cache hash (SHA-256 of all mol binaries, sorted by key) is stable across
    platforms and embeddable in run provenance records.
  - Standalone CLI: python -m autoqsar.conformers --dataset <NAME> ...

Reproducibility:
  The conformer seed is separate from the model seed (default 42) so the
  same 3D geometry is used across all 5 model seeds, enabling fair comparison.
  The conformer_cache_hash in the run record binds each result row to its
  exact 3D input geometry.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

_DB_FILENAME = "conformer_cache.db"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conformers (
    smiles      TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    mol_binary  BLOB NOT NULL,
    method      TEXT NOT NULL DEFAULT 'ETKDGv3+MMFF',
    PRIMARY KEY (smiles, seed)
);
"""


class ConformerCache:
    """SQLite-backed cache for 3D-embedded RDKit molecules."""

    def __init__(self, cache_dir: Path) -> None:
        self._db_path = Path(cache_dir) / _DB_FILENAME
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, smiles: str, seed: int):
        """Return cached mol object or None."""
        row = self._conn.execute(
            "SELECT mol_binary FROM conformers WHERE smiles=? AND seed=?",
            (smiles, seed),
        ).fetchone()
        if row is None:
            return None
        try:
            from rdkit.Chem import Mol
            return Mol(row[0])
        except Exception:
            return pickle.loads(row[0])  # fallback for alternative serialisation

    def put(self, smiles: str, seed: int, mol) -> None:
        """Store a mol object (must have a 3D conformer)."""
        try:
            binary = mol.ToBinary()
        except AttributeError:
            binary = pickle.dumps(mol)
        self._conn.execute(
            "INSERT OR REPLACE INTO conformers (smiles, seed, mol_binary) VALUES (?,?,?)",
            (smiles, seed, binary),
        )
        self._conn.commit()

    def __contains__(self, key: tuple[str, int]) -> bool:
        smiles, seed = key
        return self._conn.execute(
            "SELECT 1 FROM conformers WHERE smiles=? AND seed=?", (smiles, seed)
        ).fetchone() is not None

    def cache_hash(self) -> str:
        """SHA-256 over all (smiles, seed, mol_binary) rows sorted by key."""
        rows = self._conn.execute(
            "SELECT smiles, seed, mol_binary FROM conformers ORDER BY smiles, seed"
        ).fetchall()
        h = hashlib.sha256()
        for smiles, seed, binary in rows:
            h.update(smiles.encode())
            h.update(str(seed).encode())
            h.update(bytes(binary) if not isinstance(binary, bytes) else binary)
        return h.hexdigest()

    def size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM conformers").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def generate_conformer(mol, seed: int = 42):
    """Generate one ETKDGv3+MMFF conformer for mol.

    Returns mol with embedded conformer, or None on failure.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.RWMol(Chem.AddHs(mol))

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enforceChirality = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True

    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Fallback: random coords
        params2 = AllChem.EmbedParameters()
        params2.randomSeed = seed
        result = AllChem.EmbedMolecule(mol, params2)
        if result == -1:
            logger.debug("ETKDGv3 embedding failed for mol")
            return None

    try:
        ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if ff_result == -1:
            logger.debug("MMFF optimization failed (atom typing); keeping unoptimized conformer")
    except Exception as exc:
        logger.debug("MMFF optimization error: %s", exc)

    return Chem.RemoveHs(mol)


def smiles_to_conformer(smiles: str, seed: int, cache: ConformerCache):
    """Return a 3D mol for smiles, using cache if available."""
    if (smiles, seed) in cache:
        mol = cache.get(smiles, seed)
        if mol is not None:
            return mol, True  # (mol, cache_hit)

    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Could not parse SMILES: %r", smiles)
        return None, False

    mol_3d = generate_conformer(mol, seed=seed)
    if mol_3d is None:
        return None, False

    cache.put(smiles, seed, mol_3d)
    return mol_3d, False


def generate_dataset_conformers(
    dataset_name: str,
    data_dir: Path,
    cache: ConformerCache,
    conformer_seed: int = 42,
) -> tuple[int, int]:
    """Generate and cache conformers for every SMILES in a dataset.

    Returns (n_generated, n_cache_hits).

    Used in standalone CPU conformer-generation mode (§B4) so the GPU VM
    receives a pre-populated cache and skips the conformer-gen overhead.
    """
    smiles_list = _load_smiles_for_dataset(dataset_name, data_dir)
    if not smiles_list:
        logger.warning("No SMILES found for dataset %s in %s", dataset_name, data_dir)
        return 0, 0

    n_generated = 0
    n_hits = 0
    for i, smiles in enumerate(smiles_list):
        _, was_cached = smiles_to_conformer(smiles, conformer_seed, cache)
        if was_cached:
            n_hits += 1
        else:
            n_generated += 1
        if (i + 1) % 500 == 0:
            logger.info(
                "Conformer progress: %d/%d  (generated=%d, hits=%d)",
                i + 1, len(smiles_list), n_generated, n_hits,
            )

    return n_generated, n_hits


def _load_smiles_for_dataset(dataset_name: str, data_dir: Path) -> list[str]:
    """Best-effort SMILES extraction from local data files."""
    import csv

    from rdkit import Chem

    candidates = list(data_dir.glob(f"**/{dataset_name.replace('tdc_', '')}*.tab")) + \
                 list(data_dir.glob(f"**/{dataset_name.replace('tdc_', '')}*.csv"))

    for path in candidates:
        try:
            sep = "\t" if path.suffix == ".tab" else ","
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=sep)
                headers = reader.fieldnames or []
                smiles_col = next(
                    (h for h in headers if h.lower() in ("smiles", "drug", "smi", "canonical_smiles")),
                    None,
                )
                if smiles_col is None:
                    continue
                raw = [row[smiles_col] for row in reader if row.get(smiles_col)]

            # Canonicalize
            canonical = []
            for smi in raw:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    canonical.append(Chem.MolToSmiles(mol, canonical=True))
            return list(dict.fromkeys(canonical))  # deduplicate preserving order
        except Exception as exc:
            logger.debug("Could not read %s: %s", path, exc)

    return []


# ── Standalone CLI entry point ─────────────────────────────────────────────────

def _cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="AutoQSAR conformer pre-generation — runs on CPU, no GPU required."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. tdc_caco2_wang).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument("--conformer-cache", type=Path, required=True, help="Cache directory.")
    parser.add_argument("--conformer-seed", type=int, default=42, help="RDKit conformer seed.")
    args = parser.parse_args(argv)

    import time
    cache = ConformerCache(args.conformer_cache)
    t0 = time.time()
    n_gen, n_hit = generate_dataset_conformers(
        args.dataset, args.data_dir, cache, args.conformer_seed
    )
    elapsed = time.time() - t0
    h = cache.cache_hash()
    print(
        f"Done in {elapsed:.1f}s — generated={n_gen}, hits={n_hit}, "
        f"total={cache.size()}, cache_hash={h}"
    )
    cache.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
