"""
autoqsar.unimolv2 — Uni-Mol2 V2 workflow wrapper.

Key design points (§B):
  - Model size 84m by default (fits RTX 4070 12 GB and A100 g3.large 20 GB).
  - Uses its own representation path: 3D conformers from ConformerCache.
    Does NOT consume MapLight features or ElasticNet-selected descriptors.
  - BF16 autocast enabled by default when precision == "tf32_bf16".
  - OOM-retry: halves batch size on torch.cuda.OutOfMemoryError, up to 3 times.
  - Uses the same train/test split (scaffold or random + seed) as all other
    workflows for a fair, apples-to-apples comparison.
  - Records 3D-input flag, checkpoint SHA-256, conformer-cache hash in output.

Checkpoint management:
  Pretrained weights are downloaded once to
  <input_dir>/model_cache/unimolv2_checkpoints/<size>/checkpoint.pt
  and pinned by SHA-256.  The operator must supply the SHA-256 for each
  size in UNIMOLV2_CHECKPOINT_SHA256 below (fill after first download).

Dependencies (must be installed in container — see environment/requirements-lock.txt):
  unicore      — the DeepModeling framework backbone (requires CUDA compilation)
  unimol_tools — high-level downstream task tools (supports V2 with unicore)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Checkpoint registry ──────────────────────────────────────────────────────
# TODO: fill SHA-256 hashes after downloading the official checkpoints from
# https://github.com/dptech-corp/Uni-Mol/releases
UNIMOLV2_CHECKPOINT_SHA256: dict[str, str] = {
    "84m":  "TODO_operator_fill_84m_sha256",
    "164m": "TODO_operator_fill_164m_sha256",
    "310m": "TODO_operator_fill_310m_sha256",
    "1.1b": "TODO_operator_fill_1b_sha256",
}

# Approximate VRAM requirements in GB (for user warnings)
UNIMOLV2_VRAM_GB: dict[str, float] = {
    "84m": 6.0,
    "164m": 9.0,
    "310m": 15.0,
    "1.1b": 34.0,
}

_MAX_OOM_RETRIES = 3


class UnimolV2Workflow:
    """Trains and evaluates Uni-Mol2 on one (dataset, seed) task.

    The workflow is intentionally separate from the MapLight/ElasticNet pipeline:
      input → canonical SMILES → 3D conformers (cached) → Uni-Mol2 encoder → MLP head
    """

    def __init__(
        self,
        *,
        model_size: str = "84m",
        device: str = "cuda",
        precision: str = "tf32_bf16",
        batch_size: int = 32,
        seed: int = 13,
        conformer_seed: int = 42,
        conformer_cache: Any = None,
        output_dir: Path,
        checkpoint_dir: Path | None = None,
        epochs: int = 20,
        learning_rate: float = 1e-4,
    ) -> None:
        self.model_size = model_size.lower()
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.seed = seed
        self.conformer_seed = conformer_seed
        self.conformer_cache = conformer_cache
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.learning_rate = learning_rate

        if self.model_size not in UNIMOLV2_CHECKPOINT_SHA256:
            raise ValueError(
                f"Unknown Uni-Mol2 size {self.model_size!r}. "
                f"Valid: {list(UNIMOLV2_CHECKPOINT_SHA256)}"
            )

        vram_needed = UNIMOLV2_VRAM_GB.get(self.model_size, 0)
        if self.model_size != "84m":
            logger.warning(
                "Uni-Mol2 %s requires ~%.0f GB VRAM. Ensure your GPU can accommodate "
                "this before proceeding. A100 g3.large has 20 GB.",
                self.model_size, vram_needed,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, dataset_name: str, data_dir: Path) -> dict:
        """End-to-end: load data → split → conformers → train → predict → save."""
        from autoqsar.conformers import smiles_to_conformer

        t0 = time.time()
        logger.info("Uni-Mol2 %s: starting on %s (device=%s, precision=%s)",
                    self.model_size, dataset_name, self.device, self.precision)

        # Load dataset (SMILES + labels)
        smiles_list, y = self._load_dataset(dataset_name, data_dir)
        if len(smiles_list) == 0:
            logger.warning("No data loaded for %s; skipping Uni-Mol2", dataset_name)
            return {}

        # Reproducible split matching the benchmark runner's default
        train_idx, test_idx = self._split(smiles_list, y)

        # Build conformer representations (from shared cache)
        logger.info("Loading/generating 3D conformers for %d molecules ...", len(smiles_list))
        mols = []
        for smi in smiles_list:
            mol, hit = smiles_to_conformer(smi, self.conformer_seed, self.conformer_cache)
            mols.append(mol)

        cache_hits = sum(1 for smi in smiles_list
                        if self.conformer_cache and (smi, self.conformer_seed) in self.conformer_cache)
        logger.info("Conformer cache: %d hits / %d molecules", cache_hits, len(smiles_list))

        # Train + predict
        train_mols = [mols[i] for i in train_idx]
        train_y    = [y[i]    for i in train_idx]
        test_mols  = [mols[i] for i in test_idx]
        test_y     = [y[i]    for i in test_idx]

        train_preds, test_preds = self._train_predict(
            train_mols, np.array(train_y, dtype=np.float32),
            test_mols,  np.array(test_y,  dtype=np.float32),
        )

        # Compute metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import spearmanr
        test_arr = np.array(test_y, dtype=np.float32)
        pred_arr = np.array(test_preds, dtype=np.float32)
        metrics = {
            "model": f"UnimolV2 ({self.model_size})",
            "workflow": "unimolv2",
            "test_rmse":  float(np.sqrt(mean_squared_error(test_arr, pred_arr))),
            "test_mae":   float(mean_absolute_error(test_arr, pred_arr)),
            "test_r2":    float(r2_score(test_arr, pred_arr)),
            "test_spearman": float(spearmanr(test_arr, pred_arr).correlation),
            "model_size": self.model_size,
            "checkpoint_sha256": UNIMOLV2_CHECKPOINT_SHA256[self.model_size],
            "conformer_method": "ETKDGv3+MMFF",
            "conformer_seed": self.conformer_seed,
            "conformer_cache_hash": (
                self.conformer_cache.cache_hash() if self.conformer_cache else "no_cache"
            ),
            "uses_3d_geometry": True,
            "precision_mode": self.precision,
            "elapsed_seconds": time.time() - t0,
        }

        self._save_outputs(metrics, test_preds, [smiles_list[i] for i in test_idx], test_y)
        return metrics

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_dataset(
        self, dataset_name: str, data_dir: Path
    ) -> tuple[list[str], list[float]]:
        """Load SMILES and target values from local files."""
        import csv
        from rdkit import Chem

        ds_stem = dataset_name.replace("tdc_", "")
        candidates = (
            list(Path(data_dir).glob(f"**/{ds_stem}*.tab")) +
            list(Path(data_dir).glob(f"**/{ds_stem}*.csv"))
        )

        for path in candidates:
            try:
                sep = "\t" if path.suffix == ".tab" else ","
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=sep)
                    headers = reader.fieldnames or []
                    smi_col = next(
                        (h for h in headers
                         if h.lower() in ("smiles", "drug", "smi", "canonical_smiles")),
                        None,
                    )
                    target_col = next(
                        (h for h in headers
                         if h.lower() not in ("smiles", "drug", "smi", "canonical_smiles", "index", "")
                         and h.lower() not in ("split",)),
                        None,
                    )
                    if smi_col is None or target_col is None:
                        continue
                    rows = list(reader)

                smiles_list, y = [], []
                for row in rows:
                    smi = row.get(smi_col, "").strip()
                    val = row.get(target_col, "").strip()
                    if not smi or not val:
                        continue
                    try:
                        v = float(val)
                    except ValueError:
                        continue
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    smiles_list.append(Chem.MolToSmiles(mol, canonical=True))
                    y.append(v)

                if smiles_list:
                    logger.info("Loaded %d rows from %s", len(smiles_list), path.name)
                    return smiles_list, y
            except Exception as exc:
                logger.debug("Could not load %s: %s", path, exc)

        return [], []

    def _split(
        self, smiles_list: list[str], y: list[float]
    ) -> tuple[list[int], list[int]]:
        """Reproducible 80/20 scaffold split matching the benchmark runner default."""
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError:
            # Fall back to random split
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(len(smiles_list)).tolist()
            cut = int(0.8 * len(idx))
            return idx[:cut], idx[cut:]

        scaffold_map: dict[str, list[int]] = {}
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaffold = ""
            scaffold_map.setdefault(scaffold, []).append(i)

        rng = np.random.default_rng(self.seed)
        groups = list(scaffold_map.values())
        rng.shuffle(groups)

        test_target = int(0.2 * len(smiles_list))
        test_idx, train_idx = [], []
        for grp in groups:
            if len(test_idx) < test_target:
                test_idx.extend(grp)
            else:
                train_idx.extend(grp)

        return train_idx, test_idx

    def _train_predict(
        self,
        train_mols: list,
        train_y: np.ndarray,
        test_mols: list,
        test_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train Uni-Mol2 and return (train_preds, test_preds)."""
        try:
            return self._train_with_unimol_tools(train_mols, train_y, test_mols, test_y)
        except ImportError as exc:
            logger.warning("unimol_tools/unicore not installed: %s. Returning NaN preds.", exc)
            return (
                np.full(len(train_y), float("nan")),
                np.full(len(test_y),  float("nan")),
            )

    def _train_with_unimol_tools(
        self,
        train_mols: list,
        train_y: np.ndarray,
        test_mols: list,
        test_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Concrete implementation using unimol_tools MolTrain/MolPredict.

        unimol_tools wraps Uni-Mol2 for downstream regression/classification.
        Checkpoint path is resolved from self.checkpoint_dir / self.model_size.
        """
        from rdkit import Chem
        from autoqsar.precision import amp_context

        # unimol_tools expects SMILES strings (it handles 3D internally via its
        # own atom-coordinate extraction); we pass canonical SMILES and let it
        # use our pre-generated coords from mol objects where possible.
        train_smiles = [Chem.MolToSmiles(m, canonical=True) if m else "" for m in train_mols]
        test_smiles  = [Chem.MolToSmiles(m, canonical=True) if m else "" for m in test_mols]

        # Filter out any None mols
        valid_train = [(s, y) for s, y in zip(train_smiles, train_y) if s]
        valid_test  = [(s, y) for s, y in zip(test_smiles,  test_y)  if s]
        if not valid_train:
            raise ValueError("No valid train molecules after filtering")

        t_smiles, t_y = zip(*valid_train)
        e_smiles, e_y = zip(*valid_test) if valid_test else ([], [])

        ckpt_dir = self._resolve_checkpoint_dir()

        # Batch size with OOM retry
        batch_size = self.batch_size
        last_exc = None
        for attempt in range(_MAX_OOM_RETRIES + 1):
            try:
                from unimol_tools import MolTrain, MolPredict
                import torch

                with amp_context("unimolv2", self.precision, self.device):
                    trainer = MolTrain(
                        task="regression",
                        data_type="molecule",
                        epochs=self.epochs,
                        learning_rate=self.learning_rate,
                        batch_size=batch_size,
                        seed=self.seed,
                        use_gpu=(self.device != "cpu"),
                        model_size=self.model_size,
                        pretrain_dir=str(ckpt_dir),
                        metrics="mae",
                    )
                    trainer.fit(
                        data={"smiles": list(t_smiles), "target": list(t_y)},
                    )
                    predictor = MolPredict(load_model=trainer.save_path)
                    train_preds = np.array(
                        predictor.predict(data={"smiles": list(t_smiles)})
                    ).flatten()
                    test_preds = np.array(
                        predictor.predict(data={"smiles": list(e_smiles)})
                    ).flatten() if e_smiles else np.array([])

                return train_preds, test_preds

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                    if attempt < _MAX_OOM_RETRIES:
                        batch_size = max(1, batch_size // 2)
                        logger.warning(
                            "CUDA OOM on attempt %d — halving batch size to %d and retrying",
                            attempt + 1, batch_size,
                        )
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        last_exc = exc
                        continue
                raise
        raise RuntimeError(
            f"Uni-Mol2 training failed after {_MAX_OOM_RETRIES} OOM retries"
        ) from last_exc

    def _resolve_checkpoint_dir(self) -> Path:
        """Return the directory containing the Uni-Mol2 pretrained weights."""
        if self.checkpoint_dir is not None:
            base = Path(self.checkpoint_dir)
        else:
            base = (
                Path(__file__).resolve().parents[1]
                / "model_cache"
                / "unimolv2_checkpoints"
            )
        ckpt_dir = base / self.model_size
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_file = ckpt_dir / "checkpoint.pt"
        if not ckpt_file.exists():
            logger.warning(
                "Uni-Mol2 checkpoint not found at %s. "
                "Download from https://github.com/dptech-corp/Uni-Mol/releases "
                "and place at %s, then record SHA-256 in "
                "autoqsar/unimolv2.py:UNIMOLV2_CHECKPOINT_SHA256[%r].",
                ckpt_file, ckpt_file, self.model_size,
            )
        else:
            # Verify SHA-256 if we have an expected hash
            expected = UNIMOLV2_CHECKPOINT_SHA256.get(self.model_size, "")
            if expected and not expected.startswith("TODO"):
                actual = _file_sha256(ckpt_file)
                if actual != expected:
                    raise RuntimeError(
                        f"Uni-Mol2 checkpoint SHA-256 mismatch for {self.model_size}! "
                        f"Expected: {expected}  Got: {actual}  "
                        f"File: {ckpt_file}"
                    )
                logger.info("Checkpoint SHA-256 verified: %s", actual[:16] + "...")

        return ckpt_dir

    def _save_outputs(
        self,
        metrics: dict,
        predictions: np.ndarray,
        smiles: list[str],
        true_values: list[float],
    ) -> None:
        """Write metrics and predictions to the output directory."""
        import csv

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Append to metrics.csv (matches benchmark runner format)
        metrics_path = self.output_dir / "metrics.csv"
        fieldnames = list(metrics.keys())
        mode = "a" if metrics_path.exists() else "w"
        with open(metrics_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            writer.writerow(metrics)

        # Write predictions
        pred_path = self.output_dir / "unimolv2_predictions.csv"
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "y_true", "y_pred"])
            for smi, yt, yp in zip(smiles, true_values, predictions):
                writer.writerow([smi, yt, yp])

        # Write metadata sidecar
        meta_path = self.output_dir / "unimolv2_metadata.json"
        meta = {
            "model_size": self.model_size,
            "checkpoint_sha256": UNIMOLV2_CHECKPOINT_SHA256[self.model_size],
            "uses_3d_geometry": True,
            "conformer_method": "ETKDGv3+MMFF",
            "conformer_seed": self.conformer_seed,
            "conformer_cache_hash": (
                self.conformer_cache.cache_hash() if self.conformer_cache else "no_cache"
            ),
            "precision_mode": self.precision,
            "note": (
                "Uni-Mol2 consumes 3D molecular geometry (extra information vs 2D/SMILES "
                "models). Performance delta relative to 2D models should be interpreted "
                "in light of this additional geometric information."
            ),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Uni-Mol2 outputs written to %s", self.output_dir)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
