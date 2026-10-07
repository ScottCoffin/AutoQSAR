"""Label-free pretrained molecular embeddings for the feature-expansion arm (GPU; benchmark env).

- ``unimol_repr``: unimol_tools ``UniMolRepr`` (pretrained Uni-Mol V1, no fine-tuning), CLS embedding (512-d).
- ``chemeleon``: Chemprop 2.2's CheMeleon foundation model, i.e. the pretrained bond message passing
  from Zenodo record 15460715 (cite arXiv:2506.15792), mean-aggregated over atoms (2048-d). This is
  exactly the encoder ``chemprop train --from-foundation CHEMELEON`` starts from.

Neither embedding saw any benchmark labels, so feeding them to a tree model cannot leak training
targets (unlike fine-tuned embeddings, which would need out-of-fold generation). Molecules that fail
to featurize get a zero vector; the count is reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_PATH = Path.home() / ".chemprop" / "chemeleon_mp.pt"
BATCH = 256


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chemeleon(smiles: list[str]) -> np.ndarray:
    import torch
    from chemprop import data, featurizers, nn
    from chemprop.data import collate_batch
    from rdkit import Chem

    if not CHEMELEON_PATH.exists():
        from urllib.request import urlretrieve

        CHEMELEON_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(CHEMELEON_URL, CHEMELEON_PATH)
    ckpt = torch.load(CHEMELEON_PATH, weights_only=True, map_location="cpu")
    mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    agg = nn.MeanAggregation()
    device = _device()
    mp = mp.to(device).eval()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    dim = int(ckpt["hyper_parameters"].get("d_h", 2048))
    out = np.zeros((len(smiles), dim), dtype=np.float32)
    valid = [i for i, s in enumerate(smiles) if Chem.MolFromSmiles(s) is not None]
    with torch.no_grad():
        for start in range(0, len(valid), BATCH):
            idx = valid[start : start + BATCH]
            points = [data.MoleculeDatapoint.from_smi(smiles[i]) for i in idx]
            dataset = data.MoleculeDataset(points, featurizer=featurizer)
            batch = collate_batch([dataset[k] for k in range(len(dataset))])
            bmg = batch.bmg
            bmg.to(device)
            H = mp(bmg)
            emb = agg(H, bmg.batch)
            out[idx] = emb.detach().cpu().numpy()
    return out


def unimol_repr(smiles: list[str]) -> np.ndarray:
    from unimol_tools import UniMolRepr

    model = UniMolRepr(data_type="molecule", remove_hs=False, model_name="unimolv1", batch_size=32)
    reprs = model.get_repr(list(smiles), return_atomic_reprs=False)
    cls = reprs["cls_repr"] if isinstance(reprs, dict) else reprs
    return np.asarray(cls, dtype=np.float32)


def compute(family: str, smiles: list[str]) -> np.ndarray:
    if family == "chemeleon":
        return chemeleon(smiles)
    if family == "unimol_repr":
        return unimol_repr(smiles)
    raise ValueError(family)
