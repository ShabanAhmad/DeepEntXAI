"""Molecular feature generation: Morgan, RDKit descriptors, MACCS, ChemBERTa.

Each representation is computed directly from a (standardised) SMILES string.
ChemBERTa embeddings are produced by a pretrained molecular transformer and cached,
since they are the most expensive step.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors

RDLogger.DisableLog("rdApp.*")

RDKIT_NAMES: List[str] = [d[0] for d in Descriptors.descList]
_RDKIT_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(RDKIT_NAMES)


class FeatureGenerator:
    """Compute the four molecular representations for a list of SMILES."""

    def __init__(self, cfg: dict):
        self.bits = int(cfg["morgan_bits"])
        self.radius = int(cfg["morgan_radius"])
        self.chemberta_model = cfg["chemberta_model"]
        self.max_len = int(cfg["chemberta_max_len"])

    # -- per-molecule fingerprints/descriptors ----------------------------
    def morgan(self, mol) -> np.ndarray:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.bits)
        a = np.zeros((self.bits,), np.float32); DataStructs.ConvertToNumpyArray(bv, a); return a

    def maccs(self, mol) -> np.ndarray:
        bv = MACCSkeys.GenMACCSKeys(mol)
        a = np.zeros((167,), np.float32); DataStructs.ConvertToNumpyArray(bv, a); return a

    def rdkit(self, mol) -> np.ndarray:
        return np.array(_RDKIT_CALC.CalcDescriptors(mol), dtype=np.float64)

    @staticmethod
    def scaffold(mol) -> str:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    # -- ChemBERTa embeddings (batched) -----------------------------------
    def chemberta(self, smiles: List[str], batch: int = 64, log=None) -> np.ndarray:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(self.chemberta_model)
        mod = AutoModel.from_pretrained(self.chemberta_model).eval()
        out = []
        for i in range(0, len(smiles), batch):
            enc = tok(smiles[i:i + batch], padding=True, truncation=True,
                      max_length=self.max_len, return_tensors="pt")
            with torch.no_grad():
                out.append(mod(**enc).last_hidden_state[:, 0, :].numpy())   # CLS token
            if log and i % (batch * 30) == 0:
                log.info(f"  ChemBERTa {i}/{len(smiles)}")
        return np.vstack(out).astype(np.float32)

    # -- full block over a dataframe --------------------------------------
    def generate(self, smiles: List[str], log=None) -> Tuple[dict, np.ndarray]:
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        keep = [i for i, m in enumerate(mols) if m is not None]
        mols = [mols[i] for i in keep]; kept_smiles = [smiles[i] for i in keep]
        if log:
            log.info(f"featurising {len(mols)} molecules")
        feats = {
            "X_morgan": np.stack([self.morgan(m) for m in mols]).astype(np.float32),
            "X_maccs": np.stack([self.maccs(m) for m in mols]).astype(np.float32),
            "X_rdkit": np.array([self.rdkit(m) for m in mols], dtype=np.float64),
            "scaffold": np.array([self.scaffold(m) for m in mols]),
        }
        feats["X_rdkit"] = np.where(np.isfinite(feats["X_rdkit"]), feats["X_rdkit"], np.nan).astype(np.float32)
        if log:
            log.info("computing ChemBERTa embeddings ...")
        feats["X_chemberta"] = self.chemberta(kept_smiles, log=log)
        return feats, np.array(keep)
