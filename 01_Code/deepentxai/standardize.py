"""Professional cheminformatics standardisation of molecular structures.

Canonicalise SMILES, strip salts (largest fragment), neutralise charges, remove
mixtures / invalid molecules, and apply size filters. Every rejection reason is
counted so a reproducible preprocessing report can be generated.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


class MoleculeStandardizer:
    """Convert an input SMILES to a clean, canonical, neutral parent structure."""

    def __init__(self, cfg: dict):
        self.largest_fragment = cfg.get("largest_fragment", True)
        self.neutralize = cfg.get("neutralize", True)
        self.remove_stereo = cfg.get("remove_stereo", False)
        self.min_heavy = int(cfg.get("min_heavy_atoms", 3))
        self.max_heavy = int(cfg.get("max_heavy_atoms", 100))
        self._lfc = rdMolStandardize.LargestFragmentChooser()
        self._unch = rdMolStandardize.Uncharger()
        self.reasons: Counter = Counter()

    def __call__(self, smiles: str) -> Tuple[Optional[str], Optional[float]]:
        m = Chem.MolFromSmiles(str(smiles))
        if m is None:
            self.reasons["invalid_smiles"] += 1
            return None, None
        try:
            if self.largest_fragment:
                m = self._lfc.choose(m)          # remove salts / counter-ions / mixtures
            if self.neutralize:
                m = self._unch.uncharge(m)
            if self.remove_stereo:
                Chem.RemoveStereochemistry(m)
        except Exception:
            self.reasons["standardize_error"] += 1
            return None, None
        n = m.GetNumHeavyAtoms()
        if n < self.min_heavy:
            self.reasons["too_small"] += 1
            return None, None
        if n > self.max_heavy:
            self.reasons["too_large"] += 1
            return None, None
        self.reasons["kept"] += 1
        return Chem.MolToSmiles(m), Descriptors.MolWt(m)

    def report(self) -> dict:
        return dict(self.reasons)
