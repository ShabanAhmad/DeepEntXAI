"""End-to-end prediction pipeline: raw SMILES -> DEEPENTXAI activity + score.

Loads the persisted artifacts (model, RDKit selector, ChemBERTa scaler) and applies
the exact same standardisation and featurisation used in training, so predictions
on new compounds are fully reproducible.
"""
from __future__ import annotations

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

from .features import FeatureGenerator
from .model import INPUT_ORDER
from .scoring import rank_compounds
from .standardize import MoleculeStandardizer


class PredictionPipeline:
    def __init__(self, cfg):
        import keras
        self.cfg = cfg
        self.std = MoleculeStandardizer(cfg["standardize"])
        self.fg = FeatureGenerator(cfg["features"])
        self.rdkit_selector = joblib.load(os.path.join(cfg.dir_features_selected, "rdkit_selector.joblib"))
        self.chemberta_scaler = joblib.load(os.path.join(cfg.dir_features_selected, "chemberta_scaler.joblib"))
        self.model = keras.models.load_model(os.path.join(cfg.dir_models, "deepentxai_best.keras"))

    def _featurize(self, smiles: List[str]):
        std_smiles, mols = [], []
        for s in smiles:
            canon, _ = self.std(s)
            if canon is None:
                continue
            m = Chem.MolFromSmiles(canon)
            if m is not None:
                std_smiles.append(canon); mols.append(m)
        morgan = np.stack([self.fg.morgan(m) for m in mols]).astype("float32")
        maccs = np.stack([self.fg.maccs(m) for m in mols]).astype("float32")
        rdkit = np.array([self.fg.rdkit(m) for m in mols], dtype=np.float64)
        rdkit = np.where(np.isfinite(rdkit), rdkit, np.nan).astype("float32")
        rdkit_sel = self.rdkit_selector.transform(rdkit)
        chemberta = self.chemberta_scaler.transform(self.fg.chemberta(std_smiles)).astype("float32")
        X = {"morgan": morgan, "rdkit": rdkit_sel, "maccs": maccs, "chemberta": chemberta}
        return std_smiles, [X[m] for m in INPUT_ORDER]

    def predict(self, smiles: List[str]) -> pd.DataFrame:
        std_smiles, X = self._featurize(smiles)
        prob = self.model.predict(X, verbose=0).ravel()
        return rank_compounds(np.array(std_smiles), prob)
