#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 02 | Feature Engineering
=================================================================================
Generate four complementary molecular representations for every compound:
    Morgan ECFP4 (2048) | RDKit descriptors (~210) | MACCS (167) | ChemBERTa (768)
Create a leakage-free split (scaffold-disjoint hold-out + Stratified K-Fold on the
train pool), then fit numerical-descriptor selection (variance -> correlation ->
Mutual Information -> RFE) on the TRAIN pool only and persist it for prediction.

Inputs  : 02_Data/02_Processed/labelled.csv
Outputs : 02_Data/03_Features_Raw/features.npz      (all raw representations + scaffold)
          02_Data/05_Splits/split.npz               (train/test/fold indices)
          02_Data/04_Features_Selected/rdkit_selector.joblib
          02_Data/04_Features_Selected/selected_rdkit_features.json
          02_Data/04_Features_Selected/rdkit_selected.npz
          03_Results/02_Metrics/nb2_feature_report.json
Run     : ~/miniconda3/envs/ENT/bin/python 02_feature_engineering.py
=================================================================================
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, save_json
from deepentxai.features import FeatureGenerator, RDKIT_NAMES
from deepentxai.splits import DataSplitter
from deepentxai.selection import FeatureSelector


def main() -> None:
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB2", cfg.dir_logs)
    report: dict = {}

    df = pd.read_csv(os.path.join(cfg.dir_processed, "labelled.csv"))
    smiles = df["smiles"].tolist()
    y = df["label"].astype(int).to_numpy()
    log.info(f"loaded {len(df)} compounds  (active {y.mean():.3f})")

    # --- 1. generate all representations -------------------------------------
    fg = FeatureGenerator(cfg["features"])
    feats, keep = fg.generate(smiles, log=log)
    y = y[keep]; smiles = [smiles[i] for i in keep]
    scaffold = feats["scaffold"]
    np.savez_compressed(os.path.join(cfg.dir_features_raw, "features.npz"),
                        y=y, smiles=np.array(smiles), rdkit_cols=np.array(RDKIT_NAMES), **feats)
    report["features"] = {k: list(v.shape) for k, v in feats.items() if k.startswith("X_")}
    log.info(f"features: {report['features']}")

    # --- 2. leakage-free split -----------------------------------------------
    split = DataSplitter(cfg, seed=cfg.seed).split(scaffold, y)
    np.savez_compressed(os.path.join(cfg.dir_splits, "split.npz"), **split)
    tr, te = split["train_idx"], split["test_idx"]
    # verify scaffold disjointness of the hold-out
    disjoint = len(set(scaffold[tr]) & set(scaffold[te])) == 0
    report["split"] = {"strategy": str(split["strategy"]), "n_train": int(len(tr)),
                       "n_test": int(len(te)), "n_folds": int(cfg["split"]["n_folds"]),
                       "holdout_scaffold_disjoint": bool(disjoint)}
    log.info(f"split: {report['split']}")
    assert disjoint or cfg["split"]["strategy"] != "scaffold", "hold-out not scaffold-disjoint"

    # --- 3. RDKit descriptor selection (fit on TRAIN pool only) ---------------
    sel = FeatureSelector(cfg, feature_names=list(RDKIT_NAMES), seed=cfg.seed)
    sel.fit(feats["X_rdkit"][tr], y[tr])
    X_rdkit_sel = sel.transform(feats["X_rdkit"])          # applied to ALL rows (fit on train)
    joblib.dump(sel, os.path.join(cfg.dir_features_selected, "rdkit_selector.joblib"))
    save_json({"selected_rdkit_features": sel.selected_names_,
               "n_selected": len(sel.selected_names_)},
              os.path.join(cfg.dir_features_selected, "selected_rdkit_features.json"))
    np.savez_compressed(os.path.join(cfg.dir_features_selected, "rdkit_selected.npz"),
                        X_rdkit_selected=X_rdkit_sel)
    report["selection"] = {"n_selected_rdkit": len(sel.selected_names_),
                           "from_total_rdkit": len(RDKIT_NAMES)}
    log.info(f"selection: {report['selection']}")

    save_json(report, os.path.join(cfg.dir_reports, "nb2_feature_report.json"))
    print("\n================ DEEPENTXAI-02 complete ================")
    print(f"  molecules       : {len(y)}")
    print(f"  Morgan/RDKit/MACCS/ChemBERTa : {feats['X_morgan'].shape[1]}/"
          f"{feats['X_rdkit'].shape[1]}/{feats['X_maccs'].shape[1]}/{feats['X_chemberta'].shape[1]}")
    print(f"  selected RDKit  : {len(sel.selected_names_)} / {len(RDKIT_NAMES)}")
    print(f"  split           : {report['split']['n_train']} train / {report['split']['n_test']} test "
          f"(scaffold-disjoint={report['split']['holdout_scaffold_disjoint']})")


if __name__ == "__main__":
    main()
