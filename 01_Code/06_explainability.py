#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI_FINAL | 06 | Explainable AI
=================================================================================
Explain the trained DEEPENTXAI model with FOUR complementary methods. Compound
scoring & ranking is a separate step (stage 07).

  * Permutation importance  (modality-level + RDKit-descriptor-level)
  * Integrated Gradients    (RDKit branch)
  * SHAP                    (GradientExplainer, RDKit branch; guarded)
  * LIME                    (instance-level; guarded)

Inputs  : 03_Results/03_Model/deepentxai_best.keras + 02_Data features/splits
Outputs : 03_Results/04_Explainability/*  (importances, plots, SHAP/LIME)
Run     : ~/miniconda3/envs/ENT/bin/python 06_explainability.py
=================================================================================
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, save_json
from deepentxai.train import load_matrices, slice_inputs
from deepentxai import explain as X_


def _barh(df, value_col, title, path, top=20):
    d = df.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(d.iloc[:, 0], d[value_col], color="#4c72b0", edgecolor="black")
    ax.set_xlabel(value_col); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB5", cfg.dir_logs)
    import keras

    X, y, split = load_matrices(cfg)
    te = split["test_idx"]
    Xte = slice_inputs(X, te); yte = y[te]
    smiles = np.load(os.path.join(cfg.dir_features_raw, "features.npz"), allow_pickle=True)["smiles"][te]
    names = json.load(open(os.path.join(cfg.dir_features_selected, "selected_rdkit_features.json")))["selected_rdkit_features"]
    model = keras.models.load_model(os.path.join(cfg.dir_models, "deepentxai_best.keras"))
    outdir = cfg.dir_explain
    log.info(f"explaining {len(yte)} hold-out compounds")

    # 1) permutation importance (modality)
    mod = X_.permutation_modality(model, Xte, yte, n_repeats=5, seed=cfg.seed)
    save_json(mod, os.path.join(outdir, "modality_importance.json"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(mod.keys()), list(mod.values()), color="#dd8452", edgecolor="black")
    ax.set_ylabel("ROC-AUC drop when permuted"); ax.set_title("Modality importance")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "modality_importance.png"), dpi=300); plt.close(fig)
    log.info(f"modality importance: {mod}")

    # 2) permutation importance (RDKit descriptors)
    perm = X_.permutation_descriptor(model, Xte, yte, names, n_repeats=3, seed=cfg.seed)
    perm.to_csv(os.path.join(outdir, "permutation_descriptor_importance.csv"), index=False)
    _barh(perm, "importance", "Permutation importance (RDKit)", os.path.join(outdir, "permutation_top20.png"))

    # 3) Integrated Gradients (RDKit)
    ig = X_.integrated_gradients_rdkit(model, Xte, names, steps=32, n_samples=300, seed=cfg.seed)
    ig.to_csv(os.path.join(outdir, "integrated_gradients.csv"), index=False)
    _barh(ig, "attribution", "Integrated Gradients (RDKit)", os.path.join(outdir, "integrated_gradients_top20.png"))

    # 4) SHAP + LIME (guarded)
    shap_ok = X_.shap_rdkit(model, Xte, names, outdir, log=log)
    pos = int(np.where(yte == 1)[0][0]); neg = int(np.where(yte == 0)[0][0])
    lime_ok = X_.lime_instances(model, Xte, names, outdir, instances=[pos, neg], log=log)

    print("\n================ DEEPENTXAI-06 (explainability) complete ================")
    print("  modality importance (AUC drop):", {k: round(v, 4) for k, v in mod.items()})
    print("  top-5 RDKit descriptors (permutation):", perm['descriptor'].head(5).tolist())
    print("  top-5 RDKit descriptors (IntGrad)   :", ig['descriptor'].head(5).tolist())
    print(f"  SHAP: {'saved' if shap_ok else 'skipped'} | LIME: {'saved' if lime_ok else 'skipped'}")
    print("  compound scoring + ranking is now stage 07 (07_compound_scoring_and_ranking.py)")


if __name__ == "__main__":
    main()
