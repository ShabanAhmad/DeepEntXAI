#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI_FINAL | 15 | External Cross-Source Validation   (reviewer R2.3)
=================================================================================
Reviewer concern: no external dataset evaluation; high accuracy needs external
validation to rule out overfitting / assay-specific bias.

This is the strongest external test available in-house: TRAIN only on ChEMBL
compounds, TEST only on PubChem compounds (a different database with different
assays and submitters), and vice-versa. It is leakage-free by construction --
feature scaling is fit on the training source ONLY, and the model never sees a
single test-source compound during training. The 505 mixed-provenance compounds
are dropped so the two sets are provenance-disjoint.

This does NOT alter the headline model or its reported accuracy; it is an extra,
harder generalisation test. External AUC is expected to be a little below the
internal scaffold-disjoint AUC (a different chemical space) and still demonstrates
real transfer.

Consumes : 02_Data/03_Features_Raw/features.npz, 04_Features_Selected/rdkit_selected.npz
           02_Data/02_Processed/labelled.csv (sources), 03_Model/best_hparams.json
Outputs  : 03_Results/02_Metrics/external_cross_source.json
Run      : DEEPENT_THREADS=8 ~/miniconda3/envs/ENT/bin/python 15_external_cross_source.py
=================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, save_json
from deepentxai.model import INPUT_ORDER
from deepentxai.train import train_model, slice_inputs
from deepentxai.evaluate import compute_metrics


def build_X(feats, rsel, scaler_fit_idx):
    """Binary blocks raw; rdkit + chemberta scaled on the TRAIN-SOURCE rows only."""
    from sklearn.preprocessing import StandardScaler
    rk = rsel["X_rdkit_selected"].astype("float32")
    cb = feats["X_chemberta"].astype("float32")
    s_rk = StandardScaler().fit(rk[scaler_fit_idx])
    s_cb = StandardScaler().fit(cb[scaler_fit_idx])
    return {"morgan": feats["X_morgan"].astype("float32"),
            "rdkit": s_rk.transform(rk).astype("float32"),
            "maccs": feats["X_maccs"].astype("float32"),
            "chemberta": s_cb.transform(cb).astype("float32")}


def run_direction(cfg, feats, rsel, y, src, train_src, test_src, best, log):
    import keras
    tr_all = np.where(src == train_src)[0]
    te = np.where(src == test_src)[0]
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(tr_all)
    n_val = int(0.15 * len(tr_all))
    va, tr = tr_all[:n_val], tr_all[n_val:]

    X = build_X(feats, rsel, tr)                      # scale on TRAIN source rows only
    dims = {m: X[m].shape[1] for m in INPUT_ORDER}
    set_seed(cfg.seed)
    model, _ = train_model(X, y, dims, best, cfg, tr, va,
                           epochs=cfg["train"]["epochs"], patience=8, verbose=0)
    p = model.predict(slice_inputs(X, te), verbose=0).ravel()
    keras.backend.clear_session()
    from sklearn.metrics import roc_auc_score
    m = compute_metrics(y[te], p, threshold=0.5)
    m["roc_auc"] = float(roc_auc_score(y[te], p))
    m["n_train"] = int(len(tr)); m["n_test"] = int(len(te))
    m["train_source"] = train_src; m["test_source"] = test_src
    log.info(f"{train_src}->{test_src}: AUC {m['roc_auc']:.4f} acc {m['accuracy']:.4f} "
             f"(train {len(tr)} / test {len(te)})")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def main() -> None:
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB15", cfg.dir_logs)

    feats = np.load(os.path.join(cfg.dir_features_raw, "features.npz"), allow_pickle=True)
    rsel = np.load(os.path.join(cfg.dir_features_selected, "rdkit_selected.npz"))
    y = feats["y"].astype(int)
    src = pd.read_csv(os.path.join(cfg.dir_processed, "labelled.csv"))["sources"].to_numpy()
    assert len(src) == len(y), "sources/features misalignment"
    best = {**cfg["model"], **json.load(open(
        os.path.join(cfg.dir_models, "best_hparams.json")))["best_params"]}

    results = {}
    results["ChEMBL_to_PubChem"] = run_direction(cfg, feats, rsel, y, src, "ChEMBL", "PubChem", best, log)
    results["PubChem_to_ChEMBL"] = run_direction(cfg, feats, rsel, y, src, "PubChem", "ChEMBL", best, log)

    aucs = [v["roc_auc"] for v in results.values()]
    out = {
        "design": ("Provenance-disjoint external validation. Train on one database, "
                   "test on the other. Scaling fit on the training source only; the "
                   "model never trains on any test-source compound. 505 mixed compounds "
                   "dropped."),
        "internal_scaffold_disjoint_auc": 0.901,
        "external_directions": results,
        "external_mean_auc": round(float(np.mean(aucs)), 4),
        "verdict": (f"Above-chance but MODERATE cross-database transfer: external mean "
                    f"ROC-AUC {round(float(np.mean(aucs)),3)} (well above random 0.5, but "
                    "below the internal scaffold-disjoint 0.901). This indicates a genuine "
                    "domain shift between ChEMBL and PubChem chemical spaces / activity "
                    "definitions. Reported transparently as a limitation; the model learns "
                    "real transferable signal but is not database-agnostic."),
    }
    save_json(out, os.path.join(cfg.dir_metrics, "external_cross_source.json"))

    print("\n================ DEEPENTXAI-15 (external cross-source) ================")
    for k, v in results.items():
        print(f"  {k:20s} AUC {v['roc_auc']}  acc {v['accuracy']}  "
              f"MCC {v['mcc']}  (train {v['n_train']} / test {v['n_test']})")
    print(f"  external mean AUC {out['external_mean_auc']}  (internal 0.901)")
    print(f"  -> {out['verdict']}")


if __name__ == "__main__":
    main()
