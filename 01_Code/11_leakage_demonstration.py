#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 10 | LEAKAGE DEMONSTRATION  (random split vs scaffold split)
=================================================================================
!!! DIAGNOSTIC ONLY -- THE RANDOM-SPLIT NUMBER IS NOT A VALID PREDICTIVE RESULT !!!

Same features, same CNN-LSTM, same 5-model ensemble as the honest pipeline. The
ONLY change is the train/test split:

    scaffold-disjoint  -> honest generalisation to NEW chemistry  (~0.79 acc)
    random  (row-wise) -> near-duplicate analogs of the SAME scaffold leak into
                          both train and test, so the model "recognises" test
                          molecules it effectively already saw            (inflated)

The gap between the two is the data leakage that inflated accuracy figures in the
literature are (often unknowingly) reporting. Report the SCAFFOLD number.

Outputs : 03_Results/02_Metrics/leakage_demo.json
Run     : DEEPENT_THREADS=8 ~/miniconda3/envs/ENT/bin/python 10_leakage_demo_random_split.py
=================================================================================
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, save_json
from deepentxai.model import INPUT_ORDER
from deepentxai.train import slice_inputs, train_model
from deepentxai.evaluate import compute_metrics


def main() -> None:
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB10", cfg.dir_logs)
    import keras
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    feats = np.load(os.path.join(cfg.dir_features_raw, "features.npz"), allow_pickle=True)
    rsel = np.load(os.path.join(cfg.dir_features_selected, "rdkit_selected.npz"))
    y = feats["y"].astype(int)
    n = len(y)
    best = {**cfg["model"], **json.load(open(
        os.path.join(cfg.dir_models, "best_hparams.json")))["best_params"]}
    n_folds = int(cfg["split"]["n_folds"])
    test_size = float(cfg["split"]["test_size"])

    # ---- RANDOM stratified split (row-wise) -- THIS IS THE LEAKY SETUP -------
    all_idx = np.arange(n)
    tr, te = train_test_split(all_idx, test_size=test_size, random_state=cfg.seed,
                              stratify=y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    fold = np.empty(len(tr), dtype=int)
    for fi, (_, va_pos) in enumerate(skf.split(tr, y[tr])):
        fold[va_pos] = fi

    # chemberta scaled on the (random) train pool only
    sb = StandardScaler().fit(feats["X_chemberta"][tr])
    X = {"morgan": feats["X_morgan"].astype("float32"),
         "rdkit": rsel["X_rdkit_selected"].astype("float32"),
         "maccs": feats["X_maccs"].astype("float32"),
         "chemberta": sb.transform(feats["X_chemberta"]).astype("float32")}
    dims = {m: X[m].shape[1] for m in INPUT_ORDER}
    log.info(f"RANDOM split: train={len(tr)} test={len(te)}  (LEAKAGE DEMO)")

    # ---- same 5-model ensemble as the honest run ----------------------------
    test_probs = []
    for f in range(n_folds):
        set_seed(cfg.seed + f)
        tr_idx = tr[fold != f]; va_idx = tr[fold == f]
        model, _ = train_model(X, y, dims, best, cfg, tr_idx, va_idx,
                               epochs=cfg["train"]["epochs"], patience=8, verbose=0)
        test_probs.append(model.predict(slice_inputs(X, te), verbose=0).ravel())
        keras.backend.clear_session()
        log.info(f"  fold {f} trained")
    p_te = np.mean(test_probs, axis=0)
    yte = y[te]
    m = compute_metrics(yte, p_te)
    m["roc_auc"] = float(roc_auc_score(yte, p_te))

    # ---- honest scaffold ensemble for side-by-side --------------------------
    scaf = {}
    ens_path = os.path.join(cfg.dir_metrics, "ensemble_test_metrics.json")
    if os.path.exists(ens_path):
        scaf = json.load(open(ens_path)).get("variants", {}).get("ensemble_@0.5", {})

    save_json({"WARNING": "random-split result is a LEAKAGE ARTIFACT, not a valid predictive estimate",
               "random_split_LEAKY": m, "scaffold_split_HONEST": scaf},
              os.path.join(cfg.dir_metrics, "leakage_demo.json"))

    print("\n" + "=" * 66)
    print("  LEAKAGE DEMONSTRATION  (same model, same data, split changed)")
    print("=" * 66)
    print(f"  {'metric':14s} {'RANDOM (leaky)':>16} {'SCAFFOLD (honest)':>18}")
    for k in ("accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc"):
        rv = m.get(k, float('nan')); sv = scaf.get(k, float('nan'))
        print(f"  {k:14s} {rv:16.4f} {sv:18.4f}")
    print("=" * 66)
    print("  >>> The RANDOM column is INFLATED BY LEAKAGE. Do NOT report it as a")
    print("  >>> predictive result. The SCAFFOLD column is the real performance.")
    print("=" * 66)


if __name__ == "__main__":
    main()
