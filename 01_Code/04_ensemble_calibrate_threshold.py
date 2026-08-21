#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 08 | Ensemble + Calibration + Threshold Optimisation
=================================================================================
Three LEGITIMATE, leakage-free levers to raise the *accuracy* of the final model
without touching the hold-out test during any tuning step:

  1. 5-model ENSEMBLE      - one model per CV fold; test prob = mean of the five.
                             Averaging cuts variance -> small ROC-AUC gain.
  2. Probability CALIBRATION - isotonic regression fit on OUT-OF-FOLD train
                             predictions (never on test).
  3. THRESHOLD optimisation - the operating point that maximises accuracy on the
                             out-of-fold train predictions, then applied to test.
                             (A 0.5 cut is wrong here: probabilities peak near 0.39.)

Everything tuned on out-of-fold TRAIN predictions; the hold-out test is scored
exactly once, at the end, with the frozen calibrator + threshold.

Outputs : 03_Results/02_Metrics/ensemble_test_metrics.json
          03_Results/03_Model/calibrator.joblib + operating_point.json
          03_Results/05_Predictions/ensemble_test_predictions.npz
Run     : DEEPENT_THREADS=8 ~/miniconda3/envs/ENT/bin/python 08_ensemble_calibrate_threshold.py
=================================================================================
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, save_json
from deepentxai.model import INPUT_ORDER
from deepentxai.train import load_matrices, slice_inputs, train_model
from deepentxai.evaluate import compute_metrics


def best_threshold(y, p, objective="accuracy"):
    """Threshold on [0.05,0.95] maximising accuracy or balanced accuracy."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    score = accuracy_score if objective == "accuracy" else balanced_accuracy_score
    grid = np.linspace(0.05, 0.95, 181)
    vals = [score(y, (p >= t).astype(int)) for t in grid]
    j = int(np.argmax(vals))
    return float(grid[j]), float(vals[j])


def main() -> None:
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB8", cfg.dir_logs)
    import keras

    X, y, split = load_matrices(cfg)
    dims = {m: X[m].shape[1] for m in INPUT_ORDER}
    tr, te, fold = split["train_idx"], split["test_idx"], split["fold_of_train"]
    best = {**cfg["model"], **json.load(open(
        os.path.join(cfg.dir_models, "best_hparams.json")))["best_params"]}
    n_folds = int(cfg["split"]["n_folds"])
    log.info(f"ensemble over {n_folds} folds  dims={dims}  test={len(te)}")

    # --- train one model per fold: OOF train preds + per-fold test preds ------
    oof_prob = np.zeros(len(tr), dtype="float64")
    test_prob_folds = []
    for f in range(n_folds):
        set_seed(cfg.seed + f)                         # decorrelate the ensemble
        va_idx = tr[fold == f]; tr_idx = tr[fold != f]
        model, _ = train_model(X, y, dims, best, cfg, tr_idx, va_idx,
                               epochs=cfg["train"]["epochs"], patience=8, verbose=0)
        oof_prob[fold == f] = model.predict(slice_inputs(X, va_idx), verbose=0).ravel()
        test_prob_folds.append(model.predict(slice_inputs(X, te), verbose=0).ravel())
        model.save(os.path.join(cfg.dir_models, f"ensemble_fold{f}.keras"))   # persist the 5 headline weights
        keras.backend.clear_session()
        log.info(f"  fold {f} trained ({len(tr_idx)} train / {len(va_idx)} val)")

    y_oof = y[tr]
    test_prob = np.mean(test_prob_folds, axis=0)       # ENSEMBLE (mean of 5)
    yte = y[te]

    # --- calibrate on OOF, then choose threshold on calibrated OOF -----------
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_prob, y_oof)
    joblib.dump(iso, os.path.join(cfg.dir_models, "calibrator.joblib"))
    oof_cal = iso.predict(oof_prob)
    test_cal = iso.predict(test_prob)

    t_acc, oof_acc = best_threshold(y_oof, oof_cal, "accuracy")
    t_bal, oof_bal = best_threshold(y_oof, oof_cal, "balanced")
    save_json({"threshold_accuracy": t_acc, "threshold_balanced": t_bal,
               "oof_accuracy_at_t": oof_acc, "oof_balanced_at_t": oof_bal},
              os.path.join(cfg.dir_models, "operating_point.json"))
    log.info(f"OOF-tuned threshold (max acc) = {t_acc:.3f} -> OOF acc {oof_acc:.4f}")

    # --- score the hold-out test exactly once, four ways ---------------------
    from sklearn.metrics import roc_auc_score
    def acc_at(p, t):
        from sklearn.metrics import accuracy_score
        return accuracy_score(yte, (p >= t).astype(int))

    single = np.load(os.path.join(cfg.dir_predictions, "test_predictions.npz"))["y_prob"]
    variants = {
        "single_model_@0.5":        (single,   0.5),
        "ensemble_@0.5":            (test_prob, 0.5),
        "ensemble_cal_@0.5":        (test_cal,  0.5),
        "ensemble_cal_@tuned":      (test_cal,  t_acc),
    }
    results = {}
    for name, (p, t) in variants.items():
        m = compute_metrics(yte, p, threshold=t)
        results[name] = m

    final = compute_metrics(yte, test_cal, threshold=t_acc)
    final["roc_auc"] = float(roc_auc_score(yte, test_cal))   # calibration is monotone; AUC ~ ensemble
    save_json({"final": final, "variants": results,
               "threshold": t_acc, "n_models": n_folds, "n_test": int(len(te))},
              os.path.join(cfg.dir_metrics, "ensemble_test_metrics.json"))
    np.savez(os.path.join(cfg.dir_predictions, "ensemble_test_predictions.npz"),
             y_true=yte, y_prob=test_prob, y_prob_cal=test_cal, threshold=t_acc,
             oof_true=y_oof, oof_prob=oof_prob, oof_prob_cal=oof_cal)

    print("\n================ DEEPENTXAI-08 complete ================")
    print(f"  tuned threshold (OOF, max-accuracy): {t_acc:.3f}")
    print(f"  {'variant':24s} {'ACC':>7} {'BAL-ACC':>8} {'F1':>7} {'MCC':>7} {'ROC-AUC':>8}")
    for name, m in results.items():
        print(f"  {name:24s} {m['accuracy']:7.4f} {m['balanced_accuracy']:8.4f} "
              f"{m['f1']:7.4f} {m['mcc']:7.4f} {m['roc_auc']:8.4f}")
    print(f"\n  ACCURACY: {results['single_model_@0.5']['accuracy']:.4f} (baseline) "
          f"-> {results['ensemble_cal_@tuned']['accuracy']:.4f} (ensemble+calibrated+tuned)")


if __name__ == "__main__":
    main()
