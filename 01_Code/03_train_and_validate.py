#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 04 | Hyperparameter Optimization, Training & Validation
=================================================================================
1. Optuna search over model + training hyper-parameters (validated on a held-out
   scaffold fold of the train pool — no test leakage).
2. Stratified/scaffold 5-fold cross-validation with the best configuration.
3. Retrain the best configuration on the full train pool and evaluate ONCE on the
   independent scaffold-disjoint hold-out test set (13 metrics + publication curves).

Inputs  : 02_Data/03_Features_Raw + 04_Features_Selected + 05_Splits
Outputs : 03_Results/03_Model/deepentxai_best.keras (+ best_hparams.json)
          03_Results/07_Optuna/study.pkl
          03_Results/02_Metrics/{cv_metrics.json, test_metrics.json}
          03_Results/01_Figures/{roc,pr,confusion,calibration,learning}_curve.png
Run     : ~/miniconda3/envs/ENT/bin/python 04_train_and_validate.py
=================================================================================
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, detect_gpu, save_json
from deepentxai.train import (load_matrices, slice_inputs, train_model, OptunaTuner)
from deepentxai.evaluate import compute_metrics, plot_curves
from deepentxai.model import INPUT_ORDER


def main() -> None:
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB4", cfg.dir_logs)
    gpu = detect_gpu()
    if cfg["train"].get("mixed_precision") == "auto" and gpu:
        import keras; keras.mixed_precision.set_global_policy("mixed_float16")
    log.info(f"DEEPENTXAI-04 | GPU={gpu}")

    X, y, split = load_matrices(cfg)
    dims = {m: X[m].shape[1] for m in INPUT_ORDER}
    tr, te, fold = split["train_idx"], split["test_idx"], split["fold_of_train"]
    log.info(f"dims={dims}  train={len(tr)}  test={len(te)}")

    # --- 1. Optuna hyper-parameter optimisation ------------------------------
    log.info(f"Optuna: {cfg['optuna']['n_trials']} trials ...")
    study = OptunaTuner(cfg, X, y, dims, split, log).run()
    best = {**cfg["model"], **study.best_params}
    joblib.dump(study, os.path.join(cfg.dir_optuna, "study.pkl"))
    save_json({"best_params": study.best_params, "best_val_roc_auc": study.best_value},
              os.path.join(cfg.dir_models, "best_hparams.json"))

    # --- 2. 5-fold cross-validation with the best configuration --------------
    n_folds = int(cfg["split"]["n_folds"])
    cv_rows = []
    for f in range(n_folds):
        va_idx = tr[fold == f]; tr_idx = tr[fold != f]
        model, _ = train_model(X, y, dims, best, cfg, tr_idx, va_idx,
                               epochs=cfg["train"]["epochs"], patience=8, verbose=0)
        p = model.predict(slice_inputs(X, va_idx), verbose=0).ravel()
        m = compute_metrics(y[va_idx], p)
        cv_rows.append(m)
        import keras; keras.backend.clear_session()
        log.info(f"  fold {f}: ROC-AUC {m['roc_auc']:.4f}  MCC {m['mcc']:.4f}")
    cv_mean = {k: float(np.mean([r[k] for r in cv_rows])) for k in cv_rows[0]}
    cv_std = {k: float(np.std([r[k] for r in cv_rows])) for k in cv_rows[0]}
    save_json({"per_fold": cv_rows, "mean": cv_mean, "std": cv_std},
              os.path.join(cfg.dir_metrics, "cv_metrics.json"))
    log.info(f"CV ROC-AUC {cv_mean['roc_auc']:.4f} +/- {cv_std['roc_auc']:.4f}")

    # --- 3. Final model on full train pool -> independent hold-out test -------
    inner_va = tr[fold == 0]; inner_tr = tr[fold != 0]           # val for early stopping
    ckpt = os.path.join(cfg.dir_models, "deepentxai_best.keras")
    model, hist = train_model(X, y, dims, best, cfg, inner_tr, inner_va,
                              epochs=cfg["train"]["epochs"],
                              patience=int(cfg["train"]["early_stopping_patience"]),
                              ckpt_path=ckpt, verbose=0)
    model.save(ckpt)
    p_test = model.predict(slice_inputs(X, te), verbose=0).ravel()
    test_metrics = compute_metrics(y[te], p_test)
    save_json({"test": test_metrics, "cv_mean": cv_mean, "cv_std": cv_std,
               "best_params": study.best_params, "n_test": int(len(te))},
              os.path.join(cfg.dir_metrics, "test_metrics.json"))
    plot_curves(y[te], p_test, cfg.dir_figures, history=hist)
    # cache test predictions for scoring/XAI
    np.savez(os.path.join(cfg.dir_predictions, "test_predictions.npz"),
             y_true=y[te], y_prob=p_test, idx=te)

    print("\n================ DEEPENTXAI-04 complete ================")
    print(f"  best val ROC-AUC (Optuna): {study.best_value:.4f}")
    print(f"  5-fold CV ROC-AUC       : {cv_mean['roc_auc']:.4f} +/- {cv_std['roc_auc']:.4f}")
    print("  --- independent hold-out test ---")
    for k in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1", "mcc",
              "recall_sensitivity", "specificity", "cohen_kappa", "brier_score"):
        print(f"    {k:20s}: {test_metrics[k]:.4f}")
    print(f"  model  : {ckpt}")
    print(f"  figures: {cfg.dir_figures}")


if __name__ == "__main__":
    main()
