#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 09 | Stacked Ensemble (CNN-LSTM + XGBoost + RandomForest)
=================================================================================
Combine three complementary base learners with a logistic meta-learner, trained
ONLY on out-of-fold (OOF) train predictions -> leakage-free. The hold-out test is
scored once at the end. Base learners disagree in different regions, so stacking
typically adds ~0.01-0.03 ROC-AUC over any single model.

  base 1 : DEEPENTXAI CNN-LSTM  (multimodal, per-fold, OOF + test)
  base 2 : XGBoost              (flat 3083-dim feature vector)
  base 3 : RandomForest         (flat 3083-dim feature vector)
  meta   : LogisticRegression on the three OOF probability columns

Outputs : 03_Results/02_Metrics/stacked_test_metrics.json
          03_Results/03_Model/stack_meta.joblib
Run     : DEEPENT_THREADS=8 ~/miniconda3/envs/ENT/bin/python 09_stacked_ensemble.py
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


def best_threshold(y, p):
    from sklearn.metrics import accuracy_score
    grid = np.linspace(0.05, 0.95, 181)
    vals = [accuracy_score(y, (p >= t).astype(int)) for t in grid]
    j = int(np.argmax(vals))
    return float(grid[j]), float(vals[j])


def main() -> None:
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB9", cfg.dir_logs)
    import keras
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y, split = load_matrices(cfg)
    dims = {m: X[m].shape[1] for m in INPUT_ORDER}
    tr, te, fold = split["train_idx"], split["test_idx"], split["fold_of_train"]
    best = {**cfg["model"], **json.load(open(
        os.path.join(cfg.dir_models, "best_hparams.json")))["best_params"]}
    n_folds = int(cfg["split"]["n_folds"])

    # flat feature matrix for the tree learners (all modalities concatenated)
    Xflat = np.concatenate([X[m] for m in INPUT_ORDER], axis=1)
    log.info(f"stacking: flat dim={Xflat.shape[1]}  train={len(tr)}  test={len(te)}")

    y_oof = y[tr]
    oof = {k: np.zeros(len(tr)) for k in ("cnn", "xgb", "rf")}
    tst = {k: [] for k in ("cnn", "xgb", "rf")}

    for f in range(n_folds):
        tr_idx = tr[fold != f]; va_idx = tr[fold == f]     # GLOBAL indices into X / Xflat
        pos_va = np.where(fold == f)[0]                     # positions into the train-pool arrays (oof)

        # base 1: CNN-LSTM
        set_seed(cfg.seed + f)
        model, _ = train_model(X, y, dims, best, cfg, tr_idx, va_idx,
                               epochs=cfg["train"]["epochs"], patience=8, verbose=0)
        oof["cnn"][pos_va] = model.predict(slice_inputs(X, va_idx), verbose=0).ravel()
        tst["cnn"].append(model.predict(slice_inputs(X, te), verbose=0).ravel())
        keras.backend.clear_session()

        # base 2 + 3: trees on the flat vector (index Xflat with GLOBAL indices)
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.6, eval_metric="logloss",
                            n_jobs=8, random_state=cfg.seed + f, tree_method="hist")
        xgb.fit(Xflat[tr_idx], y[tr_idx])
        oof["xgb"][pos_va] = xgb.predict_proba(Xflat[va_idx])[:, 1]
        tst["xgb"].append(xgb.predict_proba(Xflat[te])[:, 1])

        rf = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=8,
                                    max_features="sqrt", random_state=cfg.seed + f)
        rf.fit(Xflat[tr_idx], y[tr_idx])
        oof["rf"][pos_va] = rf.predict_proba(Xflat[va_idx])[:, 1]
        tst["rf"].append(rf.predict_proba(Xflat[te])[:, 1])
        log.info(f"  fold {f}: cnn/xgb/rf OOF done")

    yte = y[te]
    test_base = {k: np.mean(v, axis=0) for k, v in tst.items()}     # mean over folds
    Z_oof = np.column_stack([oof["cnn"], oof["xgb"], oof["rf"]])
    Z_te = np.column_stack([test_base["cnn"], test_base["xgb"], test_base["rf"]])

    # meta-learner on OOF only
    meta = LogisticRegression(max_iter=1000).fit(Z_oof, y_oof)
    joblib.dump(meta, os.path.join(cfg.dir_models, "stack_meta.joblib"))
    p_oof = meta.predict_proba(Z_oof)[:, 1]
    p_te = meta.predict_proba(Z_te)[:, 1]
    t_acc, oof_acc = best_threshold(y_oof, p_oof)
    log.info(f"meta coefs (cnn,xgb,rf)={meta.coef_.ravel().round(3).tolist()}  "
             f"tuned t={t_acc:.3f} OOF acc={oof_acc:.4f}")

    # score hold-out once
    results = {}
    for name, p, t in [
        ("cnn_lstm_@0.5",  test_base["cnn"], 0.5),
        ("xgboost_@0.5",   test_base["xgb"], 0.5),
        ("randomforest_@0.5", test_base["rf"], 0.5),
        ("stack_@0.5",     p_te, 0.5),
        ("stack_@tuned",   p_te, t_acc),
    ]:
        m = compute_metrics(yte, p, threshold=t)
        m["roc_auc"] = float(roc_auc_score(yte, p))
        results[name] = m

    save_json({"variants": results, "meta_coef": meta.coef_.ravel().tolist(),
               "threshold": t_acc, "n_test": int(len(te))},
              os.path.join(cfg.dir_metrics, "stacked_test_metrics.json"))

    print("\n================ DEEPENTXAI-09 complete ================")
    print(f"  {'model':22s} {'ACC':>7} {'BAL-ACC':>8} {'F1':>7} {'MCC':>7} {'ROC-AUC':>8} {'PR-AUC':>7}")
    for name, m in results.items():
        print(f"  {name:22s} {m['accuracy']:7.4f} {m['balanced_accuracy']:8.4f} "
              f"{m['f1']:7.4f} {m['mcc']:7.4f} {m['roc_auc']:8.4f} {m['pr_auc']:7.4f}")


if __name__ == "__main__":
    main()
