#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 11 | E. coli whole-cell MIC REGRESSION (pMIC)
=================================================================================
Reframe as a REGRESSION QSAR on a single coherent endpoint: predict pMIC of
whole-cell E. coli growth inhibition. Single species, single endpoint (fixes the
affinity/MIC pooling critique), exact '=' measurements only, censored '>' handled
by exclusion, per-structure median (robust to 2-fold assay noise), scaffold-
disjoint split (no leakage).

Multimodal fusion CNN-LSTM with a LINEAR head + MSE. Reports R2/RMSE/MAE/Pearson/
Spearman with bootstrap 95% CIs, RF/XGB regressor baselines, and a 5-model
ensemble. All transforms fit on TRAIN only.

Input   : 02_Data/06_Regression/labelled_reg.csv  (smiles, pmic, n_meas)
Outputs : 03_Results/02_Metrics/ecolimic_regression.json + 03_Results/01_Figures/regression_scatter.*
Run     : DEEPENT_THREADS=8 ~/miniconda3/envs/ENT/bin/python 11_regression_ecoli_mic.py
=================================================================================
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, save_json
from deepentxai.features import FeatureGenerator

INPUT_ORDER = ["morgan", "rdkit", "maccs", "chemberta"]


# ---------------- scaffold-disjoint split (whole Murcko groups) --------------
def scaffold_split(scaffolds, test_frac=0.15, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    groups = {}
    for i, s in enumerate(scaffolds):
        groups.setdefault(s, []).append(i)
    keys = list(groups.keys()); rng.shuffle(keys)
    n = len(scaffolds); n_test = int(test_frac * n)
    test = []
    for k in keys:
        if len(test) >= n_test: break
        test += groups[k]
    test = np.array(sorted(test)); train = np.array(sorted(set(range(n)) - set(test)))
    # scaffold-aware folds on train
    tr_keys = [k for k in keys if all(i in set(train) for i in groups[k])]
    fold_of = np.full(len(train), -1); pos = {g: p for p, g in enumerate(train)}
    tk = [k for k in keys if k in set(scaffolds[train])]
    rng.shuffle(tk)
    for j, k in enumerate(tk):
        for i in groups[k]:
            if i in pos: fold_of[pos[i]] = j % n_folds
    fold_of[fold_of == -1] = 0
    return train, test, fold_of


def metrics(y, p):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return {"r2": float(r2_score(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "pearson": float(pearsonr(y, p)[0]), "spearman": float(spearmanr(y, p)[0])}


def boot_ci(y, p, fn, n=2000, seed=42):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y)); vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def build_reg_model(dims, hp):
    import keras
    from keras import layers as L
    ins, encs = [], []
    for m in INPUT_ORDER:
        x = keras.Input(shape=(dims[m],), name=m); ins.append(x)
        encs.append(L.Dense(hp["embed_dim"], activation=hp["activation"])(x))
    h = L.Concatenate()(encs); h = L.BatchNormalization()(h)
    h = L.Dense(64 * 8, activation=hp["activation"])(h); h = L.Reshape((64, 8))(h)
    for _ in range(hp["cnn_blocks"]):
        r = h
        h = L.Conv1D(hp["cnn_filters"], hp["cnn_kernel"], padding="same", activation=hp["activation"])(h)
        h = L.Conv1D(8, hp["cnn_kernel"], padding="same")(h)
        h = L.Add()([r, h]); h = L.Activation(hp["activation"])(h)
    # squeeze-excite channel attention
    s = L.GlobalAveragePooling1D()(h); s = L.Dense(8 // 2 or 1, activation="relu")(s)
    s = L.Dense(8, activation="sigmoid")(s); h = L.Multiply()([h, L.Reshape((1, 8))(s)])
    h = L.Bidirectional(L.LSTM(hp["lstm_units"]))(h)
    d = L.Dense(hp["dense_units"], activation=hp["activation"])(h)
    d = L.Dense(hp["dense_units"] // 2, activation=hp["activation"])(d); d = L.Add()([d, d])
    d = L.Dropout(hp["dropout"])(d)
    out = L.Dense(1, dtype="float32")(d)                       # LINEAR head
    mdl = keras.Model(ins, out)
    mdl.compile(optimizer=keras.optimizers.Nadam(hp["learning_rate"], clipnorm=1.0), loss="mse")
    return mdl


def main():
    cfg = Config.load(); set_seed(cfg.seed)
    log = get_logger("NB11", cfg.dir_logs)
    import keras
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor
    from rdkit import Chem

    df = pd.read_csv(os.path.join(cfg.root, "02_Data", "06_Regression", "labelled_reg.csv"))
    y = df["pmic"].values.astype("float32"); smiles = df["smiles"].tolist()
    log.info(f"{len(df)} compounds  pMIC mean={y.mean():.2f} std={y.std():.2f}")

    fg = FeatureGenerator(cfg["features"])
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ok = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in ok]; smiles = [smiles[i] for i in ok]; y = y[ok]
    morgan = np.stack([fg.morgan(m) for m in mols])
    maccs = np.stack([fg.maccs(m) for m in mols])
    rdk = np.array([fg.rdkit(m) for m in mols], dtype=np.float64)
    rdk = np.where(np.isfinite(rdk), rdk, np.nan).astype("float32")
    scaf = np.array([fg.scaffold(m) or smiles[i] for i, m in enumerate(mols)])  # M2 fix: fallback
    log.info("computing ChemBERTa embeddings ...")
    cberta = fg.chemberta(smiles, log=log)

    tr, te, fold = scaffold_split(scaf, seed=cfg.seed)
    # scaffold-disjoint sanity
    assert len(set(scaf[tr]) & set(scaf[te])) == 0, "scaffold leak!"
    log.info(f"train={len(tr)} test={len(te)} scaffold-disjoint=True")

    # RDKit: impute+select(MI regression, top100)+scale  — TRAIN ONLY
    imp = SimpleImputer(strategy="median").fit(rdk[tr]); rdk_i = imp.transform(rdk)
    mi = mutual_info_regression(rdk_i[tr], y[tr], random_state=cfg.seed)
    keep = np.argsort(mi)[::-1][:100]
    sc_r = StandardScaler().fit(rdk_i[tr][:, keep]); rdk_s = sc_r.transform(rdk_i[:, keep])
    sc_b = StandardScaler().fit(cberta[tr]); cb_s = sc_b.transform(cberta)
    X = {"morgan": morgan.astype("float32"), "rdkit": rdk_s.astype("float32"),
         "maccs": maccs.astype("float32"), "chemberta": cb_s.astype("float32")}
    dims = {m: X[m].shape[1] for m in INPUT_ORDER}

    hp = {"embed_dim": 128, "cnn_blocks": 2, "cnn_filters": 64, "cnn_kernel": 3,
          "lstm_units": 128, "dense_units": 256, "dropout": 0.3, "activation": "gelu",
          "learning_rate": 5e-4}

    def sl(idx): return [X[m][idx] for m in INPUT_ORDER]

    # 5-model scaffold-fold ensemble
    test_preds = []
    for f in range(5):
        set_seed(cfg.seed + f)
        tri = tr[fold != f]; vai = tr[fold == f]
        mdl = build_reg_model(dims, hp)
        cb = [keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau("val_loss", patience=5, factor=0.5, min_lr=1e-6)]
        mdl.fit(sl(tri), y[tri], validation_data=(sl(vai), y[vai]),
                epochs=120, batch_size=64, callbacks=cb, verbose=0)
        test_preds.append(mdl.predict(sl(te), verbose=0).ravel())
        keras.backend.clear_session(); log.info(f"  fold {f} trained")
    p_ens = np.mean(test_preds, axis=0)

    # baselines on concatenated features
    Xflat = np.concatenate([X[m] for m in INPUT_ORDER], axis=1)
    rf = RandomForestRegressor(n_estimators=400, n_jobs=8, random_state=cfg.seed).fit(Xflat[tr], y[tr])
    p_rf = rf.predict(Xflat[te])
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.6, n_jobs=8, random_state=cfg.seed, tree_method="hist").fit(Xflat[tr], y[tr])
        p_xgb = xgb.predict(Xflat[te])
    except Exception:
        p_xgb = None

    yte = y[te]
    out = {"n_total": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
           "pmic_std": float(y.std())}
    for name, p in [("cnn_lstm_ensemble", p_ens), ("randomforest", p_rf)] + ([("xgboost", p_xgb)] if p_xgb is not None else []):
        m = metrics(yte, p)
        m["r2_ci"] = boot_ci(yte, p, lambda a, b: __import__("sklearn.metrics", fromlist=["r2_score"]).r2_score(a, b))
        m["pearson_ci"] = boot_ci(yte, p, lambda a, b: __import__("scipy.stats", fromlist=["pearsonr"]).pearsonr(a, b)[0])
        out[name] = m
    save_json(out, os.path.join(cfg.dir_metrics, "ecolimic_regression.json"))

    # titleless scatter: predicted vs measured pMIC (ensemble)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(yte, p_ens, s=8, alpha=0.35, color="#4c72b0", edgecolors="none")
    lo, hi = min(yte.min(), p_ens.min()), max(yte.max(), p_ens.max())
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
    ax.set_xlabel("measured pMIC"); ax.set_ylabel("predicted pMIC")
    m = out["cnn_lstm_ensemble"]
    ax.text(0.05, 0.92, f"R$^2$={m['r2']:.3f}  r={m['pearson']:.3f}\nRMSE={m['rmse']:.2f}",
            transform=ax.transAxes, fontsize=10, va="top")
    figdir = cfg._mk("03_Results", "01_Figures")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(figdir, f"F11_regression_scatter.{ext}"), dpi=300)
    plt.close(fig)

    print("\n================ DEEPENTXAI-11 (E. coli MIC regression) ================")
    print(f"  {len(y)} compounds | train {len(tr)} / test {len(te)} (scaffold-disjoint)")
    print(f"  {'model':20s} {'R2':>7} {'Pearson':>8} {'Spearman':>9} {'RMSE':>7} {'MAE':>7}")
    for name in ("cnn_lstm_ensemble", "randomforest", "xgboost"):
        if name in out:
            m = out[name]
            print(f"  {name:20s} {m['r2']:7.3f} {m['pearson']:8.3f} {m['spearman']:9.3f} {m['rmse']:7.3f} {m['mae']:7.3f}")
    m = out["cnn_lstm_ensemble"]
    print(f"  ensemble R2 95% CI {m['r2_ci']}  Pearson 95% CI {m['pearson_ci']}")


if __name__ == "__main__":
    main()
