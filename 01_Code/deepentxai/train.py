"""Training utilities for DEEPENTXAI: data assembly, callbacks, class weights,
Optuna hyper-parameter optimisation, and fold/full training — all leakage-free."""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from .model import INPUT_ORDER, build_deepentxai


# --------------------------------------------------------------------------- #
# Data assembly (leakage-free: ChemBERTa scaled on the train pool only)
# --------------------------------------------------------------------------- #
def load_matrices(cfg) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    import joblib
    feats = np.load(os.path.join(cfg.dir_features_raw, "features.npz"), allow_pickle=True)
    rsel = np.load(os.path.join(cfg.dir_features_selected, "rdkit_selected.npz"))
    split = dict(np.load(os.path.join(cfg.dir_splits, "split.npz"), allow_pickle=True))
    y = feats["y"].astype(int)
    tr = split["train_idx"]

    from sklearn.preprocessing import StandardScaler
    sb = StandardScaler().fit(feats["X_chemberta"][tr])            # fit on TRAIN pool only
    joblib.dump(sb, os.path.join(cfg.dir_features_selected, "chemberta_scaler.joblib"))

    X = {"morgan": feats["X_morgan"].astype("float32"),
         "rdkit": rsel["X_rdkit_selected"].astype("float32"),
         "maccs": feats["X_maccs"].astype("float32"),
         "chemberta": sb.transform(feats["X_chemberta"]).astype("float32")}
    return X, y, split


def slice_inputs(X: Dict[str, np.ndarray], idx: np.ndarray) -> List[np.ndarray]:
    # list in the model's canonical INPUT_ORDER (positional match, no name ambiguity)
    return [X[m][idx] for m in INPUT_ORDER]


def class_weights(y: np.ndarray) -> Dict[int, float]:
    n = len(y); npos = max(int(y.sum()), 1); nneg = max(int((y == 0).sum()), 1)
    return {0: n / (2 * nneg), 1: n / (2 * npos)}


# --------------------------------------------------------------------------- #
# Compile + train one model
# --------------------------------------------------------------------------- #
def _compile(model, hp, cfg):
    import keras
    opt_name = hp.get("optimizer", "adam")
    lr = float(hp.get("learning_rate", cfg["train"]["learning_rate"]))
    clip = float(cfg["train"].get("gradient_clipnorm", 1.0))
    opt = {"adam": keras.optimizers.Adam, "nadam": keras.optimizers.Nadam,
           "rmsprop": keras.optimizers.RMSprop}.get(opt_name, keras.optimizers.Adam)(
        learning_rate=lr, clipnorm=clip)
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def train_model(X, y, dims, hp, cfg, tr_idx, va_idx, epochs, patience,
                ckpt_path=None, verbose=0):
    import keras
    model = _compile(build_deepentxai(dims, hp), hp, cfg)
    cbs: List = [keras.callbacks.EarlyStopping("val_loss", patience=patience,
                                               restore_best_weights=True)]
    cbs.append(keras.callbacks.ReduceLROnPlateau("val_loss",
               patience=int(cfg["train"]["reduce_lr_patience"]), factor=0.5, min_lr=1e-6))
    if ckpt_path:
        cbs.append(keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc",
                   mode="max", save_best_only=True))
    cw = class_weights(y[tr_idx]) if cfg["train"].get("class_weight", True) else None
    hist = model.fit(slice_inputs(X, tr_idx), y[tr_idx],
                     validation_data=(slice_inputs(X, va_idx), y[va_idx]),
                     epochs=epochs, batch_size=int(hp.get("batch_size", cfg["train"]["batch_size"])),
                     class_weight=cw, callbacks=cbs, verbose=verbose)
    return model, hist.history


# --------------------------------------------------------------------------- #
# Optuna hyper-parameter search (leakage-free: only the train pool folds used)
# --------------------------------------------------------------------------- #
class OptunaTuner:
    def __init__(self, cfg, X, y, dims, split, log):
        self.cfg, self.X, self.y, self.dims, self.split, self.log = cfg, X, y, dims, split, log
        tr = split["train_idx"]; fold = split["fold_of_train"]
        self.pool = tr
        self.va = tr[fold == 0]                     # validation fold
        self.tr = tr[fold != 0]                     # training folds

    def _space(self, trial) -> dict:
        return {
            "embed_dim": trial.suggest_categorical("embed_dim", [96, 128, 192]),
            "cnn_blocks": trial.suggest_int("cnn_blocks", 1, 3),
            "cnn_filters": trial.suggest_categorical("cnn_filters", [32, 64, 128]),
            "cnn_kernel": trial.suggest_categorical("cnn_kernel", [3, 5]),
            "lstm_units": trial.suggest_categorical("lstm_units", [64, 128]),
            "dense_units": trial.suggest_categorical("dense_units", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "nadam"]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

    def objective(self, trial):
        from sklearn.metrics import roc_auc_score
        hp = self._space(trial)
        model, _ = train_model(self.X, self.y, self.dims, hp, self.cfg,
                               self.tr, self.va, epochs=int(self.cfg["train"]["hpo_epochs"]),
                               patience=5, verbose=0)
        p = model.predict(slice_inputs(self.X, self.va), verbose=0).ravel()
        import keras; keras.backend.clear_session()
        return roc_auc_score(self.y[self.va], p)

    def run(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=self.cfg.seed))
        study.optimize(self.objective, n_trials=int(self.cfg["optuna"]["n_trials"]),
                       timeout=int(self.cfg["optuna"]["timeout_min"]) * 60,
                       show_progress_bar=False)
        self.log.info(f"best val ROC-AUC={study.best_value:.4f}  params={study.best_params}")
        return study
