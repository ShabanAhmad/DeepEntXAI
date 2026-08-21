"""Leakage-free numerical-descriptor selection for the RDKit descriptor block.

Pipeline (fit on TRAIN only, then applied to every split):
    median impute -> variance filter -> correlation filter (|r|>thr)
    -> Mutual Information pre-ranking -> Recursive Feature Elimination -> scale.
The fitted transformer + selected feature names are persisted so the exact same
descriptors are reproduced at prediction time.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    def __init__(self, cfg: dict, feature_names: List[str], seed: int = 42):
        self.var_thr = float(cfg["features"]["variance_threshold"])
        self.corr_thr = float(cfg["features"]["correlation_threshold"])
        self.n_select = int(cfg["features"]["n_select"])
        self.names = list(feature_names)
        self.seed = seed
        self.imputer_: SimpleImputer | None = None
        self.keep_idx_: np.ndarray | None = None      # indices into original columns
        self.scaler_: StandardScaler | None = None
        self.selected_names_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        Xi = self.imputer_.transform(X)

        # 1) variance filter
        var = Xi.var(axis=0)
        vmask = var > self.var_thr
        idx = np.where(vmask)[0]

        # 2) correlation filter (greedy drop of the later of any |r|>thr pair)
        Xc = Xi[:, idx]
        corr = np.corrcoef(Xc, rowvar=False)
        corr = np.nan_to_num(corr)
        drop = set()
        for i in range(corr.shape[0]):
            if i in drop:
                continue
            for j in range(i + 1, corr.shape[0]):
                if j not in drop and abs(corr[i, j]) > self.corr_thr:
                    drop.add(j)
        idx = idx[[k for k in range(len(idx)) if k not in drop]]

        # 3) Mutual Information pre-ranking -> top 2*n_select
        Xm = Xi[:, idx]
        mi = mutual_info_classif(Xm, y, random_state=self.seed)
        top = np.argsort(mi)[::-1][: max(self.n_select * 2, self.n_select)]
        idx = idx[top]

        # 4) RFE with a fast linear estimator down to n_select
        k = min(self.n_select, idx.shape[0])
        Xr = Xi[:, idx]
        rfe = RFE(LogisticRegression(max_iter=1000, class_weight="balanced"),
                  n_features_to_select=k, step=0.1).fit(Xr, y)
        idx = idx[rfe.support_]

        self.keep_idx_ = np.array(sorted(idx))
        self.selected_names_ = [self.names[i] for i in self.keep_idx_]
        self.scaler_ = StandardScaler().fit(Xi[:, self.keep_idx_])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xi = self.imputer_.transform(X)
        return self.scaler_.transform(Xi[:, self.keep_idx_]).astype(np.float32)
