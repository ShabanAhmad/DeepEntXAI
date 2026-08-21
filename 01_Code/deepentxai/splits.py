"""Leakage-free data splitting.

Produces an independent hold-out test set and Stratified K-Fold indices on the
remaining train pool. With ``strategy='scaffold'`` whole Bemis-Murcko scaffolds
are kept disjoint between train and test (and across CV folds), so structural
analogues never leak across the evaluation boundary — the key requirement for a
credible QSAR benchmark.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedKFold


class DataSplitter:
    def __init__(self, cfg: dict, seed: int = 42):
        self.strategy = cfg["split"]["strategy"]
        self.test_size = float(cfg["split"]["test_size"])
        self.n_folds = int(cfg["split"]["n_folds"])
        self.seed = seed

    # -- scaffold-disjoint hold-out ---------------------------------------
    def _scaffold_holdout(self, scaffolds: np.ndarray, y: np.ndarray):
        groups = defaultdict(list)
        for i, s in enumerate(scaffolds):
            groups[s].append(i)
        rng = np.random.default_rng(self.seed)
        sets = list(groups.values()); rng.shuffle(sets)
        n_test = int(self.test_size * len(scaffolds))
        test: List[int] = []
        for g in sets:
            if len(test) + len(g) <= n_test:
                test.extend(g)
        test = np.array(sorted(test))
        train = np.array(sorted(set(range(len(scaffolds))) - set(test)))
        return train, test

    def _random_holdout(self, y: np.ndarray):
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        train, test = train_test_split(idx, test_size=self.test_size,
                                       stratify=y, random_state=self.seed)
        return np.array(sorted(train)), np.array(sorted(test))

    # -- scaffold-aware CV folds on the train pool ------------------------
    def _scaffold_folds(self, scaffolds: np.ndarray, train: np.ndarray) -> np.ndarray:
        groups = defaultdict(list)
        for pos, i in enumerate(train):
            groups[scaffolds[i]].append(pos)
        rng = np.random.default_rng(self.seed + 1)
        sets = list(groups.values()); rng.shuffle(sets)
        fold_of = np.full(len(train), -1, dtype=int)
        sizes = np.zeros(self.n_folds, dtype=int)
        for g in sorted(sets, key=len, reverse=True):
            f = int(np.argmin(sizes))          # greedy balance
            for pos in g:
                fold_of[pos] = f
            sizes[f] += len(g)
        return fold_of

    def split(self, scaffolds: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        if self.strategy == "scaffold":
            train, test = self._scaffold_holdout(scaffolds, y)
            fold_of = self._scaffold_folds(scaffolds, train)
        else:
            train, test = self._random_holdout(y)
            fold_of = np.full(len(train), -1)
            skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
            for f, (_, va) in enumerate(skf.split(train, y[train])):
                fold_of[va] = f
        return {"train_idx": train, "test_idx": test, "fold_of_train": fold_of,
                "strategy": np.array(self.strategy)}
