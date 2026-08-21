"""Automatic, configurable binary activity labelling from experimental values.

Default: Active if IC50/EC50/Ki/Kd <= 10 uM (configurable). MIC is handled on its
own ug/mL scale. Ambiguous records are dropped. Per-structure conflicts are
resolved by majority vote (ties dropped), yielding one binary label per compound.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# unit -> factor to micromolar (uM)
_TO_UM = {"nM": 1e-3, "uM": 1.0, "µM": 1.0, "mM": 1e3, "M": 1e6, "pM": 1e-6, "fM": 1e-9}
_AFFINITY = {"IC50", "EC50", "Ki", "Kd"}


class ActivityLabeler:
    """Turn (type, value, units, MW) rows into a binary Active/Inactive label."""

    def __init__(self, cfg: dict):
        self.thr_uM = float(cfg["affinity_threshold_uM"])
        self.inactive_uM = float(cfg.get("affinity_inactive_uM", cfg["affinity_threshold_uM"]))
        self.mic_active = float(cfg["mic_active_ugml"])
        self.mic_inactive = float(cfg["mic_inactive_ugml"])
        self.drop_ambiguous = bool(cfg.get("drop_ambiguous", True))
        self.conflict_policy = cfg.get("conflict_policy", "majority")

    # -- per-measurement label --------------------------------------------
    def label_one(self, typ: str, value, units, mw: Optional[float]) -> Optional[int]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if v <= 0:
            return None
        if typ in _AFFINITY:
            if units not in _TO_UM:
                return None
            uM = v * _TO_UM[units]
            if uM <= self.thr_uM:
                return 1
            if uM > self.inactive_uM:
                return 0
            return None
        if typ == "MIC":
            if units in ("ug.mL-1", "ug/mL"):
                ugml = v
            elif units in _TO_UM and mw:                 # molar MIC -> ug/mL
                ugml = (v * _TO_UM[units] * 1e-6) * mw * 1e6
            else:
                return None
            if ugml <= self.mic_active:
                return 1
            if ugml >= self.mic_inactive:
                return 0
            return None
        return None

    # -- aggregate to one label per structure -----------------------------
    def resolve(self, df: pd.DataFrame, smiles_col: str = "smiles", label_col: str = "lab") -> pd.DataFrame:
        """Majority vote per unique structure; drop 50/50 ties."""
        g = df.groupby(smiles_col)[label_col]
        agg = g.mean().reset_index(name="frac_active")
        agg["n_measurements"] = g.count().values
        if self.conflict_policy == "majority":
            agg = agg[agg["frac_active"] != 0.5]
        agg["label"] = (agg["frac_active"] > 0.5).astype(int)
        return agg[[smiles_col, "label", "n_measurements", "frac_active"]]
