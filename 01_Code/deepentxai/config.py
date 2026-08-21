"""Typed configuration for DEEPENTXAI_FINAL — numbered, self-contained layout.

Layout (root = DeepEntXAI_Final):
    01_Code/{config.yaml, thresholds.yaml, deepentxai/*, 00..07 numbered stages}
    02_Data/{01_Raw, 02_Processed, 03_Features_Raw, 04_Features_Selected, 05_Splits}
    03_Results/{01_Figures, 02_Metrics, 03_Model, 04_Explainability,
                05_Predictions, 06_Logs, 07_Optuna, 08_Rankings}

Every directory carries a numeric prefix so the pipeline order is unambiguous.
The engine scripts refer to directories only through the `dir_*` properties below,
so re-homing the whole project is a matter of editing this one file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


def _root() -> str:
    # this file: <root>/01_Code/deepentxai/config.py  ->  root is 3 levels up
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)
    root: str = field(default_factory=_root)
    variant: str = ""          # retained for API compatibility; unused in the final layout

    @classmethod
    def load(cls, path: str | None = None, variant: str | None = None) -> "Config":
        root = _root()
        path = path or os.environ.get("DEEPENT_CONFIG") or os.path.join(root, "01_Code", "config.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"config not found: {path}")
        with open(path) as fh:
            return cls(raw=yaml.safe_load(fh), root=root, variant="")

    def __getitem__(self, k: str) -> Any: return self.raw[k]
    def get(self, k: str, d: Any = None) -> Any: return self.raw.get(k, d)

    @property
    def seed(self) -> int: return int(self.raw.get("seed", 42))

    def _mk(self, *parts: str) -> str:
        p = os.path.join(self.root, *parts)
        os.makedirs(p, exist_ok=True)
        return p + os.sep

    # --- 02_Data ---
    @property
    def dir_raw(self) -> str:
        override = os.environ.get("DEEPENT_RAW_DIR")
        if override:
            os.makedirs(override, exist_ok=True)
            return override + os.sep
        return self._mk("02_Data", "01_Raw")
    @property
    def dir_processed(self) -> str: return self._mk("02_Data", "02_Processed")
    @property
    def dir_features_raw(self) -> str: return self._mk("02_Data", "03_Features_Raw")
    @property
    def dir_features_selected(self) -> str: return self._mk("02_Data", "04_Features_Selected")
    @property
    def dir_splits(self) -> str: return self._mk("02_Data", "05_Splits")

    # --- 03_Results ---
    @property
    def dir_figures(self) -> str: return self._mk("03_Results", "01_Figures")
    @property
    def dir_metrics(self) -> str: return self._mk("03_Results", "02_Metrics")
    @property
    def dir_reports(self) -> str: return self._mk("03_Results", "02_Metrics")
    @property
    def dir_models(self) -> str: return self._mk("03_Results", "03_Model")
    @property
    def dir_explain(self) -> str: return self._mk("03_Results", "04_Explainability")
    @property
    def dir_predictions(self) -> str: return self._mk("03_Results", "05_Predictions")
    @property
    def dir_logs(self) -> str: return self._mk("03_Results", "06_Logs")
    @property
    def dir_optuna(self) -> str: return self._mk("03_Results", "07_Optuna")
    @property
    def dir_rankings(self) -> str: return self._mk("03_Results", "08_Rankings")
