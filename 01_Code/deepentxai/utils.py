"""Shared utilities: reproducible seeding, structured logging, small IO helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed every RNG that can affect a DEEPENTXAI result."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def get_logger(name: str, log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Dual (console + file) structured logger; safe to call repeatedly."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:                      # avoid duplicate handlers in notebooks
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def detect_gpu() -> bool:
    """True if TensorFlow can see a GPU (drives mixed-precision decisions)."""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices("GPU")) > 0
    except Exception:
        return False


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
