"""DEEPENTXAI compound scoring & ranking.

Turns raw activity probabilities into a 0-100 **DEEPENTXAI Score** that blends the
predicted probability of activity with prediction confidence (distance from the
decision boundary), then ranks compounds for downstream drug-discovery triage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def deepentxai_score(prob: np.ndarray, confidence_weight: float = 0.3) -> np.ndarray:
    """Score in [0, 100]: mostly P(active), boosted by boundary confidence.

    score = 100 * ( (1-w) * p  +  w * p * |2p-1| )
    where |2p-1| is 0 at the 0.5 boundary and 1 at the extremes.
    """
    p = np.clip(prob, 0.0, 1.0)
    conf = np.abs(2 * p - 1.0)
    return 100.0 * ((1 - confidence_weight) * p + confidence_weight * p * conf)


def rank_compounds(smiles: np.ndarray, prob: np.ndarray, y_true: np.ndarray | None = None,
                   confidence_weight: float = 0.3) -> pd.DataFrame:
    """Ranked table: SMILES, P(active), DEEPENTXAI Score, predicted call, (truth)."""
    score = deepentxai_score(prob, confidence_weight)
    df = pd.DataFrame({
        "smiles": smiles,
        "prob_active": np.round(prob, 4),
        "deepentxai_score": np.round(score, 2),
        "prediction": np.where(prob >= 0.5, "Active", "Inactive"),
    })
    if y_true is not None:
        df["observed"] = np.where(np.asarray(y_true) == 1, "Active", "Inactive")
    return df.sort_values("deepentxai_score", ascending=False).reset_index(drop=True)


# =============================================================================
# Retrospective virtual-screening validation of the ranking.
#
# A ranking is only scientifically meaningful if the actives really do pile up at
# the top. On the scaffold-disjoint hold-out (labels known) we quantify that with
# the standard early-recognition metrics used in virtual screening:
#   * Enrichment Factor (EF@x%)  = (hit-rate in the top x%) / (overall active rate).
#                                  EF = 1 means "no better than random"; higher is
#                                  better; the ceiling is 1/active_rate.
#   * hit-rate@x%                = fraction of the top x% that are truly active.
#   * precision@K                = hit-rate among the top-K compounds.
#   * accumulation curve         = actives recovered vs fraction screened (its area
#                                  equals the ranking ROC-AUC).
# All of these are computed on TRUE labels, so they are directly falsifiable.
# =============================================================================
def _ranked_labels(y_true: np.ndarray, score: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    return np.asarray(y_true, dtype=int)[order]


def enrichment(y_true: np.ndarray, score: np.ndarray,
               fractions=(0.01, 0.05, 0.10, 0.20)) -> dict:
    y = _ranked_labels(y_true, score)
    N, n_act = len(y), int(y.sum())
    base = n_act / N if N else float("nan")               # random active rate
    out = {"n_total": int(N), "n_active": int(n_act),
           "active_rate": round(base, 4), "max_EF": round(1.0 / base, 2) if base else float("nan")}
    rows = {}
    for f in fractions:
        k = max(1, int(round(f * N)))
        hits = int(y[:k].sum())
        hit_rate = hits / k
        rows[f"top_{int(f*100)}pct"] = {
            "k": k, "hits": hits,
            "hit_rate": round(hit_rate, 4),
            "EF": round(hit_rate / base, 2) if base else float("nan"),
        }
    out["by_fraction"] = rows
    return out


def precision_at_k(y_true: np.ndarray, score: np.ndarray,
                   ks=(10, 25, 50, 100, 250, 500)) -> dict:
    y = _ranked_labels(y_true, score)
    N = len(y)
    return {f"P@{k}": round(float(y[:min(k, N)].sum()) / min(k, N), 4) for k in ks}


def accumulation_curve(y_true: np.ndarray, score: np.ndarray):
    """Return (fraction_screened, fraction_of_actives_recovered) for plotting."""
    y = _ranked_labels(y_true, score)
    N, n_act = len(y), int(y.sum())
    frac_screened = np.arange(1, N + 1) / N
    frac_found = np.cumsum(y) / (n_act if n_act else 1)
    return frac_screened, frac_found
