#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI_FINAL | 07 | Compound Scoring & Ranking (retrospective virtual screen)
=================================================================================
Turns the calibrated 5-fold ENSEMBLE probabilities into a ranked hit list and,
crucially, VALIDATES that ranking on the scaffold-disjoint hold-out where the true
labels are known. A ranking is only scientifically defensible if the actives
actually concentrate at the top; we prove that with the standard early-recognition
metrics used in virtual screening:

  * DEEPENTXAI Score (0-100)  — P(active) blended with boundary confidence.
  * Enrichment Factor EF@x%   — (hit-rate in the top x%) / (overall active rate).
                                EF = 1 is random; ceiling is 1 / active_rate.
  * hit-rate@x% and precision@K.
  * accumulation curve        — actives recovered vs fraction screened.

Scores use the CALIBRATED ENSEMBLE probabilities (the headline model), so "score 90"
is a meaningful probability, and every validation number is computed on TRUE labels
(hence falsifiable). No leakage: the ranking is on the held-out, scaffold-novel set;
nothing here touches training.

Consumes : 03_Results/05_Predictions/ensemble_test_predictions.npz  (from stage 04)
           02_Data/03_Features_Raw/features.npz + 02_Data/05_Splits/split.npz
Outputs  : 03_Results/08_Rankings/compound_ranking.csv   (all hold-out compounds)
           03_Results/08_Rankings/top_hits.csv           (top 100)
           03_Results/02_Metrics/screening_metrics.json  (EF / hit-rate / precision@K)
           03_Results/01_Figures/enrichment_curve.{png,pdf}
Run      : DEEPENT_THREADS=8 ~/miniconda3/envs/ENT/bin/python 07_compound_scoring_and_ranking.py
=================================================================================
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import get_logger, save_json
from deepentxai.scoring import (rank_compounds, enrichment, precision_at_k,
                                accumulation_curve, deepentxai_score)


def main() -> None:
    cfg = Config.load()
    log = get_logger("NB7scr", cfg.dir_logs)

    # --- load the calibrated ENSEMBLE predictions + aligned SMILES -----------
    pred = np.load(os.path.join(cfg.dir_predictions, "ensemble_test_predictions.npz"))
    y_true = pred["y_true"].astype(int)
    prob = (pred["y_prob_cal"] if "y_prob_cal" in pred else pred["y_prob"]).astype(float)

    split = np.load(os.path.join(cfg.dir_splits, "split.npz"))
    te = split["test_idx"]
    smiles = np.load(os.path.join(cfg.dir_features_raw, "features.npz"),
                     allow_pickle=True)["smiles"][te]
    assert len(smiles) == len(y_true) == len(prob), "test alignment mismatch"
    log.info(f"scoring {len(y_true)} scaffold-disjoint hold-out compounds "
             f"({int(y_true.sum())} active)")

    # --- ranking -------------------------------------------------------------
    ranking = rank_compounds(np.asarray(smiles), prob, y_true)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(os.path.join(cfg.dir_rankings, "compound_ranking.csv"), index=False)
    ranking.head(100).to_csv(os.path.join(cfg.dir_rankings, "top_hits.csv"), index=False)

    # --- retrospective virtual-screening validation --------------------------
    ef = enrichment(y_true, prob, fractions=(0.01, 0.05, 0.10, 0.20))
    pk = precision_at_k(y_true, prob, ks=(10, 25, 50, 100, 250, 500))
    from sklearn.metrics import roc_auc_score, average_precision_score
    metrics = {
        "driver": "calibrated_ensemble_probability",
        "ranking_roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "ranking_pr_auc": round(float(average_precision_score(y_true, prob)), 4),
        "enrichment": ef,
        "precision_at_k": pk,
        "note": ("Retrospective virtual screen on the scaffold-disjoint hold-out. "
                 "EF=1 is random; ceiling is 1/active_rate = "
                 f"{ef['max_EF']}. All figures computed on true labels."),
    }
    save_json(metrics, os.path.join(cfg.dir_metrics, "screening_metrics.json"))

    # --- accumulation (enrichment) curve, titleless --------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fx, fy = accumulation_curve(y_true, prob)
    base = ef["active_rate"]
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot(fx, fy, "-", color="#4c72b0", lw=1.8, label="DEEPENTXAI ranking")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="random")
    # ideal early-recognition curve (all actives first)
    ideal_x = [0, base, 1]; ideal_y = [0, 1, 1]
    ax.plot(ideal_x, ideal_y, ":", color="#55a868", lw=1.2, label="ideal")
    ax.set_xlabel("fraction of compounds screened (ranked by score)")
    ax.set_ylabel("fraction of true actives recovered")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.dir_figures, f"F08_enrichment_curve.{ext}"), dpi=300)
    plt.close(fig)

    # --- console summary -----------------------------------------------------
    print("\n================ DEEPENTXAI-07 (scoring & ranking) ================")
    print(f"  compounds ranked: {ef['n_total']}  |  actives: {ef['n_active']} "
          f"({ef['active_rate']*100:.1f}%)  |  max possible EF: {ef['max_EF']}")
    print(f"  ranking ROC-AUC {metrics['ranking_roc_auc']}  PR-AUC {metrics['ranking_pr_auc']}")
    print(f"  {'top-x%':>8} {'k':>6} {'hits':>6} {'hit-rate':>9} {'EF':>7}")
    for f, r in ef["by_fraction"].items():
        print(f"  {f:>8} {r['k']:>6} {r['hits']:>6} {r['hit_rate']:>9.3f} {r['EF']:>7.2f}")
    print("  precision@K:", {k: v for k, v in pk.items()})
    print(f"  top hit score: {ranking['deepentxai_score'].iloc[0]:.1f} "
          f"(rank 1, observed {ranking['observed'].iloc[0]})")


if __name__ == "__main__":
    main()
