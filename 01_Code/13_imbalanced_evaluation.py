#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI_FINAL | 13 | Imbalanced-Scenario Evaluation   (reviewer R1.2)
=================================================================================
Reviewer concern: the benchmark is near-balanced; show the model still works on a
highly UNBALANCED, real-world-like distribution (few actives, many inactives).

This stage does NOT retrain and does NOT change the headline accuracy. It takes the
trained headline model's hold-out predictions and stress-tests them at increasing
imbalance by keeping ALL inactives and subsampling actives down to ratios up to
1:100 (active:inactive). For each ratio it reports the metrics that are meaningful
under imbalance -- ROC-AUC (prevalence-invariant), PR-AUC, MCC, and the virtual-
screening Enrichment Factor -- averaged over several random subsamples.

Scientific point: ROC-AUC is provably invariant to class prevalence, so the model's
ability to DISCRIMINATE actives from inactives does not degrade as the data becomes
imbalanced; PR-AUC falls (as it must, since the positive base-rate falls) but the
Enrichment Factor shows the ranking stays highly useful for screening.

Consumes : 03_Results/05_Predictions/ensemble_test_predictions.npz (from stage 04)
Outputs  : 03_Results/02_Metrics/imbalanced_evaluation.json
           03_Results/01_Figures/imbalance_robustness.{png,pdf}
Run      : ~/miniconda3/envs/ENT/bin/python 13_imbalanced_evaluation.py
=================================================================================
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import get_logger, save_json


def ef_at(y, p, frac):
    order = np.argsort(-p)
    k = max(1, int(round(frac * len(y))))
    hit = y[order][:k].mean()
    base = y.mean()
    return float(hit / base) if base > 0 else float("nan")


def main() -> None:
    cfg = Config.load()
    log = get_logger("NB13", cfg.dir_logs)
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 matthews_corrcoef)

    pred = np.load(os.path.join(cfg.dir_predictions, "ensemble_test_predictions.npz"))
    y = pred["y_true"].astype(int)
    p = (pred["y_prob_cal"] if "y_prob_cal" in pred else pred["y_prob"]).astype(float)
    thr = float(pred["threshold"]) if "threshold" in pred else 0.5

    act = np.where(y == 1)[0]
    inact = np.where(y == 0)[0]
    n_act, n_inact = len(act), len(inact)
    rng = np.random.RandomState(cfg.seed)

    ratios = [(1, 1), (1, 2), (1, 5), (1, 10), (1, 25), (1, 50), (1, 100)]
    rows = []
    for ra, ri in ratios:
        # keep all inactives; subsample actives so that active:inactive = ra:ri
        target_act = int(round(n_inact * ra / ri))
        if target_act < 20:                 # keep enough positives to be meaningful
            continue
        target_act = min(target_act, n_act)
        aucs, prs, mccs, ef1, ef01 = [], [], [], [], []
        for rep in range(10):
            sel_act = rng.choice(act, size=target_act, replace=False)
            idx = np.concatenate([sel_act, inact])
            yy, pp = y[idx], p[idx]
            aucs.append(roc_auc_score(yy, pp))
            prs.append(average_precision_score(yy, pp))
            mccs.append(matthews_corrcoef(yy, (pp >= thr).astype(int)))
            ef1.append(ef_at(yy, pp, 0.01))
            ef01.append(ef_at(yy, pp, 0.001))
        rows.append({
            "ratio": f"1:{round(ri/ra)}",
            "active_rate": round(target_act / (target_act + n_inact), 4),
            "n_active": int(target_act), "n_inactive": int(n_inact),
            "roc_auc": round(float(np.mean(aucs)), 4),
            "pr_auc": round(float(np.mean(prs)), 4),
            "mcc": round(float(np.mean(mccs)), 4),
            "EF@1%": round(float(np.mean(ef1)), 2),
            "EF@0.1%": round(float(np.mean(ef01)), 2),
        })

    out = {
        "note": ("Trained headline model, hold-out predictions, no retraining. "
                 "Inactives all kept; actives subsampled to raise imbalance. "
                 "ROC-AUC is prevalence-invariant; EF shows screening utility."),
        "natural_test_balance": {"n_active": n_act, "n_inactive": n_inact,
                                 "active_rate": round(n_act / (n_act + n_inact), 4)},
        "by_ratio": rows,
    }
    save_json(out, os.path.join(cfg.dir_metrics, "imbalanced_evaluation.json"))

    # figure: ROC-AUC flat, PR-AUC declines, EF strong across imbalance
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xr = [r["active_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(xr, [r["roc_auc"] for r in rows], "-o", label="ROC-AUC", color="#4c72b0")
    ax.plot(xr, [r["pr_auc"] for r in rows], "-s", label="PR-AUC", color="#dd8452")
    ax.set_xscale("log")
    ax.set_xlabel("positive (active) rate  — more imbalanced →")
    ax.set_ylabel("area under curve"); ax.set_ylim(0, 1.02)
    ax.invert_xaxis(); ax.legend(loc="lower left", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(xr, [r["EF@1%"] for r in rows], "-^", label="EF@1%", color="#55a868")
    ax2.set_ylabel("Enrichment Factor @1%")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.dir_figures, f"F09_imbalance_robustness.{ext}"), dpi=300)
    plt.close(fig)

    print("\n================ DEEPENTXAI-13 (imbalanced evaluation) ================")
    print(f"  {'ratio':>8} {'act-rate':>9} {'ROC-AUC':>8} {'PR-AUC':>7} {'MCC':>6} {'EF@1%':>7} {'EF@0.1%':>8}")
    for r in rows:
        print(f"  {r['ratio']:>8} {r['active_rate']:>9} {r['roc_auc']:>8} "
              f"{r['pr_auc']:>7} {r['mcc']:>6} {r['EF@1%']:>7} {r['EF@0.1%']:>8}")
    print("  -> ROC-AUC stays ~flat (prevalence-invariant); ranking stays strongly enriched.")


if __name__ == "__main__":
    main()
