#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 13 | Conformal Selective Prediction (honest high-confidence)
=================================================================================
The scaffold-disjoint full-coverage ceiling on this phenotypic antibacterial data
is ~0.90 ROC-AUC / ~0.84 accuracy -- limited by the LABELS themselves (replicate
MIC measurements of the same compound disagree ~12% of the time). 98-99% at full
coverage is therefore not physically attainable without leakage.

The legitimate route to a high-accuracy statement is SELECTIVE PREDICTION: let the
model abstain on its least-confident compounds and report accuracy on the ones it
does call, together with the coverage. Everything here is calibrated on OUT-OF-FOLD
train predictions only; the hold-out test is scored once. Two rules:

  (A) confidence cutoff : pick the cutoff tau on OOF so OOF coverage = target c,
      apply tau to test, report the realised test coverage + accuracy.
  (B) Mondrian conformal: class-conditional nonconformity quantile on OOF at level
      alpha; a test point is 'accepted' iff its score <= q_class. Distribution-free
      finite-sample validity, and class-conditional so it does not abstain on one
      class only.

HONESTY RULE baked into the output: the full-coverage row (coverage = 1.0) is
always reported next to every selective row, and every row states its coverage.

Consumes : 03_Results/05_Predictions/ensemble_test_predictions.npz
           (needs oof_true, oof_prob, y_true, y_prob -- produced by script 08)
Outputs  : 03_Results/02_Metrics/conformal_selective.json
           03_Results/01_Figures/risk_coverage.{png,pdf}
Run      : DEEPENT_VARIANT=gapfull DEEPENT_RAW_DIR=... DEEPENT_CONFIG=... \
           ~/miniconda3/envs/ENT/bin/python 13_conformal_selective.py
=================================================================================
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import get_logger, save_json


# --- SECTION 1 : selective helpers ------------------------------------------
def decision(p, t=0.5):
    return (p >= t).astype(int)


def confidence(p):
    """Distance from the 0.5 boundary, mapped to [0.5, 1]. Higher = more confident."""
    return np.maximum(p, 1.0 - p)


def acc(y, yhat):
    return float(np.mean(y == yhat)) if len(y) else float("nan")


def selective_by_confidence(oof_y, oof_p, te_y, te_p, coverages, t=0.5):
    """Rule A: cutoff chosen on OOF to hit target coverage, applied to test."""
    oof_conf = confidence(oof_p)
    te_conf = confidence(te_p)
    te_hat = decision(te_p, t)
    rows = []
    for c in coverages:
        # tau = the confidence below which we abstain, set so OOF keeps fraction c
        tau = float(np.quantile(oof_conf, 1.0 - c)) if c < 1.0 else float(oof_conf.min())
        keep = te_conf >= tau
        rows.append({
            "target_coverage": round(float(c), 3),
            "tau": round(tau, 4),
            "test_coverage": round(float(keep.mean()), 4),
            "n_kept": int(keep.sum()),
            "accuracy": round(acc(te_y[keep], te_hat[keep]), 4),
        })
    return rows


def mondrian_conformal(oof_y, oof_p, te_y, te_p, alphas, t=0.5):
    """Rule B: class-conditional (Mondrian) conformal. score = 1 - p(pred class)."""
    oof_hat = decision(oof_p, t)
    oof_pred_prob = np.where(oof_hat == 1, oof_p, 1.0 - oof_p)
    oof_score = 1.0 - oof_pred_prob                      # nonconformity

    te_hat = decision(te_p, t)
    te_pred_prob = np.where(te_hat == 1, te_p, 1.0 - te_p)
    te_score = 1.0 - te_pred_prob

    rows = []
    for a in alphas:
        accept = np.zeros(len(te_y), dtype=bool)
        for k in (0, 1):
            s_k = oof_score[oof_hat == k]
            if len(s_k) == 0:
                continue
            n = len(s_k)
            q = float(np.quantile(s_k, min(1.0, np.ceil((n + 1) * (1 - a)) / n),
                                  method="higher"))
            accept |= (te_hat == k) & (te_score <= q)
        rows.append({
            "alpha": round(float(a), 3),
            "test_coverage": round(float(accept.mean()), 4),
            "n_accepted": int(accept.sum()),
            "accuracy": round(acc(te_y[accept], te_hat[accept]), 4),
        })
    return rows


# --- SECTION 2 : main -------------------------------------------------------
def main() -> None:
    cfg = Config.load()
    log = get_logger("NB13", cfg.dir_logs)

    npz = os.path.join(cfg.dir_predictions, "ensemble_test_predictions.npz")
    d = np.load(npz)
    if "oof_prob" not in d:
        log.error(f"{npz} lacks OOF arrays -- rerun script 08 (patched) first.")
        sys.exit(1)
    oof_y, oof_p = d["oof_true"].astype(int), d["oof_prob"].astype(float)
    te_y, te_p = d["y_true"].astype(int), d["y_prob"].astype(float)
    log.info(f"loaded OOF n={len(oof_y)}  test n={len(te_y)}")

    full = acc(te_y, decision(te_p))
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(te_y, te_p))

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ruleA = selective_by_confidence(oof_y, oof_p, te_y, te_p, coverages)
    ruleB = mondrian_conformal(oof_y, oof_p, te_y, te_p,
                               alphas=[0.30, 0.25, 0.20, 0.15, 0.10, 0.05])

    out = {
        "note": ("Selective prediction. Full-coverage row is the honest headline; "
                 "selective rows trade coverage for accuracy. All cutoffs calibrated "
                 "on OOF train predictions only."),
        "full_coverage": {"coverage": 1.0, "accuracy": round(full, 4),
                          "roc_auc": round(auc, 4), "n_test": int(len(te_y))},
        "rule_A_confidence_cutoff": ruleA,
        "rule_B_mondrian_conformal": ruleB,
    }
    save_json(out, os.path.join(cfg.dir_metrics, "conformal_selective.json"))

    # --- risk-coverage figure (titleless) -----------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cov = [r["test_coverage"] for r in ruleA]
    ac = [r["accuracy"] for r in ruleA]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(cov, ac, "-o", color="#4c72b0", lw=1.6, ms=5)
    ax.axhline(full, ls="--", color="grey", lw=1)
    ax.text(0.98, full + 0.004, f"full-coverage acc = {full:.3f}",
            ha="right", va="bottom", fontsize=9, color="grey")
    for r in ruleA:
        if r["target_coverage"] in (0.5, 0.2):
            ax.annotate(f"{r['accuracy']:.3f}\n@cov {r['test_coverage']:.2f}",
                        (r["test_coverage"], r["accuracy"]),
                        textcoords="offset points", xytext=(6, -18), fontsize=8)
    ax.set_xlabel("coverage (fraction of test compounds called)")
    ax.set_ylabel("accuracy on called compounds")
    ax.set_xlim(0, 1.02); ax.invert_xaxis()
    fig.tight_layout()
    figdir = cfg.dir_figures
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"F07_risk_coverage.{ext}"), dpi=300)
    plt.close(fig)

    # --- console summary ----------------------------------------------------
    print("\n================ DEEPENTXAI-13 (conformal selective) ================")
    print(f"  full-coverage:  acc {full:.4f}   ROC-AUC {auc:.4f}   n={len(te_y)}")
    print(f"  {'target-cov':>10} {'test-cov':>9} {'accuracy':>9}   (rule A: OOF confidence cutoff)")
    for r in ruleA:
        print(f"  {r['target_coverage']:10.2f} {r['test_coverage']:9.3f} {r['accuracy']:9.4f}")
    print(f"  {'alpha':>10} {'test-cov':>9} {'accuracy':>9}   (rule B: Mondrian conformal)")
    for r in ruleB:
        print(f"  {r['alpha']:10.2f} {r['test_coverage']:9.3f} {r['accuracy']:9.4f}")
    print("  HONEST HEADLINE: report the selective number ALWAYS beside full-coverage.")


if __name__ == "__main__":
    main()
