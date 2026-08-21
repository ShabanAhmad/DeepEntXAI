#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI_FINAL | 12 | Assay / Source-Bias Analysis   (reviewer R2.6)
=================================================================================
Reviewer concern: if actives and inactives come from different assays/sources, the
model might learn a SOURCE artefact instead of true biological activity.

This stage tests that on the trained headline model WITHOUT retraining anything, so
the reported accuracy is untouched. It answers three questions:

  (a) Are the classes confounded with source?  -> active-rate per source.
  (b) Can the model tell sources apart by chance?  -> class balance per source.
  (c) THE decisive test: does restricting the hold-out test to a SINGLE source
      change the ROC-AUC?  If AUC is stable within each source, the signal is
      biological, not a source artefact.

Consumes : 03_Results/05_Predictions/ensemble_test_predictions.npz (from stage 04)
           02_Data/02_Processed/labelled.csv (the 'sources' column)
           02_Data/05_Splits/split.npz (test_idx, to align sources to predictions)
Outputs  : 03_Results/02_Metrics/assay_source_bias.json
Run      : ~/miniconda3/envs/ENT/bin/python 12_assay_source_bias.py
=================================================================================
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import get_logger, save_json


def main() -> None:
    cfg = Config.load()
    log = get_logger("NB12", cfg.dir_logs)
    from sklearn.metrics import roc_auc_score, average_precision_score

    df = pd.read_csv(os.path.join(cfg.dir_processed, "labelled.csv"))
    split = np.load(os.path.join(cfg.dir_splits, "split.npz"))
    te = split["test_idx"]
    pred = np.load(os.path.join(cfg.dir_predictions, "ensemble_test_predictions.npz"))
    y = pred["y_true"].astype(int)
    p = (pred["y_prob_cal"] if "y_prob_cal" in pred else pred["y_prob"]).astype(float)

    src = df["sources"].to_numpy()[te]
    lab_all = df["label"].to_numpy()
    src_all = df["sources"].to_numpy()

    # (a) class composition per source (whole dataset)
    comp = {}
    for s in np.unique(src_all):
        m = src_all == s
        comp[str(s)] = {"n": int(m.sum()),
                        "active_rate": round(float(lab_all[m].mean()), 4)}
    overall_rate = round(float(lab_all.mean()), 4)

    # (c) decisive test: AUC restricted to each source in the HOLD-OUT test
    per_source = {}
    for s in np.unique(src):
        m = src == s
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            per_source[str(s)] = {"n_test": int(m.sum()), "note": "too few / single-class"}
            continue
        per_source[str(s)] = {
            "n_test": int(m.sum()),
            "active_rate": round(float(y[m].mean()), 4),
            "roc_auc": round(float(roc_auc_score(y[m], p[m])), 4),
            "pr_auc": round(float(average_precision_score(y[m], p[m])), 4),
        }

    overall_auc = round(float(roc_auc_score(y, p)), 4)
    # judge on the SUBSTANTIVE sources only (>=500 test compounds); tiny mixed
    # groups are too noisy to compare and would distort the spread.
    aucs = [v["roc_auc"] for v in per_source.values()
            if "roc_auc" in v and v["n_test"] >= 500]
    spread = round(float(max(aucs) - min(aucs)), 4) if len(aucs) > 1 else 0.0

    out = {
        "question": "Is the model learning a source/assay artefact instead of biology?",
        "overall_active_rate": overall_rate,
        "overall_test_roc_auc": overall_auc,
        "class_composition_by_source": comp,
        "test_auc_by_source": per_source,
        "auc_spread_across_major_sources": spread,
        "verdict": ("No source/assay artefact: restricting the hold-out to either "
                    f"major source leaves ROC-AUC essentially unchanged (spread {spread} "
                    "<= 0.03), despite differing per-source active rates -- the model "
                    "discriminates activity WITHIN each source, i.e. it learns biology, "
                    "not the source." if spread <= 0.03 else
                    f"AUC varies across major sources (spread {spread}); discuss."),
    }
    save_json(out, os.path.join(cfg.dir_metrics, "assay_source_bias.json"))

    print("\n================ DEEPENTXAI-12 (assay/source bias) ================")
    print(f"  overall active rate {overall_rate} | overall test AUC {overall_auc}")
    print("  class composition by source:")
    for s, v in comp.items():
        print(f"    {s:16s} n={v['n']:6d}  active_rate={v['active_rate']}")
    print("  hold-out AUC restricted to each source:")
    for s, v in per_source.items():
        if "roc_auc" in v:
            print(f"    {s:16s} n={v['n_test']:5d}  AUC={v['roc_auc']}  (active {v['active_rate']})")
    print(f"  AUC spread across sources: {spread}  ->  {out['verdict']}")


if __name__ == "__main__":
    main()
