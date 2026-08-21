#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 07 | Publication-Quality Figures & Final Reports
=================================================================================
Consolidate every artifact from Scripts 01-05 into manuscript-ready deliverables:
a metrics table, a summary figure, and a full markdown report describing the
dataset, features, architecture, validation, explainability and top compounds.

Outputs : 03_Results/02_Metrics/publication_metrics_table.csv
          03_Results/01_Figures/07_summary.png
          03_Results/DEEPENTXAI_Report.md
Run     : ~/miniconda3/envs/ENT/bin/python 07_reports_and_figures.py
=================================================================================
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import get_logger


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def main() -> None:
    cfg = Config.load()
    log = get_logger("NB7", cfg.dir_logs)
    M = cfg.dir_metrics
    nb1 = _load(os.path.join(M, "nb1_preprocessing_report.json"))
    nb2 = _load(os.path.join(M, "nb2_feature_report.json"))
    test = _load(os.path.join(M, "test_metrics.json"))
    cv = _load(os.path.join(M, "cv_metrics.json"))
    mod = _load(os.path.join(cfg.dir_explain, "modality_importance.json"))

    # Headline = the 5-fold CNN-LSTM ENSEMBLE (stage 04) if available, else the
    # single tuned hold-out model (stage 03). Keeps the report consistent with the
    # ensemble numbers quoted everywhere else.
    ens = _load(os.path.join(M, "ensemble_test_metrics.json"))
    tm = ens.get("variants", {}).get("ensemble_@0.5") or test.get("test", {})
    headline = "5-fold CNN-LSTM ensemble" if ens.get("variants", {}).get("ensemble_@0.5") else "single hold-out model"
    cvm, cvs = cv.get("mean", {}), cv.get("std", {})

    # --- metrics table -------------------------------------------------------
    order = ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity",
             "specificity", "f1", "roc_auc", "pr_auc", "mcc", "cohen_kappa",
             "log_loss", "brier_score"]
    rows = [{"metric": k,
             "hold_out_test": round(tm.get(k, float('nan')), 4),
             "cv_mean": round(cvm.get(k, float('nan')), 4),
             "cv_std": round(cvs.get(k, float('nan')), 4)} for k in order if k in tm]
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(M, "publication_metrics_table.csv"), index=False)

    # --- summary figure ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1", "mcc"]
    vals = [tm.get(k, 0) for k in keys]
    ax.bar(keys, vals, color="#4c72b0", edgecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("DEEPENTXAI — independent hold-out performance")
    fig.tight_layout(); fig.savefig(os.path.join(cfg.dir_figures, "F12_summary.png"), dpi=300); plt.close(fig)

    # --- markdown report -----------------------------------------------------
    top_desc = []
    p = os.path.join(cfg.dir_explain, "permutation_descriptor_importance.csv")
    if os.path.exists(p):
        top_desc = pd.read_csv(p)["descriptor"].head(10).tolist()
    rk = os.path.join(cfg.dir_rankings, "top_hits.csv")
    n_top = len(pd.read_csv(rk)) if os.path.exists(rk) else 0

    def m(k): return f"{tm.get(k, float('nan')):.3f}"

    rep = f"""# DEEPENTXAI — Final Report

Explainable multimodal fusion **CNN-LSTM** for anti-*Enterobacteriaceae* compound
bioactivity prediction (binary Active/Inactive). Leakage-free, scaffold-disjoint,
reproducible.

## 1. Dataset
- Sources: ChEMBL + PubChem BioAssay (official APIs)
- Unique standardised compounds: **{nb1.get('final', {}).get('unique_compounds', '—')}**
  (active {nb1.get('final', {}).get('active', '—')} / inactive {nb1.get('final', {}).get('inactive', '—')})
- Standardisation: salts stripped, neutralised, canonicalised; conflicts resolved by majority vote
- Activity label: IC50/EC50/Ki/Kd ≤ {cfg['labeling']['affinity_threshold_uM']} µM = Active

## 2. Features
- Morgan ECFP4 (2048) · RDKit descriptors ({nb2.get('selection', {}).get('from_total_rdkit', '—')} → **{nb2.get('selection', {}).get('n_selected_rdkit', '—')}** selected) · MACCS (167) · ChemBERTa (768)
- Selection: variance → correlation → Mutual Information → RFE, **fit on train only**
- Split: **scaffold-disjoint** hold-out ({nb2.get('split', {}).get('n_test', '—')}) + {nb2.get('split', {}).get('n_folds', 5)}-fold CV on {nb2.get('split', {}).get('n_train', '—')} train (disjoint = {nb2.get('split', {}).get('holdout_scaffold_disjoint', '—')})

## 3. Model
- Multimodal fusion CNN-LSTM: per-modality encoders → fusion → BatchNorm → Residual CNN → Channel Attention → BiLSTM → Residual Dense → Sigmoid
- Hyper-parameters selected by **Optuna** ({cfg['optuna']['n_trials']} trials)

## 4. Performance
| | ROC-AUC | PR-AUC | Accuracy | Bal-Acc | F1 | MCC |
|---|---|---|---|---|---|---|
| **Hold-out test** | {m('roc_auc')} | {m('pr_auc')} | {m('accuracy')} | {m('balanced_accuracy')} | {m('f1')} | {m('mcc')} |
| **5-fold CV** | {cvm.get('roc_auc', float('nan')):.3f} ± {cvs.get('roc_auc', 0):.3f} | {cvm.get('pr_auc', float('nan')):.3f} | {cvm.get('accuracy', float('nan')):.3f} | {cvm.get('balanced_accuracy', float('nan')):.3f} | {cvm.get('f1', float('nan')):.3f} | {cvm.get('mcc', float('nan')):.3f} |

Also reported: sensitivity {m('recall_sensitivity')}, specificity {m('specificity')},
Cohen's κ {m('cohen_kappa')}, log-loss {m('log_loss')}, Brier {m('brier_score')}.
Curves: `03_Results/01_Figures/{{roc,pr,confusion,calibration,learning}}_curve.png`.

## 5. Explainability
- Modality importance (permutation AUC-drop): { {k: round(v,4) for k,v in mod.items()} }
- Top RDKit descriptors: {', '.join(top_desc) if top_desc else '—'}
- SHAP summary, Integrated-Gradients and LIME plots in `03_Results/04_Explainability/`

## 6. Compound scoring
- **DEEPENTXAI Score** (0–100) ranks compounds by confidence-weighted P(active);
  top {n_top} hits exported to `03_Results/08_Rankings/top_hits.csv`.

*Generated by DEEPENTXAI Script 07.*
"""
    rep_path = os.path.join(cfg.root, "03_Results", "DEEPENTXAI_Report.md")
    with open(rep_path, "w") as fh:
        fh.write(rep)

    print("\n================ DEEPENTXAI-07 complete ================")
    print(f"  metrics table : {os.path.join(M, 'publication_metrics_table.csv')}")
    print(f"  summary figure: {os.path.join(cfg.dir_figures, 'F12_summary.png')}")
    print(f"  final report  : {rep_path}")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
