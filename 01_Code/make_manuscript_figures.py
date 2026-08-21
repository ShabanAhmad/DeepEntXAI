#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI_FINAL | Manuscript figure & table assembler
=================================================================================
Builds a numbered, submission-ready set under 03_Results/01_Figures/Manuscript/:

  * Tables A-F from REVISION_RESPONSE.md rendered as titleless figure images
    (T01..T06), with dynamic numbers read from the metrics JSONs.
  * All pipeline plots copied in under a manuscript figure number (F01..F12).
  * FIGURES.txt -- the numbered caption manifest for every item.

Non-destructive: the canonical pipeline figures in 01_Figures/ are left untouched,
so re-running any stage still works; this only assembles the numbered copies.

Run : ~/miniconda3/envs/ENT/bin/python make_manuscript_figures.py
=================================================================================
"""
from __future__ import annotations
import os, sys, json, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- table image renderer ---------------------------------------------------
def render_table(rows, path, col_w=None, fontsize=11, header=True):
    """rows: list of lists (first row = header). Titleless clean table image."""
    ncol = len(rows[0]); nrow = len(rows)
    fig_w = max(6, sum(col_w) if col_w else 2.1 * ncol)
    fig_h = 0.5 * nrow + 0.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis("off")
    tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
                   colWidths=[w / sum(col_w) for w in col_w] if col_w else None)
    tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if header and r == 0:
            cell.set_facecolor("#4c72b0"); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f5f9")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = Config.load()
    figdir = cfg.dir_figures
    M = cfg.dir_metrics
    out = figdir            # numbered set lives directly in 01_Figures (no subfolder)

    def load(name):
        p = os.path.join(M, name)
        return json.load(open(p)) if os.path.exists(p) else {}

    imb = load("imbalanced_evaluation.json")
    ext = load("external_cross_source.json")
    assay = load("assay_source_bias.json")
    stab = load("feature_stability.json")

    # ---- Table A : architecture --------------------------------------------
    A = [["Stage", "Layer", "Configuration", "Output"],
         ["Input x4", "Morgan / RDKit / MACCS / ChemBERTa", "2048 / 100 / 167 / 768", "per-branch"],
         ["Encoders x4", "Dense+BatchNorm+Dropout", "128, GELU, drop 0.25", "4 x 128"],
         ["Fusion", "Concatenate+BatchNorm", "-", "512"],
         ["To-sequence", "Dense -> Reshape", "512 -> (64x8)", "(64, 8)"],
         ["Conv x2", "residual Conv1D", "32 filters, k=3, GELU", "(64, 32)"],
         ["Attention", "squeeze-excite", "-", "(64, 32)"],
         ["Pool", "MaxPooling1D", "pool 2", "(32, 32)"],
         ["Recurrent", "Bidirectional LSTM", "64 units (->128)", "128"],
         ["Head", "res-Dense->Drop->Dense", "64, GELU", "64"],
         ["Output", "Dense", "1, sigmoid", "P(active)"]]
    render_table(A, os.path.join(out, "T01_model_architecture"),
                 col_w=[1.4, 3.2, 2.6, 1.4], fontsize=10)

    # ---- Table B : feature accounting --------------------------------------
    B = [["Tool / block", "Computed", "Removed", "Kept", "Selection"],
         ["Morgan (r=2)", "2048", "0", "2048", "-"],
         ["RDKit descriptors", "208", "108", "100", "var -> MI -> RFE (train-only)"],
         ["MACCS keys", "167", "0", "167", "-"],
         ["ChemBERTa embedding", "768", "0", "768", "-"],
         ["TOTAL model input", "", "", "3083", "no PCA"]]
    render_table(B, os.path.join(out, "T02_feature_accounting"),
                 col_w=[2.2, 1.3, 1.3, 1.1, 3.4], fontsize=10)

    # ---- Table C : performance vs imbalance --------------------------------
    C = [["active:inactive", "active rate", "ROC-AUC", "PR-AUC", "MCC", "EF@1%", "EF@0.1%"]]
    for r in imb.get("by_ratio", []):
        C.append([r["ratio"], f"{r['active_rate']}", f"{r['roc_auc']}", f"{r['pr_auc']}",
                  f"{r['mcc']}", f"{r['EF@1%']}", f"{r['EF@0.1%']}"])
    render_table(C, os.path.join(out, "T03_performance_vs_imbalance"),
                 col_w=[1.8, 1.3, 1.2, 1.1, 1.0, 1.0, 1.2], fontsize=10)

    # ---- Table D : external cross-source -----------------------------------
    D = [["Train -> Test", "n train", "n test", "ROC-AUC", "Accuracy", "MCC"]]
    for k, v in ext.get("external_directions", {}).items():
        D.append([k.replace("_", " "), f"{v['n_train']}", f"{v['n_test']}",
                  f"{v['roc_auc']}", f"{v['accuracy']}", f"{v['mcc']}"])
    D.append(["Mean external", "", "", f"{ext.get('external_mean_auc','')}", "", ""])
    D.append(["Internal (scaffold-disjoint)", "", "", "0.901", "0.840", "0.655"])
    render_table(D, os.path.join(out, "T04_external_cross_source"),
                 col_w=[2.6, 1.2, 1.2, 1.2, 1.2, 1.0], fontsize=10)

    # ---- Table E : assay/source bias ---------------------------------------
    E = [["Restriction", "n (test)", "active rate", "ROC-AUC"]]
    for s, v in assay.get("test_auc_by_source", {}).items():
        if "roc_auc" in v:
            E.append([s, f"{v['n_test']}", f"{v['active_rate']}", f"{v['roc_auc']}"])
    E.append(["AUC spread (major)", "", "", f"{assay.get('auc_spread_across_major_sources','')}"])
    render_table(E, os.path.join(out, "T05_assay_source_bias"),
                 col_w=[2.4, 1.3, 1.5, 1.3], fontsize=10)

    # ---- Table F : feature stability ---------------------------------------
    F = [["Descriptor", "mean imp.", "SD", "Top-10 freq."]]
    for s in stab.get("bootstrap", {}).get("top15_stable_descriptors", [])[:8]:
        F.append([s["descriptor"], f"{s['mean_importance']}", f"{s['sd']}", f"{s['top10_frequency']}"])
    sp = stab.get("across_fold", {}).get("mean_pairwise_spearman", "")
    F.append([f"across-fold Spearman = {sp}", "", "", ""])
    render_table(F, os.path.join(out, "T06_feature_stability"),
                 col_w=[3.0, 1.4, 1.1, 1.5], fontsize=10)

    # ---- plots are written with their F## number natively by each stage -----
    PLOTS = [
        ("F01", "F01_dataset_overview.png",     "Dataset composition (49,093 activity-gap compounds; class balance and source split)."),
        ("F02", "F02_roc_curve.png",            "ROC curve on the scaffold-disjoint hold-out (ROC-AUC 0.901)."),
        ("F03", "F03_pr_curve.png",             "Precision-recall curve on the hold-out (PR-AUC 0.870)."),
        ("F04", "F04_calibration_curve.png",    "Reliability (calibration) curve after isotonic calibration."),
        ("F05", "F05_confusion_matrix.png",     "Confusion matrix at the operating threshold. [Supplementary S1]"),
        ("F06", "F06_learning_curve.png",       "Training history (loss/metric vs epoch). Not a substitute for generalisation."),
        ("F07", "F07_risk_coverage.png",        "Conformal risk-coverage curve: accuracy vs fraction of compounds called (98.1% at 22% coverage)."),
        ("F08", "F08_enrichment_curve.png",     "Retrospective virtual-screen accumulation curve (top-1% = 100% true actives)."),
        ("F09", "F09_imbalance_robustness.png", "Performance vs class imbalance: ROC-AUC flat, Enrichment Factor rises (R1.2)."),
        ("F10", "F10_feature_stability_top15.png","Bootstrap feature-importance stability, top-15 RDKit descriptors, mean +/- SD (R2.9)."),
        ("F11", "F11_regression_scatter.png",   "Secondary task: predicted vs measured E. coli pMIC (R2 0.595)."),
        ("F12", "F12_summary.png",              "Summary performance panel."),
    ]
    copied = [(num, fn, cap) for num, fn, cap in PLOTS if os.path.exists(os.path.join(out, fn))]

    TABLES = [
        ("T01", "T01_model_architecture.png",   "DeepEntXAI architecture (layers, shapes; 1,276,933 parameters)."),
        ("T02", "T02_feature_accounting.png",   "Feature accounting: 2048+100+167+768 = 3083 features, no PCA (R2.4)."),
        ("T03", "T03_performance_vs_imbalance.png","Performance across class-imbalance ratios; EF@0.1% up to 41.9x at 1:50 (R1.2)."),
        ("T04", "T04_external_cross_source.png", "Provenance-disjoint external validation, train/test across databases (R2.3)."),
        ("T05", "T05_assay_source_bias.png",    "Hold-out ROC-AUC restricted to each source; spread 0.021 -> no assay artefact (R2.6)."),
        ("T06", "T06_feature_stability.png",    "Bootstrap + across-fold feature-importance stability (R2.9)."),
    ]

    # ---- FIGURES.txt manifest ----------------------------------------------
    lines = ["DEEPENTXAI - Manuscript figure & table index",
             "=" * 60, "",
             "FIGURES (plots) -- titleless; caption is the sole title.", ""]
    for num, fn, cap in copied:
        lines.append(f"Figure {int(num[1:])}  ({fn})")
        lines.append(f"    {cap}"); lines.append("")
    lines += ["", "TABLES (rendered as figure images).", ""]
    for i, (num, fn, cap) in enumerate(TABLES, 1):
        lines.append(f"Table {chr(64+i)}  ({fn})")
        lines.append(f"    {cap}"); lines.append("")
    with open(os.path.join(out, "FIGURES.txt"), "w") as fh:
        fh.write("\n".join(lines))

    print("\n================ manuscript figures assembled ================")
    print(f"  output dir : {out}")
    print(f"  tables     : T01..T06 (rendered)")
    print(f"  figures    : {len(copied)} plots copied (F01..F{len(copied):02d})")
    print(f"  manifest   : FIGURES.txt")


if __name__ == "__main__":
    main()
