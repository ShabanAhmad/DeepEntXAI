"""Rigorous binary-classification evaluation: 13 metrics + publication figures."""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "log_loss": log_loss(y_true, p),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 120, "font.size": 12,
                         "axes.grid": True, "grid.alpha": 0.3})
    return plt


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, history=None) -> None:
    """ROC, PR, confusion matrix, calibration and (optional) learning curves."""
    from sklearn.metrics import (roc_curve, precision_recall_curve,
                                 ConfusionMatrixDisplay, average_precision_score, roc_auc_score)
    from sklearn.calibration import calibration_curve
    plt = _style()
    os.makedirs(out_dir, exist_ok=True)
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "F02_roc_curve.png")); plt.close(fig)

    pr, rc, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rc, pr, lw=2, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.3f}")
    ax.axhline(y_true.mean(), ls="--", c="grey"); ax.set_xlabel("Recall")
    ax.set_ylabel("Precision"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "F03_pr_curve.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                           display_labels=["Inactive", "Active"]).plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "F05_confusion_matrix.png")); plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean_pred, frac_pos, "o-", label="DEEPENTXAI")
    ax.plot([0, 1], [0, 1], "--", c="grey", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "F04_calibration_curve.png")); plt.close(fig)

    if history is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k in ("loss", "val_loss"):
            if k in history: ax[0].plot(history[k], label=k)
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
        for k in history:
            if "auc" in k.lower(): ax[1].plot(history[k], label=k)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("AUC"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "F06_learning_curve.png")); plt.close(fig)
