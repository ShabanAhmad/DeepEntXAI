"""Multi-method explainability for DEEPENTXAI.

  * Permutation importance  — modality-level and RDKit-descriptor-level (robust).
  * Integrated Gradients    — per-descriptor attribution on the RDKit branch (TF).
  * SHAP                    — GradientExplainer over the RDKit branch (guarded).
  * LIME                    — instance-level tabular explanations (guarded).

The RDKit descriptor block is the human-interpretable channel, so global feature
explanations target it; the other modalities are summarised at the block level.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

# canonical input order (must match the model)
_ORDER = ["morgan", "rdkit", "maccs", "chemberta"]
_RDKIT = 1


def _predict(model, X: List[np.ndarray]) -> np.ndarray:
    return model.predict(X, verbose=0).ravel()


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #
def permutation_modality(model, X: List[np.ndarray], y: np.ndarray,
                         n_repeats: int = 5, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, _predict(model, X))
    out = {}
    for m in range(len(X)):
        drops = []
        for _ in range(n_repeats):
            Xp = [x.copy() for x in X]
            Xp[m] = Xp[m][rng.permutation(len(y))]
            drops.append(base - roc_auc_score(y, _predict(model, Xp)))
        out[_ORDER[m]] = float(np.mean(drops))
    return out


def permutation_descriptor(model, X: List[np.ndarray], y: np.ndarray,
                           names: List[str], n_repeats: int = 3, seed: int = 42):
    import pandas as pd
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, _predict(model, X))
    imp = np.zeros(X[_RDKIT].shape[1])
    for j in range(X[_RDKIT].shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = [x.copy() for x in X]
            col = Xp[_RDKIT][:, j].copy()
            Xp[_RDKIT][:, j] = col[rng.permutation(len(y))]
            drops.append(base - roc_auc_score(y, _predict(model, Xp)))
        imp[j] = np.mean(drops)
    return (pd.DataFrame({"descriptor": names, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Integrated Gradients (RDKit branch)
# --------------------------------------------------------------------------- #
def integrated_gradients_rdkit(model, X: List[np.ndarray], names: List[str],
                               steps: int = 32, n_samples: int = 300, seed: int = 42):
    import pandas as pd
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X[0]), size=min(n_samples, len(X[0])), replace=False)
    Xs = [tf.convert_to_tensor(x[idx]) for x in X]
    baseline = [tf.zeros_like(t) for t in Xs]
    total = tf.zeros_like(Xs[_RDKIT])
    for a in np.linspace(0, 1, steps):
        interp = [baseline[k] + a * (Xs[k] - baseline[k]) for k in range(len(Xs))]
        with tf.GradientTape() as tape:
            tape.watch(interp[_RDKIT])
            out = model(interp, training=False)
        g = tape.gradient(out, interp[_RDKIT])
        total += g
    ig = (Xs[_RDKIT] - baseline[_RDKIT]) * (total / steps)
    attr = tf.reduce_mean(tf.abs(ig), axis=0).numpy()
    return (pd.DataFrame({"descriptor": names, "attribution": attr})
            .sort_values("attribution", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# SHAP (guarded) — GradientExplainer over the RDKit branch
# --------------------------------------------------------------------------- #
def shap_rdkit(model, X: List[np.ndarray], names: List[str], out_dir: str,
               n_bg: int = 100, n_explain: int = 200, seed: int = 42, log=None):
    try:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(seed)
        bg = [x[rng.choice(len(x), n_bg, replace=False)] for x in X]
        ex = [x[rng.choice(len(x), n_explain, replace=False)] for x in X]
        sv = shap.GradientExplainer(model, bg).shap_values(ex)
        sv_rdkit = sv[_RDKIT] if isinstance(sv, list) else sv
        sv_rdkit = np.array(sv_rdkit).reshape(len(ex[_RDKIT]), -1)
        shap.summary_plot(sv_rdkit, ex[_RDKIT], feature_names=names, show=False, max_display=20)
        plt.tight_layout(); plt.savefig(f"{out_dir}/shap_summary.png", dpi=300, bbox_inches="tight"); plt.close()
        return True
    except Exception as e:
        if log: log.info(f"SHAP skipped: {type(e).__name__}: {e}")
        return False


# --------------------------------------------------------------------------- #
# LIME (guarded) — a few representative instances
# --------------------------------------------------------------------------- #
def lime_instances(model, X: List[np.ndarray], names: List[str], out_dir: str,
                   instances: List[int], log=None):
    try:
        from lime.lime_tabular import LimeTabularExplainer
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rdk = X[_RDKIT]

        def predict_fn(z):                       # vary RDKit, hold others at their row values
            n = len(z); reps = []
            for k in range(len(X)):
                reps.append(np.repeat(X[k][:1], n, axis=0) if k != _RDKIT else z)
            p = model.predict(reps, verbose=0).ravel()
            return np.column_stack([1 - p, p])

        expl = LimeTabularExplainer(rdk, feature_names=names,
                                    class_names=["Inactive", "Active"], mode="classification")
        for i in instances:
            e = expl.explain_instance(rdk[i], predict_fn, num_features=10)
            fig = e.as_pyplot_figure(); fig.tight_layout()
            fig.savefig(f"{out_dir}/lime_instance_{i}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
        return True
    except Exception as e:
        if log: log.info(f"LIME skipped: {type(e).__name__}: {e}")
        return False
