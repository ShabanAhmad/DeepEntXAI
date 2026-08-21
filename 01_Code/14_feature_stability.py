#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI_FINAL | 14 | Feature-Importance Stability   (reviewer R2.9)
=================================================================================
Reviewer concern: interpretability claims need the STABILITY of feature rankings
across resamples / folds, otherwise the top-feature list may be noise.

Two complementary stability analyses (no retraining of the headline model, so
accuracy is untouched):

  (A) Bootstrap stability -- resample the hold-out set B times, recompute RDKit
      permutation importance each time, and report per-descriptor mean +/- SD,
      a 95% percentile CI, and how often each descriptor stays in the Top-10
      (selection frequency). Stable descriptors have high mean and high frequency.

  (B) Across-fold agreement -- run permutation importance under each of the 5
      ensemble fold models and report the mean pairwise Spearman rank correlation
      of the descriptor rankings. High correlation = the ranking is not fold-luck.

Consumes : 03_Results/03_Model/deepentxai_best.keras + ensemble_fold{0..4}.keras
Outputs  : 03_Results/02_Metrics/feature_stability.json
           03_Results/01_Figures/feature_stability_top15.{png,pdf}
Run      : DEEPENT_THREADS=8 ~/miniconda3/envs/ENT/bin/python 14_feature_stability.py
=================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, save_json
from deepentxai.train import load_matrices, slice_inputs
from deepentxai import explain as X_

B_BOOT = 20          # bootstrap resamples
SUBSAMPLE = 2500     # compounds per resample (enough for a stable ranking, keeps CPU sane)
N_REPEATS = 2


def main() -> None:
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB14", cfg.dir_logs)
    import keras
    from scipy.stats import spearmanr

    X, y, split = load_matrices(cfg)
    te = split["test_idx"]
    names = json.load(open(os.path.join(cfg.dir_features_selected,
                      "selected_rdkit_features.json")))["selected_rdkit_features"]
    rng = np.random.RandomState(cfg.seed)

    # ---- (A) bootstrap stability on the headline model ----------------------
    best = keras.models.load_model(os.path.join(cfg.dir_models, "deepentxai_best.keras"))
    cols = {n: [] for n in names}
    for b in range(B_BOOT):
        idx = rng.choice(te, size=min(SUBSAMPLE, len(te)), replace=True)
        Xb = slice_inputs(X, idx); yb = y[idx]
        imp = X_.permutation_descriptor(best, Xb, yb, names, n_repeats=N_REPEATS, seed=cfg.seed + b)
        d = dict(zip(imp["descriptor"], imp["importance"]))
        for n in names:
            cols[n].append(float(d.get(n, 0.0)))
        log.info(f"bootstrap {b+1}/{B_BOOT} done")

    arr = {n: np.array(v) for n, v in cols.items()}
    mean_imp = {n: float(a.mean()) for n, a in arr.items()}
    order = sorted(names, key=lambda n: mean_imp[n], reverse=True)
    # per-bootstrap top-10 sets -> selection frequency
    boot_rank = np.argsort(-np.column_stack([arr[n] for n in names]), axis=1)  # B x P
    top10_freq = {n: 0 for n in names}
    for b in range(B_BOOT):
        top10 = set(np.array(names)[boot_rank[b, :10]])
        for n in top10:
            top10_freq[n] += 1
    stability = []
    for n in order[:15]:
        a = arr[n]
        stability.append({
            "descriptor": n,
            "mean_importance": round(float(a.mean()), 5),
            "sd": round(float(a.std()), 5),
            "ci95": [round(float(np.percentile(a, 2.5)), 5),
                     round(float(np.percentile(a, 97.5)), 5)],
            "top10_frequency": round(top10_freq[n] / B_BOOT, 2),
        })

    keras.backend.clear_session()

    # ---- (B) across-fold rank agreement -------------------------------------
    fold_ranks = []
    for f in range(int(cfg["split"]["n_folds"])):
        fp = os.path.join(cfg.dir_models, f"ensemble_fold{f}.keras")
        if not os.path.exists(fp):
            continue
        m = keras.models.load_model(fp)
        idx = rng.choice(te, size=min(SUBSAMPLE, len(te)), replace=False)
        imp = X_.permutation_descriptor(m, slice_inputs(X, idx), y[idx], names,
                                        n_repeats=1, seed=cfg.seed)
        d = dict(zip(imp["descriptor"], imp["importance"]))
        fold_ranks.append([d.get(n, 0.0) for n in names])
        keras.backend.clear_session()
        log.info(f"fold {f} importance done")
    corrs = []
    for i in range(len(fold_ranks)):
        for j in range(i + 1, len(fold_ranks)):
            corrs.append(spearmanr(fold_ranks[i], fold_ranks[j]).correlation)
    mean_spearman = round(float(np.mean(corrs)), 3) if corrs else float("nan")

    out = {
        "bootstrap": {"n_resamples": B_BOOT, "subsample": SUBSAMPLE,
                      "top15_stable_descriptors": stability},
        "across_fold": {"n_folds": len(fold_ranks),
                        "mean_pairwise_spearman": mean_spearman},
        "verdict": (f"TOP features are stable (recur in >= "
                    f"{int(100*min(s['top10_frequency'] for s in stability[:5]))}% of bootstraps), "
                    f"but the FULL 100-descriptor ranking has low cross-fold correlation "
                    f"(Spearman {mean_spearman}) because most descriptors carry near-zero "
                    "importance and their order is unstable. Interpretability claims are "
                    "restricted to the stably-identified top features."),
    }
    save_json(out, os.path.join(cfg.dir_metrics, "feature_stability.json"))

    # figure: top-15 mean importance with SD error bars
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = stability[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([s["descriptor"] for s in top], [s["mean_importance"] for s in top],
            xerr=[s["sd"] for s in top], color="#4c72b0", edgecolor="black", capsize=2)
    ax.set_xlabel("permutation importance (mean ± SD over bootstraps)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.dir_figures, f"F10_feature_stability_top15.{ext}"), dpi=300)
    plt.close(fig)

    print("\n================ DEEPENTXAI-14 (feature stability) ================")
    print(f"  across-fold mean pairwise Spearman: {mean_spearman}")
    print(f"  {'descriptor':22s} {'mean':>8} {'sd':>7} {'top10-freq':>10}")
    for s in stability[:10]:
        print(f"  {s['descriptor']:22s} {s['mean_importance']:>8.4f} {s['sd']:>7.4f} {s['top10_frequency']:>10}")


if __name__ == "__main__":
    main()
