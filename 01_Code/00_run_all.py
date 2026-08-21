#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI_FINAL | 00 | ORCHESTRATOR  — one command runs the entire pipeline
=================================================================================
From raw public data to the final figures, in order, with one process per stage
(each stage gets a clean TensorFlow/Keras state):

    01  download + preprocess ...... ChEMBL + PubChem -> standardise -> label
                                     (activity-gap high-confidence subset)
    02  feature engineering ........ Morgan+RDKit+MACCS+ChemBERTa, scaffold split,
                                     train-only RDKit descriptor selection
    03  train + validate ........... Optuna HPO -> 5-fold scaffold CV -> hold-out
    04  ensemble + calibrate ....... 5-fold CNN-LSTM ensemble (the headline model),
                                     isotonic calibration + OOF threshold, saves
                                     the 5 fold weights + OOF/test predictions
    05  conformal selective ........ honest high-accuracy statement WITH coverage
    06  explainability ............. permutation / Integrated Gradients / SHAP / LIME
    07  scoring + ranking .......... DEEPENTXAI score + hit list, validated by
                                     enrichment factor / precision@K on the hold-out
    08  reports + figures .......... publication metrics table + summary figure

Everything is scaffold-disjoint (leakage-free). Data lands under 02_Data/,
results under 03_Results/ — all numbered.

USAGE
    P=~/miniconda3/envs/ENT/bin/python
    $P 00_run_all.py                      # run every stage 01..08
    $P 00_run_all.py --only 04 05         # run just these stages
    $P 00_run_all.py --from 03            # resume from stage 03 onward
    $P 00_run_all.py --to 02              # only 01..02
    $P 00_run_all.py --threads 8          # CPU threads per stage (default 8)
    $P 00_run_all.py --list               # show the stage table and exit

Notes
  * No GPU required — tuned for CPU (DEEPENT_THREADS default 8).
  * Stage 01 reuses cached raw downloads if 02_Data/01_Raw is already populated
    (config download.reuse_cache: true), so re-runs skip the network.
  * Stages write into fixed numbered dirs; re-running a stage overwrites cleanly.
=================================================================================
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (stage id, script filename, one-line description)
STAGES = [
    ("01", "01_download_and_preprocess.py",       "download ChEMBL+PubChem, standardise, activity-gap labelling"),
    ("02", "02_feature_engineering.py",           "Morgan+RDKit+MACCS+ChemBERTa, scaffold split, descriptor selection"),
    ("03", "03_train_and_validate.py",            "Optuna HPO, 5-fold scaffold CV, hold-out test"),
    ("04", "04_ensemble_calibrate_threshold.py",  "5-fold CNN-LSTM ensemble + calibration + fold weights"),
    ("05", "05_conformal_selective.py",           "conformal selective prediction (accuracy vs coverage)"),
    ("06", "06_explainability.py",                "permutation / IntegratedGradients / SHAP / LIME"),
    ("07", "07_compound_scoring_and_ranking.py",  "DEEPENTXAI score + ranking, validated by enrichment (EF/precision@K)"),
    ("08", "08_reports_and_figures.py",           "publication metrics table + summary figure"),
]
IDS = [s[0] for s in STAGES]

# Optional secondary analyses — NOT part of the default 01..08 run.
# Invoke explicitly, e.g.  00_run_all.py --only 09 10 11
EXTRAS = [
    ("09", "09_baselines_and_stacking.py",  "XGBoost/RandomForest baselines + CNN+XGB+RF stacking"),
    ("10", "10_regression_ecoli_mic.py",    "E. coli MIC pMIC regression (secondary framing)"),
    ("11", "11_leakage_demonstration.py",   "leakage control: random-split artifact (integrity check)"),
    ("12", "12_assay_source_bias.py",       "assay/source-bias analysis (reviewer R2.6)"),
    ("13", "13_imbalanced_evaluation.py",   "highly-imbalanced stress test, PR-AUC/EF (reviewer R1.2)"),
    ("14", "14_feature_stability.py",       "bootstrap + across-fold feature-importance stability (R2.9)"),
    ("15", "15_external_cross_source.py",   "external validation: train ChEMBL / test PubChem (R2.3)"),
]
ALL = STAGES + EXTRAS


def select(args) -> list:
    ids = IDS
    if args.only:                                   # --only may name core OR extra stages
        want = set(args.only)
        return [s for s in ALL if s[0] in want]
    if args.from_:
        ids = IDS[IDS.index(args.from_):]
    if args.to:
        ids = [i for i in ids if i <= args.to]
    return [s for s in STAGES if s[0] in set(ids)]   # default run = core 01..08 only


def main() -> int:
    ap = argparse.ArgumentParser(description="DEEPENTXAI_FINAL orchestrator")
    ap.add_argument("--only", nargs="+", metavar="ID", help="run only these stage ids, e.g. 04 05")
    ap.add_argument("--from", dest="from_", metavar="ID", help="resume from this stage id onward")
    ap.add_argument("--to", metavar="ID", help="stop after this stage id")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("DEEPENT_THREADS", "8")),
                    help="CPU threads per stage (default 8)")
    ap.add_argument("--list", action="store_true", help="print the stage table and exit")
    args = ap.parse_args()

    if args.list:
        print("\nDEEPENTXAI_FINAL core pipeline (default run = 01..08):")
        for sid, script, desc in STAGES:
            print(f"  {sid}  {script:38s} {desc}")
        print("\nOptional secondary analyses (run with --only):")
        for sid, script, desc in EXTRAS:
            print(f"  {sid}  {script:38s} {desc}")
        return 0

    stages = select(args)
    if not stages:
        print("no stages selected"); return 1

    env = dict(os.environ)
    env["DEEPENT_THREADS"] = str(args.threads)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        env[v] = str(args.threads)
    env.setdefault("DEEPENT_CONFIG", os.path.join(HERE, "config.yaml"))

    py = sys.executable
    print("=" * 79)
    print(f"DEEPENTXAI_FINAL orchestrator | python={py} | threads={args.threads}")
    print(f"stages: {', '.join(s[0] for s in stages)}")
    print("=" * 79)

    t0 = time.time()
    for sid, script, desc in stages:
        print(f"\n>>> STAGE {sid}  {desc}\n    {script}", flush=True)
        st = time.time()
        r = subprocess.run([py, os.path.join(HERE, script)], env=env, cwd=HERE)
        dt = time.time() - st
        if r.returncode != 0:
            print(f"\n!!! STAGE {sid} FAILED (exit {r.returncode}) after {dt/60:.1f} min. Stopping.")
            return r.returncode
        print(f"<<< STAGE {sid} done in {dt/60:.1f} min", flush=True)

    print(f"\n{'='*79}\nALL SELECTED STAGES COMPLETE in {(time.time()-t0)/60:.1f} min")
    print("results: 03_Results/  (01_Figures, 02_Metrics, 03_Model, 04_Explainability, ...)")
    print("=" * 79)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
