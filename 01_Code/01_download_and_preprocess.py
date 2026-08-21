#!/usr/bin/env python3
"""
=================================================================================
DEEPENTXAI | Script 01 | Data Download & Preprocessing
=================================================================================
Automated, reproducible acquisition of experimentally validated bioactivity data
from ChEMBL and PubChem BioAssay, followed by professional cheminformatics
standardisation, configurable binary labelling, conflict resolution and
de-duplication. Produces a clean modelling dataset + a preprocessing report.

Inputs  : 01_Code/config.yaml  (+ cached 02_Data/01_Raw/*.csv, else the public APIs)
Outputs : 02_Data/02_Processed/labelled.csv
          03_Results/02_Metrics/nb1_preprocessing_report.json
          03_Results/01_Figures/01_dataset_overview.png
          03_Results/06_Logs/NB1.log

Run     : ~/miniconda3/envs/ENT/bin/python 01_data_download_and_preprocessing.py
=================================================================================
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # find the deepentxai package
from deepentxai.config import Config
from deepentxai.utils import set_seed, get_logger, detect_gpu, save_json
from deepentxai.download import DataDownloader
from deepentxai.standardize import MoleculeStandardizer
from deepentxai.labeling import ActivityLabeler


def main() -> None:
    # --- 1. setup -------------------------------------------------------------
    cfg = Config.load()
    set_seed(cfg.seed)
    log = get_logger("NB1", cfg.dir_logs)
    report: dict = {}
    log.info(f"DEEPENTXAI-01 | seed={cfg.seed} | GPU={detect_gpu()} | "
             f"threshold={cfg['labeling']['affinity_threshold_uM']} uM")
    log.info(f"Target: {cfg['target']['description']}")

    # --- 2. download (cached if already fetched) ------------------------------
    data = DataDownloader(cfg).download()
    chembl = data.get("ChEMBL", pd.DataFrame())
    pubchem = data.get("PubChem", pd.DataFrame())
    report["download"] = {"chembl_rows": int(len(chembl)), "pubchem_rows": int(len(pubchem))}
    log.info(f"downloaded  ChEMBL={chembl.shape}  PubChem={pubchem.shape}")

    # --- 3. standardise structures -------------------------------------------
    std = MoleculeStandardizer(cfg["standardize"])
    all_smiles = pd.unique(pd.concat([
        chembl.get("smiles", pd.Series(dtype=str)),
        pubchem.get("smiles", pd.Series(dtype=str))], ignore_index=True))
    canon_map, mw_map = {}, {}
    for s in all_smiles:
        canon_map[s], mw_map[s] = std(s)
    report["standardization"] = std.report()
    log.info(f"standardisation: {report['standardization']}")

    # --- 4. label -------------------------------------------------------------
    labeler = ActivityLabeler(cfg["labeling"])
    rows = []
    for r in chembl.itertuples(index=False):
        canon = canon_map.get(r.smiles)
        if canon is None:
            continue
        lab = labeler.label_one(r.type, r.value, r.units, mw_map.get(r.smiles))
        if lab is not None:
            rows.append({"smiles": canon, "lab": lab, "source": "ChEMBL"})
    for r in pubchem.itertuples(index=False):
        canon = canon_map.get(r.smiles)
        if canon is not None:
            rows.append({"smiles": canon, "lab": int(r.label), "source": "PubChem"})
    measurements = pd.DataFrame(rows)
    report["labelling"] = {"labelled_measurements": int(len(measurements)),
                           "active_frac": round(float(measurements["lab"].mean()), 4)}
    log.info(f"labelled measurements: {report['labelling']}")

    # --- 5. resolve conflicts + de-duplicate ---------------------------------
    final = labeler.resolve(measurements, "smiles", "lab")
    srcmix = measurements.groupby("smiles")["source"].agg(lambda s: "+".join(sorted(set(s))))
    final["sources"] = final["smiles"].map(srcmix)
    report["final"] = {"unique_compounds": int(len(final)),
                       "active": int(final["label"].sum()),
                       "inactive": int((final["label"] == 0).sum()),
                       "active_frac": round(float(final["label"].mean()), 4)}
    log.info(f"final dataset: {report['final']}")

    # --- 6. save -------------------------------------------------------------
    out_csv = os.path.join(cfg.dir_processed, "labelled.csv")
    final.to_csv(out_csv, index=False)
    save_json(report, os.path.join(cfg.dir_reports, "nb1_preprocessing_report.json"))
    log.info(f"saved {len(final)} unique compounds -> {out_csv}")

    # --- 7. QC figure --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    final["label"].map({1: "Active", 0: "Inactive"}).value_counts().plot.bar(
        ax=ax[0], color=["#2a9d8f", "#e76f51"], edgecolor="black")
    ax[0].set_ylabel("compounds"); ax[0].set_xlabel("")
    final["sources"].value_counts().plot.bar(ax=ax[1], color="#4c72b0", edgecolor="black")
    ax[1].set_xlabel("")
    fig.tight_layout()
    figp = os.path.join(cfg.dir_figures, "F01_dataset_overview.png")
    fig.savefig(figp, dpi=150, bbox_inches="tight"); plt.close(fig)

    print("\n================ DEEPENTXAI-01 complete ================")
    print(f"  unique compounds : {report['final']['unique_compounds']}")
    print(f"  active / inactive: {report['final']['active']} / {report['final']['inactive']}")
    print(f"  active fraction  : {report['final']['active_frac']}")
    print(f"  dataset          : {out_csv}")
    print(f"  QC figure        : {figp}")


if __name__ == "__main__":
    main()
