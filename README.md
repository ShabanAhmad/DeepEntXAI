# DeepEntXAI — Explainable Multimodal Deep Learning for Anti-*Enterobacteriaceae* Bioactivity

**DeepEntXAI** predicts the biological activity of chemical compounds against
*Enterobacteriaceae* pathogens and explains *why*. It fuses four molecular
representations in a CNN-LSTM architecture and layers four explainable-AI methods
on top, supporting labelled evaluation, unlabelled scoring, and a calibrated
high-confidence operating mode.

This repository is the **rebuilt, leakage-free, fully reproducible** successor to
the original DeepEntXAI. Every result here is measured under a **Bemis–Murcko
scaffold-disjoint** protocol (structurally distinct molecules in train vs test),
with all feature selection and scaling fit on the training pool only. One
orchestrator runs the entire job — from raw public data to the final figures.

> **Why the numbers differ from the first release.** The original README reported
> ~99.8 % accuracy. That figure came from a *row-wise* split in which the same
> molecules (and near-duplicate scaffolds) appeared in both train and test, plus
> PCA fit on the full matrix — i.e. **data leakage**, which inflates accuracy far
> above what the chemistry supports. This version removes that leakage and reports
> the **honest ceiling** (0.901 ROC-AUC / 0.840 accuracy at full coverage), and
> reaches high accuracy the legitimate way — **conformal selective prediction with
> declared coverage** (98.1 % on the 22 % most-confident compounds). See
> *Reproducibility & Integrity* below.

---

## What's New in This Release

Everything below is new relative to the original DeepEntXAI (which used a single
descriptor sequence, PCA on the full matrix, and a row-wise split):

- **Leakage-free protocol** — Bemis–Murcko **scaffold-disjoint** split and CV
  (0 scaffold / 0 SMILES overlap verified), replacing the row-wise split that let
  the same molecules sit in train and test.
- **Multimodal fusion** — four representations (Morgan, RDKit, MACCS, ChemBERTa),
  each with its own encoder branch, instead of one descriptor block.
- **Bidirectional CNN-LSTM + squeeze-and-excite attention** — a stronger, tuned
  architecture.
- **Genus-wide data** — 29 *Enterobacteriaceae* species from ChEMBL **and** PubChem
  BioAssay (49,093 high-confidence compounds).
- **Activity-gap labelling** — potent-active vs clearly-inactive with the ambiguous
  middle dropped; documented cut-offs in `thresholds.yaml`.
- **Optuna hyper-parameter optimisation** on validation folds only.
- **5-fold ensemble** with seed-decorrelated members (the headline model).
- **Isotonic calibration** + **out-of-fold decision-threshold tuning**.
- **Conformal selective prediction** — the honest high-accuracy story: confidence
  cutoff (rule A) **and** distribution-free class-conditional **Mondrian** conformal
  (rule B), with an accuracy-vs-coverage curve.
- **Compound scoring & ranking, validated** — a DEEPENTXAI Score (0–100) turns the
  calibrated ensemble probability into a ranked hit list, *validated* on the
  scaffold-disjoint hold-out with virtual-screening metrics (Enrichment Factor,
  hit-rate, precision@K) — the top 1 % of ranked compounds are 100 % true actives.
- **Classical baselines + stacking** — XGBoost, RandomForest, and a CNN+XGB+RF
  logistic stack, all OOF-honest (extra stage 09).
- **Regression framing** — *E. coli* whole-cell MIC → pMIC, with bootstrap 95 % CIs
  (extra stage 10).
- **Leakage demonstration** — the same model on a random split, reported as an
  artifact, to quantify exactly how much leakage inflates accuracy (extra stage 11).
- **Replicate-noise ceiling analysis** — measures assay reproducibility (~12 %
  label disagreement) to bound what is honestly achievable.
- **Four XAI methods** — permutation (modality + descriptor), Integrated Gradients,
  SHAP, LIME.
- **One numbered orchestrator** — `00_run_all.py` runs the whole pipeline; every
  folder, stage, and figure is numeric-prefixed.

---

## Key Features

- **Multimodal representation** — Morgan-2048 fingerprints + RDKit descriptors +
  MACCS-167 keys + ChemBERTa-768 embeddings, each with its own encoder branch.
- **Hybrid deep model** — per-modality Dense encoders → residual Conv1D blocks with
  squeeze-and-excite channel attention → **bidirectional LSTM** → residual Dense →
  sigmoid. CNN captures local descriptor structure, LSTM the long-range dependencies.
- **Leakage-free evaluation** — scaffold-disjoint hold-out **and** scaffold-aware
  5-fold CV; 0 scaffold / 0 SMILES overlap verified.
- **High-confidence activity-gap labelling** — potent actives vs clearly-weak
  inactives; the ambiguous middle band is dropped, giving cleaner ground truth.
- **Honest high accuracy** — **conformal selective prediction**: the model abstains
  on its least-confident compounds and reports accuracy *with* coverage (up to
  98.1 %), plus a distribution-free Mondrian conformal guarantee.
- **Four XAI methods** — permutation importance (modality + descriptor), Integrated
  Gradients, SHAP, and LIME.
- **One-command orchestrator** — `00_run_all.py` runs all eight numbered stages.
- **Fully numbered layout** — every folder and stage is prefixed `01_`, `02_`, …
  so the pipeline order is unambiguous.
- **CPU-only** — no GPU required (tuned for 8 threads).

---

## Repository Structure

```
DeepEntXAI_Final/
├── 00_run_all.py                          # top-level launcher (forwards to 01_Code)
├── README.md
│
├── 01_Code/
│   ├── 00_run_all.py                      # ORCHESTRATOR — runs stages 01→08
│   ├── 01_download_and_preprocess.py      # ChEMBL + PubChem → standardise → gap-label
│   ├── 02_feature_engineering.py          # 4 modalities + scaffold split + descriptor selection
│   ├── 03_train_and_validate.py           # Optuna HPO → 5-fold scaffold CV → hold-out
│   ├── 04_ensemble_calibrate_threshold.py # 5-fold CNN-LSTM ensemble (headline) + calibration
│   ├── 05_conformal_selective.py          # accuracy-vs-coverage (the honest 98 %)
│   ├── 06_explainability.py               # permutation / IntegratedGradients / SHAP / LIME
│   ├── 07_compound_scoring_and_ranking.py # DEEPENTXAI score + hit list, validated by enrichment
│   ├── 08_reports_and_figures.py          # publication metrics table + summary figure
│   ├── 09_baselines_and_stacking.py       # (extra) XGBoost/RF baselines + CNN+XGB+RF stacking
│   ├── 10_regression_ecoli_mic.py         # (extra) E. coli MIC pMIC regression — secondary framing
│   ├── 11_leakage_demonstration.py        # (extra) random-split leakage artifact — integrity control
│   ├── config.yaml                        # single source of truth
│   ├── thresholds.yaml                    # documented activity-gap cut-offs
│   ├── requirements.txt                   # pinned environment (CPU)
│   └── deepentxai/                        # engine package (15 modules)
│       ├── config.py    model.py    train.py     features.py   selection.py
│       ├── splits.py    standardize.py  labeling.py  download.py  evaluate.py
│       └── explain.py   scoring.py   predict.py   utils.py     __init__.py
│
├── 02_Data/
│   ├── 01_Raw/                 # chembl_raw.csv, pubchem_raw.csv
│   ├── 02_Processed/           # labelled.csv  (49,093 gap compounds)
│   ├── 03_Features_Raw/        # features.npz  (all 4 modalities + scaffolds)
│   ├── 04_Features_Selected/   # RDKit selector + scaler + selected feature names
│   ├── 05_Splits/              # split.npz  (scaffold-disjoint train/test/5-fold)
│   └── 06_Regression/          # labelled_reg.csv  (E. coli MIC pMIC — secondary framing)
│
└── 03_Results/
    ├── 01_Figures/             # ROC, PR, calibration, confusion, risk-coverage, enrichment, dataset
    ├── 02_Metrics/             # all JSON metrics + publication_metrics_table.csv + screening_metrics.json
    ├── 03_Model/               # 5 ensemble fold weights + best model + calibrator + hparams
    ├── 04_Explainability/      # permutation / IntegratedGradients / SHAP / LIME
    ├── 05_Predictions/         # OOF + test probabilities
    ├── 06_Logs/                # per-stage logs
    ├── 07_Optuna/              # HPO study
    └── 08_Rankings/            # compound_ranking.csv (all) + top_hits.csv (top 100)

    01_Figures/ holds the numbered manuscript set: Figure 1-12 (F01..F12 plots) +
    Table A-F (T01..T06, rendered) + FIGURES.txt (the numbered caption index).
```

---

## Dataset

- **Sources**: ChEMBL and PubChem BioAssay, across **29 *Enterobacteriaceae* species**
  (*E. coli, Klebsiella, Enterobacter, Salmonella, Shigella, Serratia, Proteus,
  Citrobacter, Yersinia*, …).
- **Endpoints**: IC50 / EC50 / Ki / Kd (affinity) and MIC (phenotypic).
- **Standardisation**: largest fragment, neutralised, per-structure aggregation.
- **Activity-gap labelling** (high-confidence subset):
  - **Active** — potency ≤ 1 µM *or* MIC ≤ 4 µg/mL
  - **Inactive** — potency > 50 µM *or* MIC ≥ 64 µg/mL
  - the ambiguous middle band is **dropped**; per-compound label by majority vote.
- **Final composition**: **49,093 compounds** — 18,513 active / 30,580 inactive.
- **Split**: scaffold-disjoint — 41,730 train (5 scaffold folds) / 7,363 hold-out test.

---

## Model Architecture

```
        Morgan(2048)   RDKit(100*)   MACCS(167)   ChemBERTa(768)
             │             │            │              │
          Dense enc.    Dense enc.   Dense enc.    Dense enc.        (* RDKit 208→100 via MI+RFE, train-only)
             └─────────────┴────────────┴──────────────┘
                            Concatenate → BatchNorm → Reshape
                                        │
                       residual Conv1D blocks + Squeeze-Excite attention
                                        │
                             Bidirectional LSTM (128)
                                        │
                        residual Dense → Dropout → Sigmoid
```

Hyper-parameters are chosen by **Optuna** on validation folds only; the headline
model is a **5-fold ensemble** (mean of five scaffold-fold CNN-LSTMs), with isotonic
calibration and an OOF-tuned decision threshold.

---

## Getting Started

**1. Clone**
```bash
git clone https://github.com/ShabanAhmad/DeepEntXAI.git
cd DeepEntXAI
```

**2. Environment** (Python 3.10, CPU — no GPU needed)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r 01_Code/requirements.txt
```

**3. Run the whole pipeline**
```bash
P=python                     # or ~/miniconda3/envs/ENT/bin/python
$P 00_run_all.py --list      # show the 7 stages
$P 00_run_all.py             # full run: raw data → figures (CPU, 8 threads)
```

**4. Run / resume individual stages**
```bash
$P 00_run_all.py --from 04   # resume from the ensemble onward
$P 00_run_all.py --only 05   # just remake the conformal table + figure
$P 00_run_all.py --to 02     # only download + features
$P 00_run_all.py --threads 8 # CPU threads per stage
```

The package ships with all final artifacts already computed; stage 01 reuses the
cached raw pulls in `02_Data/01_Raw/`, so re-runs skip the network. Delete that
folder to force a fresh download.

---

## Pipeline Stages

| # | Stage | Produces |
|---|---|---|
| 01 | Download & preprocess | `02_Processed/labelled.csv`, dataset QC figure |
| 02 | Feature engineering | `03_Features_Raw/features.npz`, `05_Splits/split.npz`, RDKit selection |
| 03 | Train & validate | HPO `best_hparams.json`, `cv_metrics.json`, hold-out `test_metrics.json` |
| 04 | Ensemble + calibrate | 5 fold weights, `ensemble_test_metrics.json`, OOF + test predictions |
| 05 | Conformal selective | `conformal_selective.json`, `risk_coverage.{png,pdf}` |
| 06 | Explainability | permutation / IntegratedGradients / SHAP / LIME artifacts |
| 07 | Compound scoring & ranking | `08_Rankings/compound_ranking.csv`, `top_hits.csv`, `screening_metrics.json`, `enrichment_curve.{png,pdf}` |
| 08 | Reports & figures | `publication_metrics_table.csv`, summary figure |

### Optional secondary analyses (run with `--only`)

| # | Stage | Produces |
|---|---|---|
| 09 | Baselines & stacking | XGBoost/RF metrics + CNN+XGB+RF stack (`stacked_test_metrics.json`) |
| 10 | *E. coli* MIC regression | `ecolimic_regression.json`, `regression_scatter.{png,pdf}` |
| 11 | Leakage demonstration | random-split metrics, labelled as an artifact (integrity control) |

```bash
$P 00_run_all.py --only 09 10 11   # run the secondary analyses
```

---

## Results Summary

All metrics are on the **scaffold-disjoint hold-out** (7,363 compounds), leakage-free.

### Headline model — 5-fold CNN-LSTM ensemble (full coverage)

| Metric | Score |
|---|---|
| ROC-AUC | **0.901** |
| PR-AUC | 0.870 |
| Accuracy | **0.840** |
| Balanced accuracy | 0.823 |
| Precision | 0.806 |
| Recall / Sensitivity | 0.755 |
| Specificity | 0.891 |
| F1-score | 0.780 |
| MCC | **0.655** |
| Scaffold 5-fold CV ROC-AUC | 0.864 ± 0.009 |

### Baselines (identical features & split)

| Model | ROC-AUC | Accuracy | MCC |
|---|---|---|---|
| **CNN-LSTM ensemble** | **0.901** | **0.840** | **0.655** |
| XGBoost | 0.886 | 0.814 | 0.595 |
| RandomForest | 0.872 | 0.798 | 0.560 |

The deep ensemble beats every classical baseline; stacking did not improve on it.

### High accuracy the honest way — conformal selective prediction

The model abstains on its least-confident compounds; the cutoff is calibrated on
out-of-fold training predictions only. **Full-coverage accuracy is always reported
alongside.**

| Coverage (compounds called) | Accuracy |
|---|---|
| 100 % (full — headline) | 0.840 |
| 77 % | 0.901 |
| 55 % | 0.936 |
| 44 % | 0.954 |
| **22 %** | **0.981** |
| 12 % | 0.987 |

Distribution-free **Mondrian conformal**: at α = 0.30 the model calls 66 % of
compounds at **91.9 %** accuracy with a finite-sample validity guarantee.

*Secondary result* — *E. coli* MIC **regression** (single species / single endpoint):
R² 0.595, Pearson 0.774, RMSE 0.75 (near the assay's own noise floor).

---

## Compound Scoring & Ranking (retrospective virtual screen)

For drug-discovery triage, every hold-out compound gets a **DEEPENTXAI Score
(0–100)** — the calibrated ensemble probability of activity, blended with the
model's confidence. Compounds are ranked by this score (`08_Rankings/`).

A ranking is only meaningful if the true actives really concentrate at the top, so
we **validate it** on the scaffold-disjoint hold-out (labels known) with the
standard virtual-screening metrics:

| Top fraction | Compounds | Hit-rate | Enrichment Factor |
|---|---|---|---|
| Top 1 % | 74 | **100 %** | **2.67** (= ceiling) |
| Top 5 % | 368 | 99.2 % | 2.65 |
| Top 10 % | 736 | 99.1 % | 2.64 |
| Top 20 % | 1,473 | 92.7 % | 2.47 |

`precision@10 = @25 = @50 = 1.000`, `@100 = 0.99`. Ranking ROC-AUC 0.901.

**How to read the Enrichment Factor.** EF = 1 means "no better than random"; the
**ceiling here is only 2.67** because the hold-out is relatively balanced (37.5 %
active, so EF ≤ 1 / 0.375). The result that matters is that the ranking **hits that
ceiling** — the top 1 % of ranked compounds are *entirely* true actives. On a
realistic prospective library where actives are rare, the same ranking quality would
translate into a much larger numeric EF.

**Sanity check.** The very top-ranked compounds are **fluoroquinolone carboxylic
acids** (cyclopropyl-quinolone cores — the ciprofloxacin/moxifloxacin family), a
textbook anti-*Enterobacteriaceae* chemotype. This agrees with the XAI, which flagged
the carboxylic-acid group (`fr_COO`) as important — model and explanation point to
the same chemistry.

---

## XAI Insights

Explainability (`06_explainability.py` → `03_Results/04_Explainability/`)
uses four complementary methods:

1. **Permutation importance** — per **modality** and per **RDKit descriptor**.
2. **Integrated Gradients** — signed per-descriptor attribution.
3. **SHAP** (GradientExplainer) — descriptor contribution summary.
4. **LIME** — instance-level explanations for representative compounds.

**Modality-level result** (drop in ROC-AUC when a branch is permuted):

| Modality | Importance |
|---|---|
| Morgan fingerprints | **0.220** |
| MACCS keys | 0.022 |
| ChemBERTa embeddings | 0.013 |
| RDKit descriptors | 0.010 |

Morgan structural fingerprints dominate the decision; the other modalities add
smaller, complementary signal.

**Descriptor-level result** (interpretable RDKit branch) — the top drivers are
chemically coherent functional-group and electronic descriptors:

| Method | Top descriptors |
|---|---|
| Permutation importance | `fr_azo`, **`fr_COO`**, `fr_Ar_NH`, `VSA_EState9`, `fr_SH`, `fr_nitroso` |
| Integrated Gradients | `PEOE_VSA14`, **`fr_COO`**, `EState_VSA2`, `MinEStateIndex`, `SlogP_VSA8` |

`fr_COO` (carboxylic acid) ranks highly under **both** methods — a consistent,
method-agnostic signal. The remaining drivers (azo, aromatic amine, thiol, nitroso
groups; EState/VSA electronic-surface descriptors) are plausible determinants of
antibacterial activity. SHAP summary and per-compound LIME plots are in the
explainability folder.

Morgan/MACCS/ChemBERTa features are not human-readable, so named-descriptor
attribution is performed on the interpretable RDKit branch while modality-level
importance covers all four — the correct division for this architecture.

---

## Reproducibility & Integrity

- **Leakage-free by construction**: Bemis–Murcko scaffold-disjoint split and CV; all
  feature selection / scaling / calibration fit on the training pool only; no
  duplicate structures across the split.
- **The honest ceiling**: full-coverage 0.840 accuracy / 0.901 ROC-AUC sits close to
  the assay's own reproducibility limit — replicate MIC measurements of the same
  compound disagree on the binary label ~12 % of the time, so full-set 98–99 %
  accuracy is not physically attainable without leakage.
- **The 98 % figure is coverage-qualified**: it is a selective-prediction statement
  (98.1 % on the 22 % most-confident compounds), reported the way the
  selective-prediction literature requires — never as a full-set claim.
- **Deterministic**: fixed seeds; the orchestrator regenerates every shipped artifact.

---

## Applications

- Active-compound screening against *Enterobacteriaceae*
- Efficacy assessment and hit prioritisation
- Drug repurposing
- High-confidence triage (call only the compounds above a chosen confidence, abstain
  on the rest)

---

## License

MIT License.

---

## Authors & Contact

**Dr Shaban Ahmad, Nagmi Bano, Prof Khalid Raza**
Computational Intelligence and Bioinformatics Lab, Department of Computer Science,
Jamia Millia Islamia, New Delhi, India.
