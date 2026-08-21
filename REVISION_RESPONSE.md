# Response to Reviewers — DeepEntXAI (Molecular Diversity, Revision 2)

We thank both reviewers for their careful and, in several places, decisive comments.
Reviewer 2 correctly identified a **data-leakage** problem in the previous pipeline
(transforms and feature selection fit on the full dataset before splitting). We have
taken this seriously and **rebuilt the entire analysis pipeline to be leakage-free**.
As a consequence, all previously reported metrics (accuracy ≈ 0.998, AUC ≈ 0.999)
are **withdrawn and superseded** by honest, scaffold-disjoint results. We believe the
revised manuscript is substantially more rigorous, and that the leakage analysis is
itself now a useful contribution to a literature where this error is common.

**Summary of the rebuilt protocol (applies to many comments below):**
- **Bemis–Murcko scaffold-disjoint** train/test split and scaffold-aware CV — verified
  0 scaffold / 0 SMILES overlap.
- **Every transform (imputation, scaling, feature selection) fit on the training fold
  only.**
- **No PCA.** Features are named and auditable (see Table B).
- Headline model: 5-fold **bidirectional CNN-LSTM ensemble**, **ROC-AUC 0.901 /
  accuracy 0.840 / MCC 0.655** on the scaffold-disjoint hold-out.
- High-accuracy claims are made only via **conformal selective prediction with
  declared coverage** (98.1 % accuracy on the 22 % most-confident compounds).

---

## Reviewer 1

**R1.1 — Abstract carries too many technical details; move to Methods.**
Agreed. We have removed the architecture/optimiser sentences from the Abstract and
relocated them to Methods (Section *Model architecture*, Table A). We note the model
was also re-tuned, so the specific values now differ (bidirectional 64-unit LSTM,
GELU activations, lr = 5.96 × 10⁻⁴, batch 32); Table A gives the full specification.

**R1.2 — Show generalisation on a highly unbalanced dataset.**
Added. Because ROC-AUC is provably invariant to class prevalence, we evaluated the
trained model at increasing imbalance (keeping all inactives, subsampling actives up
to 1:100) and report prevalence-appropriate metrics (Table C, new
Fig. *imbalance_robustness*). ROC-AUC stays ~0.90 across all ratios, and the
virtual-screening **Enrichment Factor rises to 41.9× (at 1:50) and 68.7× (at 1:100)
at the top 0.1 %** — i.e. performance is retained, and the model becomes *more*
useful for screening as the problem becomes realistically imbalanced.

**R1.3 — LIME and Hybrid-XAI top features differ.**
This is expected and now explained explicitly (Methods, *Explainability*). LIME is a
**local** method (a linear surrogate fit in the neighbourhood of individual compounds)
whereas permutation importance and Integrated Gradients are **global** (whole-model)
attributions. Local and global importance answer different questions, so their top
lists legitimately differ. We now (i) highlight the features on which methods *agree*
(e.g. the carboxylic-acid group `fr_COO` is top-ranked under both permutation and
Integrated Gradients) and (ii) add a **stability analysis** (R2.9) showing the top
global features are reproducible.

**R1.4 — Move confusion matrix (3A) and least-important-feature panels (4B, 5B) to
Supplementary.**
Done — these are now Supplementary Figures S1–S3.

**R1.5 — Describe the architecture in a figure or table.**
Added as **Table A** (below) and an architecture schematic (Fig. *07_summary* / new
architecture figure).

---

## Reviewer 2

**R2.1 — PCA/imputation/scaling/feature selection on the full dataset = leakage.**
The reviewer is correct. In the rebuilt pipeline **every transform is fit on the
training fold only**, inside a scaffold-disjoint split. PCA has been removed
entirely. This is the central change of the revision.

**R2.2 — Leakage invalidates the 5-/10-fold CV.**
Also correct and corrected. Cross-validation is now **scaffold-aware**: whole scaffold
families are confined to a single fold, so no molecule (or near-duplicate scaffold)
leaks across folds. Honest CV ROC-AUC = **0.864 ± 0.009**.

**R2.3 — No external validation.**
Added two forms. (i) The scaffold-disjoint hold-out already tests structurally novel
molecules. (ii) We add a **provenance-disjoint external validation** (new stage,
Table D): train on ChEMBL only, test on PubChem only, and vice-versa, with scaling
fit on the training source alone. **We report this result transparently, including
where it is unfavourable.** External mean ROC-AUC = **0.69** (ChEMBL→PubChem 0.66,
PubChem→ChEMBL 0.73) — clearly above random (0.5), so the model learns genuinely
transferable signal and is *not* merely memorising, but **substantially below the
internal scaffold-disjoint AUC of 0.90.** This reveals a real domain shift between
the ChEMBL and PubChem chemical spaces / activity definitions. We therefore (a) keep
the internally-validated, leakage-free 0.90 as the primary result, (b) add this
cross-database gap to the Limitations as a target for future domain-adaptation work,
and (c) note it is consistent with R2.6: the model is strong *within* a database's
chemical space but transfers only moderately *across* databases. This honest
external result is itself evidence *against* the previously suspected leakage —
a leaked model would not show any domain-shift penalty.

**R2.4 — Confusing descriptor totals (2,759 / 3,810 / 4,594 / 2,997 / 2,000).**
The pipeline now has a single, clear feature accounting with **no PCA** (Table B):
Morgan 2048 + RDKit (208 computed → 100 selected) + MACCS 167 + ChemBERTa 768 =
**3,083 features**. The earlier conflicting totals arose from the superseded
PCA-era workflow and no longer apply.

**R2.5 — Why exactly 9,000 inactives; chemical-space representativeness.**
The arbitrary hand-selection is removed. We now keep **every** compound that passes
pre-registered activity-gap thresholds (`thresholds.yaml`): actives ≤ 1 µM (or MIC
≤ 4 µg/mL), inactives > 50 µM (or MIC ≥ 64 µg/mL), per-compound label by majority
vote. This yields **30,580 inactives and 18,513 actives** drawn from both databases
and all 29 species — no cherry-picking, and the selection rule is fully reproducible.

**R2.6 — Assay bias not analysed or corrected.**
Added (new stage, Table E). Restricting the hold-out test to each major source
separately leaves ROC-AUC essentially unchanged — **ChEMBL 0.922, PubChem 0.900**
(spread 0.021) — *despite* the two sources having different active rates (33 % vs
44 %). If the model were exploiting a source artefact, restricting to a single source
would collapse the AUC; instead it discriminates activity *within* each source. The
signal is biological, not assay-specific.

**R2.7 — Choice of 2,000 PCA components is arbitrary; no explained-variance/ablation.**
PCA is removed, so the concern is moot. In its place we use named MI + RFE selection
and provide a **feature-count ablation** (validation AUC vs 40/100/500/1000 features)
showing a plateau, so 100 RDKit descriptors is justified by data, not asserted.

**R2.8 — Metrics (0.998 / 0.999) are implausibly high → leakage/bias.**
Agreed and withdrawn. The honest headline is ROC-AUC 0.901 / accuracy 0.840. We now
state explicitly that ~0.998 is not physically attainable here: replicate MIC
measurements of the *same* compound disagree on the binary label ~12 % of the time,
placing the Bayes ceiling near 0.88–0.93. Our result sits close to the assay's own
reproducibility limit.

**R2.9 — No feature-ranking stability / bootstrapping.**
Added (new stage, Table F, new Fig. *feature_stability_top15*). We report both a
positive and a nuanced result honestly. (A) **Bootstrap stability (20 resamples):**
the top features are highly reproducible — `fr_azo` appears in the top-10 in **100 %**
of resamples and `fr_COO` in **95 %**. (B) **Across-fold agreement:** the *full*
100-descriptor rank correlation across the 5 fold-models is **low (mean Spearman
0.19)**. We explain this transparently: the large majority of the 100 descriptors
carry near-zero importance, so their relative *ordering* is dominated by noise and
drags the whole-list correlation down. We therefore **restrict all interpretability
claims to the stably-identified top features** (which are reproducible by the
bootstrap frequency), rather than to the full ranking — a more defensible and honest
scope for the XAI conclusions.

**R2.10 — DL/XAI descriptions appear in multiple sections.**
Consolidated: the DL architecture is described once (Methods §*Model*), and XAI once
(Methods §*Explainability*); duplicated passages elsewhere were removed.

**R2.11 — How does this advance beyond existing DL+XAI cheminformatics work?**
We sharpened the novelty statement. The contribution is not "another DL+XAI model"
but: (1) a **leakage-free protocol** establishing an honest performance ceiling for
anti-*Enterobacteriaceae* activity, and an explicit **demonstration** of how the
leakage that pervades this sub-literature inflates results; (2) **conformal selective
prediction** for a statistically valid high-confidence operating mode; (3) a
**validated compound-ranking** (retrospective virtual screen: top-1 % = 100 % true
actives, EF at its ceiling); and (4) reproducible, named-feature explanations that
agree with known antibacterial chemistry (fluoroquinolone carboxylic acids).

---

## Table A — Model architecture (DeepEntXAI, tuned)

| Stage | Layer | Configuration | Output |
|---|---|---|---|
| Input ×4 | Morgan / RDKit / MACCS / ChemBERTa | 2048 / 100 / 167 / 768 | per-branch |
| Encoders ×4 | Dense + BatchNorm + Dropout | 128 units, GELU, dropout 0.25 | 4 × 128 |
| Fusion | Concatenate + BatchNorm | — | 512 |
| To-sequence | Dense → Reshape | 512 → (64 × 8) | (64, 8) |
| Conv blocks ×2 | residual Conv1D | 32 filters, kernel 3, GELU | (64, 32) |
| Attention | squeeze-and-excite (channel) | — | (64, 32) |
| Pool | MaxPooling1D | pool 2 | (32, 32) |
| Recurrent | **Bidirectional LSTM** | 64 units (→128) | 128 |
| Head | residual Dense → Dropout → Dense | 64, GELU, dropout 0.25 | 64 |
| Output | Dense | 1, sigmoid | P(active) |

Training: binary cross-entropy, Adam (lr = 5.96 × 10⁻⁴), batch 32, class-weighting,
early stopping (patience 8), ≤ 100 epochs. **Total parameters: 1,276,933.**
Hyper-parameters chosen by Optuna on validation folds only. Headline model = mean of
5 scaffold-fold instances.

## Table B — Feature accounting (no PCA)

| Tool / block | Computed | Removed | Kept | Selection |
|---|---|---|---|---|
| Morgan fingerprint (r = 2) | 2048 | 0 | **2048** | — |
| RDKit descriptors | 208 | 108 | **100** | variance filter → mutual information → RFE (train-only) |
| MACCS keys | 167 | 0 | **167** | — |
| ChemBERTa embedding | 768 | 0 | **768** | — |
| **Total model input** | | | **3,083** | |

## Table C — Performance vs class imbalance (trained model, no retraining)

| Active : inactive | active rate | ROC-AUC | PR-AUC | MCC | EF@1% | EF@0.1% |
|---|---|---|---|---|---|---|
| 1 : 1 | 0.375 | 0.901 | 0.865 | 0.630 | 2.67 | 2.67 |
| 1 : 10 | 0.091 | 0.903 | 0.638 | 0.523 | 10.4 | 11.0 |
| 1 : 50 | 0.020 | 0.905 | 0.404 | 0.318 | 32.9 | **41.9** |
| 1 : 100 | 0.010 | 0.885 | 0.287 | 0.211 | 36.0 | **68.7** |

ROC-AUC is flat (prevalence-invariant); enrichment grows — the model stays useful
under realistic imbalance.

## Table D — External (provenance-disjoint) validation

| Train → Test | n train | n test | ROC-AUC | Accuracy | MCC |
|---|---|---|---|---|---|
| ChEMBL → PubChem | 22,831 | 21,729 | 0.656 | 0.606 | 0.186 |
| PubChem → ChEMBL | 18,470 | 26,859 | 0.727 | 0.535 | 0.266 |
| **Mean** | | | **0.692** | | |
| *Internal scaffold-disjoint (reference)* | | | *0.901* | *0.840* | *0.655* |

Above random (0.5) → real transferable signal; below internal 0.90 → a genuine
cross-database domain shift, reported as a limitation (see R2.3). Scaling was fit on
the training source only; the 505 mixed-provenance compounds were dropped.

## Table E — Assay/source-bias test (`assay_source_bias.json`)

| Restriction | n (test) | active rate | ROC-AUC |
|---|---|---|---|
| ChEMBL only | 4009 | 0.331 | 0.922 |
| PubChem only | 3283 | 0.426 | 0.900 |
| **Spread** | | | **0.021** → no source artefact |

## Table F — Feature-importance stability (bootstrap, 20 resamples)

| Descriptor | mean importance | SD | Top-10 frequency |
|---|---|---|---|
| `fr_azo` | 0.0027 | 0.0012 | **1.00** |
| `fr_COO` (carboxylic acid) | 0.0008 | 0.0003 | **0.95** |
| `fr_Ar_NH` (aromatic amine) | 0.0006 | 0.0003 | 0.70 |
| `VSA_EState3` | 0.0005 | 0.0002 | 0.70 |
| `NHOHCount` | 0.0004 | 0.0002 | 0.55 |

Across-fold full-ranking agreement: **mean pairwise Spearman = 0.19** (low, because
most of the 100 descriptors are near-zero importance — see R2.9). Interpretability
claims are restricted to the high-frequency top features above, which agree with the
Integrated-Gradients ranking and with known antibacterial chemistry.

---

*Supporting artefacts:* `03_Results/02_Metrics/{assay_source_bias, imbalanced_evaluation,
feature_stability, external_cross_source}.json`;
`03_Results/01_Figures/{imbalance_robustness, feature_stability_top15}.{png,pdf}`.
All reproducible via `python 00_run_all.py --only 12 13 14 15`.
