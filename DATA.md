# Data & model availability

To keep the repository within GitHub's size limits, two categories of large
binary files are **not tracked in git** (see `.gitignore`) and are regenerated
by the pipeline:

| Excluded | Size | How to regenerate |
|---|---|---|
| `02_Data/03_Features_Raw/features.npz` | ~155 MB | `python 00_run_all.py --only 02` (from `02_Data/02_Processed/labelled.csv`, which **is** tracked) |
| `03_Results/03_Model/*.keras` (5 fold weights + best) | ~105 MB | `python 00_run_all.py --from 03 --to 04` |

Everything else needed to understand and reproduce the work **is** tracked:
- all source code (`01_Code/`) and the orchestrator,
- the processed, labelled dataset (`02_Data/02_Processed/labelled.csv`, 49,093 compounds),
- the raw pulls, splits, selected-feature indices, scalers/selectors,
- every figure and table (`03_Results/01_Figures/`), all metrics JSON, the
  compound rankings, and the trained-model calibrator + hyper-parameters.

The full trained weights are available from the authors on request, or reproduce
them bit-for-bit with the fixed-seed pipeline above.
