# Results, provenance

These files contain the recorded WESAD benchmark and targeted reruns of transfer, personalization
and threshold evaluation. The original benchmark's lineage is in `provenance.json`; stamped result
JSONs record their own source revision. The CNN and tuning experiments were not rerun with these
corrections. WESAD is not redistributed (see [../README.md](../README.md)), so a clean
clone needs the download to regenerate results; `make demo` uses synthetic data. For dataset and
shipped-model lineage, see [../README.md](../README.md).

| File | Produced by |
| ---- | ----------- |
| metrics.json, {binary,multiclass}_model_comparison.csv, {binary,multiclass}_per_subject.csv, shap_top_features.csv | scripts/run_experiment.py |
| ablation.csv | scripts/ablation.py |
| wrist.json | scripts/wrist.py |
| cross_dataset.json | scripts/cross_dataset.py |
| stats.json | scripts/stats.py |
| calibration.json | scripts/calibration.py |
| personalization.json | scripts/personalize.py |
| threshold_metrics.json | scripts/threshold_metrics.py |
| provenance.json | scripts/stamp_provenance.py |

**Not committed** (regenerate on WESAD): tuning.json and its figures, which require the WESAD download.

The transfer experiment was rerun with per-second EDA/TEMP slopes and version-2 caches; corpus and
label differences remain. Personalization was rerun with the corrected enrollment sampler: budgets
of five and ten use exactly that many windows per subject; a budget of 20 uses the 14 to 15 available.
The result metadata records actual enrollment minima and maxima for each budget.
SHAP summarizes XGBoost fit and explained on all binary windows; it is exploratory, not held-out
importance. The Youden-J threshold is selected and scored on the same pooled LOSO predictions.
Threshold metrics were rerun with the corrected XGBoost class weighting; the Random Forest
headline metrics are unchanged.

## Provenance & leakage safety

`provenance.json` preserves the original environment and records package versions for the three
targeted reruns under `analysis_reruns`; each result file carries its own code revision.

The stamped JSONs (calibration.json, cross_dataset.json, personalization.json, threshold_metrics.json) carry a
provenance block (git_sha, generated_at) recording exactly which commit produced them; the
remaining xgboost/lightgbm-dependent artifacts gain the stamp on the next `make reproduce` (these
need OpenMP, `brew install libomp` on macOS).

Committing calibration.json and personalization.json is **leakage-free by construction**: the
recalibration map is fit only on out-of-fold *training*-subject probabilities and never the held-out
subject (`scripts/calibration.py:loso_recalibrated_proba`), and few-shot enrollment windows are disjoint
from the evaluation half (`scripts/personalize.py`). So these are held-out results, not fitted-on-test
outputs, safe to ship as a snapshot. The leakage guards are enforced by `tests/test_methodology.py`.
