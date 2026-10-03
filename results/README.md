# Results, provenance

These files preserve the WESAD benchmark (15 subjects, S2 to S17 excluding S12), with a
separate corrected transfer experiment. WESAD is not redistributed (see
[README dataset download and integrity](../README.md#dataset-download-and-integrity)), so a clean clone cannot regenerate them without first
downloading WESAD. `python scripts/calibration.py --synthetic` provides an offline calibration smoke
check. Use the [reproduction commands](../README.md#reproduce-experiments) for real datasets. For dataset and
shipped-model lineage, see [README data protocol](../README.md#data-and-evaluation-protocol) and
[shipped model](../README.md#shipped-model).

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

## Provenance & leakage safety

Historical JSONs retain their original provenance blocks and environment records. Some recorded
source SHAs do not resolve in the current checkout; these stamps are preserved, not replaced by a
claim that the historical experiment was rerun. New experiments record their actual source state
and generation time (OpenMP is needed for XGBoost/LightGBM on macOS).

## Transfer correction

`cross_dataset.json` uses version 2 portable features: EDA/TEMP slopes per second, channel-specific
sampling rates, and original timestamps retained when non-finite samples are omitted. The 2026-10-03 raw-data rerun verified all 15 WESAD pickle hashes and fingerprinted the 20 Non-EEG recordings.
Transfer balanced accuracy is **0.557 / 0.494** (within-dataset **0.868 / 0.699**).
The new result records the supported package versions (NeuroKit2 0.2.12). WESAD heart-rate features
also differ from historical NeuroKit2 0.2.7; this is a fresh raw-data recomputation, not an isolated
slope-only comparison. Source-file checksums identify the actual implementation used; the base
Git SHA and dirty-worktree flag do not claim that the base commit alone contains the fixes.
Versioned cache sidecars prevent automatic reuse of the old per-sample features. Devices, stressors and labels
still differ; the comparison does not establish generalization to a matched independent corpus.

The original result and figure remain in [historical/cross_dataset_v1.json](historical/cross_dataset_v1.json)
and [the historical plot](../docs/figures/historical/cross_dataset_v1.png). Their 0.573 / 0.500 transfer
balanced accuracies include the old slope-unit mismatch and are not the corrected estimate.

## Probability metrics

Binary Brier is `mean((P(stress) - label)**2)` for both vector and two-column inputs. Multiclass Brier
remains the sum of squared class errors. The preserved `calibration.json` used a two-class sum;
README/dashboard export converts a copy to binary MSE (divides Brier and its paired gap/CI by two).
ECE, reliability curves, p-values and original timestamps are unchanged. Historical personalization
already uses binary MSE. Full-window calibration is pooled; personalization uses subject means on a
reserved half, so their aggregation and evaluation samples remain different.

Synthetic runs write only to `results/demo/` and `outputs/figures/demo/`. Model exports refresh the
verification checksum alongside the artifact.

Committing calibration.json and personalization.json is **leakage-free by construction**: the
recalibration map is fit only on out-of-fold *training*-subject probabilities and never the held-out
subject (`scripts/calibration.py:loso_recalibrated_proba`), and few-shot enrollment windows are disjoint
from the evaluation half (`scripts/personalize.py`). So these are held-out results, not fitted-on-test
outputs, safe to ship as a snapshot. The leakage guards are enforced by `tests/test_methodology.py`.
