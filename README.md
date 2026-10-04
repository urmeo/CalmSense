# CalmSense

### A Machine Learning and Deep Learning Framework for Wearable Biosignal Analysis, Integrating 1D CNNs, Explainable AI, Probability Calibration, and Cross-Dataset Evaluation.

[Live demo](https://urmeo.github.io/CalmSense/) · [Colab](https://colab.research.google.com/github/urmeo/CalmSense/blob/main/notebooks/CalmSense.ipynb) · [Code](#architecture) · [Model](#protocol--provenance)

<p align="center">
<a href="https://urmeo.github.io/CalmSense/"><img src="outputs/figures/demo.gif" width="800" alt="CalmSense dashboard"></a>
</p>

## Overview

Stress classification from ECG, EDA, temperature, respiration, and motion.
**15-fold LOSO:** train on 14 subjects; test on one. Static results dashboard.

| **15** | **58** | **869** | **1,032** |
| :--: | :--: | :--: | :--: |
| WESAD subjects | Saved features | Binary windows | Three-class windows |

### Data flow

```mermaid
flowchart TD
    A["WESAD chest"] --> B["Filter signals; detect R-peaks; decompose EDA"]
    B --> C["60 s windows; 50% overlap; ≥90% label purity"]
    C --> D["60 registered features"]
    C --> R["Signals: 5 channels × 1,024 samples"]
    D --> E["LOSO: LR, RF, XGBoost, LightGBM"]
    R --> N["LOSO: 1D-CNN"]
    E --> F["Benchmark metrics"]
    N --> F
    D --> G["Calibration; personalization"]
    D --> H["SHAP; ablation; statistics; tuning"]
    D --> I["Full-data model refit + checksum"]
    W["WESAD wrist"] --> V["Wrist LOSO; chest comparison"]
    E --> V
    W --> P["18 portable features; transfer"]
    O["Non-EEG"] --> P
    F --> J["Dashboard export"]
    G --> J
    H --> J
    V --> J
    P --> J
    J --> K["React dashboard"]
```

## Results

**Historical snapshots.** Corrected code requires a fresh benchmark; see [provenance](#protocol--provenance).
Binary: baseline/stress. Three-class adds amusement.

| Model | Binary acc | Binary F1 | AUROC* | AUPRC* | 3-class acc | 3-class F1 |
| :-- | --: | --: | --: | --: | --: | --: |
| Random Forest | 0.913 | 0.898 | 0.973 | 0.960 | 0.637 | 0.535 |
| XGBoost | 0.903 | 0.873 | 0.975 | 0.960 | 0.633 | 0.552 |
| Logistic Regression | 0.902 | 0.883 | 0.959 | 0.947 | 0.670 | 0.613 |
| LightGBM | 0.894 | 0.860 | 0.965 | 0.946 | 0.658 | 0.568 |
| 1D-CNN | 0.718 | 0.648 | n/a | n/a | 0.626 | 0.543 |

Accuracy/macro-F1: subject means. RF pooled balanced accuracy: **0.903**.
RF accuracy **95% CI [0.860, 0.960]**; four-model comparison **p = 0.806** (no significant difference).
Plots/CSVs cover four feature models; [metrics](outputs/results/metrics.json) also include CNN.

Matched subject-mixed gaps: binary **+5.7 pp**; three-class **0.658 → 0.792 (+13.3 pp)**.

## Graphs & charts

<table width="100%">
<tr>
<td align="center" width="50%"><strong>Binary accuracy · LOSO</strong></td>
<td align="center" width="50%"><strong>Three-class accuracy · LOSO</strong></td>
</tr>
<tr>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/binary_model_comparison.png"><img src="outputs/figures/binary_model_comparison.png" width="390" alt="Feature-model binary LOSO accuracy with subject standard deviation error bars"></a></td>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/multiclass_model_comparison.png"><img src="outputs/figures/multiclass_model_comparison.png" width="390" alt="Feature-model three-class LOSO accuracy with subject standard deviation error bars"></a></td>
</tr>
<tr>
<td align="center"><sub>RF <b>0.913</b> · four feature models</sub></td>
<td align="center"><sub>LR <b>0.670</b> · four feature models</sub></td>
</tr>
<tr>
<td align="center" width="50%"><strong>Subject leakage</strong></td>
<td align="center" width="50%"><strong>Across the 15 subjects</strong></td>
</tr>
<tr>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/binary_optimism_gap.png"><img src="outputs/figures/binary_optimism_gap.png" width="390" alt="Binary accuracy on matched non-overlapping windows under LOSO and subject-mixed testing"></a></td>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/binary_per_subject.png"><img src="outputs/figures/binary_per_subject.png" width="390" alt="Binary Random Forest LOSO accuracy for each held-out subject"></a></td>
</tr>
<tr>
<td align="center"><sub><b>0.907 → 0.964</b> · +5.7 pp</sub></td>
<td align="center"><sub><b>0.712 to 1.000</b> · RF accuracy</sub></td>
</tr>
<tr>
<td align="center" width="50%"><strong>Feature ablation</strong></td>
<td align="center" width="50%"><strong>Chest vs wrist</strong></td>
</tr>
<tr>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/ablation.png"><img src="outputs/figures/ablation.png" width="390" alt="Random Forest binary LOSO accuracy for feature subsets"></a></td>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/chest_vs_wrist.png"><img src="outputs/figures/chest_vs_wrist.png" width="390" alt="Same-model Random Forest binary LOSO accuracy for chest and wrist"></a></td>
</tr>
<tr>
<td align="center"><sub>All <b>0.913</b> · no motion <b>0.901</b></sub></td>
<td align="center"><sub>RF: <b>0.913 vs 0.893</b></sub></td>
</tr>
<tr>
<td align="center" width="50%"><strong>Cross-dataset transfer</strong></td>
<td align="center" width="50%"><strong>SHAP explainability</strong></td>
</tr>
<tr>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/cross_dataset.png"><img src="outputs/figures/cross_dataset.png" width="390" alt="Within-dataset and cross-dataset balanced accuracy on 18 shared features"></a></td>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/shap_beeswarm.png"><img src="outputs/figures/shap_beeswarm.png" width="390" alt="Global signed SHAP contributions and feature values for the full-data gradient-boosted model"></a></td>
</tr>
<tr>
<td align="center"><sub>Balanced accuracy: <b>0.557 / 0.494</b></sub></td>
<td align="center"><sub>Full-data fit: motion · heart rate · EDA · respiration</sub></td>
</tr>
<tr>
<td align="center" width="50%"><strong>Probability calibration</strong></td>
<td align="center" width="50%"><strong>Few-shot personalization</strong></td>
</tr>
<tr>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/calibration_reliability.png"><img src="outputs/figures/calibration_reliability.png" width="293" alt="Confidence versus accuracy before and after training-subject isotonic recalibration"></a></td>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/personalization.png"><img src="outputs/figures/personalization.png" width="390" alt="Mean per-subject calibration error against requested enrollment budget"></a></td>
</tr>
<tr>
<td align="center"><sub>Full LOSO ECE: <b>0.070 → 0.025</b></sub></td>
<td align="center"><sub>ECE: <b>0.146 → 0.069</b> · requested 20</sub></td>
</tr>
<tr>
<td align="center" width="50%"><strong>Binary confusion · RF</strong></td>
<td align="center" width="50%"><strong>Three-class confusion · LR</strong></td>
</tr>
<tr>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/binary_confusion.png"><img src="outputs/figures/binary_confusion.png" width="390" alt="Pooled row-normalized binary Random Forest confusion matrix at default classifier decisions"></a></td>
<td align="center" valign="middle" height="300" width="50%"><a href="outputs/figures/multiclass_confusion.png"><img src="outputs/figures/multiclass_confusion.png" width="390" alt="Pooled row-normalized three-class Logistic Regression confusion matrix"></a></td>
</tr>
<tr>
<td align="center"><sub>Stress recall <b>≈0.87</b> · default decisions</sub></td>
<td align="center"><sub>Baseline ↔ amusement confusion</sub></td>
</tr>
</table>

<details>
<summary>Calibration & personalization</summary>

### Calibration

<!-- AUTOGEN:calibration START -->
| Evaluation | ECE | MCE | Brier |
| --- | :-: | :-: | :-: |
| Subject-mixed 5-fold, non-overlapping | 0.085 | 0.290 | 0.038 |
| LOSO, matched non-overlapping | 0.090 | 0.256 | 0.072 |
| LOSO, all windows | 0.070 | 0.160 | 0.068 |
| LOSO, all windows + isotonic | 0.025 | 0.271 | 0.064 |
<!-- AUTOGEN:calibration END -->

15 bins; pooled RF; training-subject OOF isotonic fit. Plot: all windows; matched rows: non-overlapping subset.
Binary Brier: stress-probability MSE; historical two-class scores/gap/CI halved for display. Source artifacts unchanged.
[Decision curve](outputs/figures/calibration_decision_curve.png).

### Personalization

<!-- AUTOGEN:personalization START -->
| Recalibration / requested enrollment budget | ECE | Brier |
| --- | :-: | :-: |
| None (LOSO) | 0.146 | 0.073 |
| Global (training subjects) | 0.108 | 0.074 |
| Per-subject, budget 5 | 0.097 | 0.061 |
| Per-subject, budget 10 | 0.071 | 0.059 |
| Per-subject, budget 20 | 0.069 | 0.058 |
<!-- AUTOGEN:personalization END -->

Subject means; reserved-half evaluation; disjoint enrollment; no classifier retraining.
Budget 5 enrolls 4 balanced windows; class availability can reduce requests.

</details>

## Protocol & provenance

1. **Evaluation:** 15-fold LOSO; train-only preprocessing/calibration; subject means. *AUROC/AUPRC: [pooled OOF](outputs/results/threshold_metrics.json), unweighted XGBoost; no CNN.
2. **Model:** [Uncalibrated chest RF](outputs/models/stress_classifier.joblib); **869** training windows; scikit-learn **1.6.1**. Use the [verified loader](src/utils.py) from a trusted checkout.
3. **Provenance:** [Historical results](outputs/results/provenance.json) predate extraction/benchmark **v2** and portable **v3**; rerun extraction and fitting. Transfer units differ; SHAP uses a full-data fit.

## Architecture

| Path | Contents |
| :-- | :-- |
| `src/` | Data, preprocessing, features, models, calibration |
| `scripts/` | Experiments, downloads, exports |
| `notebooks/` | Synthetic Colab demo |
| `frontend/` | Dashboard and configs |
| `outputs/{results,figures,models,dashboard}/` | Saved research and dashboard artifacts |
| `outputs/generated/` | Ignored caches, plots, logs, demo, site |

Settings/seed **42**: `src/config.py`. Experiments update results/models; plots stay under `generated/`.

## Models & features

| Model | Settings |
| :-- | :-- |
| Logistic Regression | C=1; L2; class-balanced |
| Random Forest | 200 trees; depth 10; class-balanced |
| XGBoost | 200 trees; depth 7; learning rate 0.1 |
| LightGBM | 200 trees; 50 leaves; learning rate 0.1 |
| 1D-CNN | Residual blocks; AdamW; early stopping |

**60 registered features**: HRV **30**, EDA **15**, temperature/respiration **10**, ACC **5**.
Saved **58** exclude all-NaN `RESP_inhale_exhale_ratio` and `RESP_variability`.

## Tech stack

| Layer | Technologies |
| :-- | :-- |
| Signals & data | Python, NumPy, pandas, SciPy, NeuroKit2, WFDB |
| Models & explanation | scikit-learn, XGBoost, LightGBM, PyTorch, SHAP |
| Figures | Matplotlib, Seaborn |
| Dashboard | React, TypeScript, Vite, Tailwind CSS, Plotly, Recharts |
| Quality | pytest, Ruff, mypy |

## Limitations

1. **Cohort:** 15 lab subjects; wide CIs; no clinical validation.
2. **Analysis:** 60 s windows limit VLF and long-term DFA; weak CNN; exploratory tests lack multiplicity correction.
3. **Transfer:** One dataset pair; units, devices, stressors, and labels differ.

## Future work

1. Validate transfer on a third corpus with matched sensors/labels.
2. Test daily-life recordings outside the lab.
3. Measure live inference latency and wearable battery use.

## Ethics & data use

1. Obtain informed consent for physiological recordings.
2. Collect required signals only; omit identifiers.
3. Research use only; follow dataset terms.

## References

1. Schmidt et al. (2018). [Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection](https://doi.org/10.1145/3242969.3242985). *ACM ICMI*, 400-408.
2. ESC/NASPE Task Force (1996). [Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use](https://doi.org/10.1161/01.CIR.93.5.1043). *Circulation*, 93(5), 1043-1065.
3. Guo et al. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). *ICML*, PMLR 70, 1321-1330.
4. Vos et al. (2023). [Generalizable Machine Learning for Stress Monitoring from Wearable Devices: A Systematic Literature Review](https://doi.org/10.1016/j.ijmedinf.2023.105026). *International Journal of Medical Informatics*, 173, 105026.

## License

[MIT](LICENSE)
