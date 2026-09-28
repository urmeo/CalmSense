# Model Card: CalmSense binary stress classifier

This card describes the model shipped in this repository.
It follows the model-card format (Mitchell et al., 2019). Numbers are the committed WESAD
snapshot; see [results/README.md](../results/README.md) and results/provenance.json for provenance.

## Model details

- **Model:** Random Forest (200 trees, depth 10, class-balanced) on 58 physiological features.
- **Task:** Binary classification, baseline versus acute stress, from a 60 second window of wearable signals.
- **Inputs:** 58 features from ECG-derived HRV (time, frequency, nonlinear), electrodermal activity, skin temperature, respiration, and accelerometer motion.
- **Output:** An uncalibrated Random Forest probability of stress, plus a class label. Recalibration is evaluated separately and is not included in the shipped artifact.
- **Pipeline:** Median imputation, standardization, then the classifier, all fit inside each evaluation fold.
- **Artifact:** Trained with scikit-learn 1.6.1 and shipped as `outputs/models/stress_classifier.joblib`.
- **License:** MIT. **Version:** 1.0.0. **Contact:** github.com/urmeo/CalmSense.

## Intended use

- **Primary use:** A research benchmark for how much reported wearable-stress accuracy survives subject-independent evaluation.
- **Users:** Researchers and engineers studying physiological stress detection, evaluation methodology, and probability calibration.
- **Scope:** Educational and methodological. The value is the honest evaluation, not a deployable stress detector.

## Out-of-scope use

- **Not a medical device.** Do not use for diagnosis, screening, treatment, or any clinical or safety-critical decision.
- **Not for surveillance.** Do not monitor, score, or penalize people without informed consent.
- **Not validated in the field.** Trained and evaluated on 15 adults under lab-induced stress; behavior on free-living data, other populations, or other devices is unknown.

## Training and evaluation data

- **Dataset:** WESAD (Schmidt et al., 2018), 15 subjects, chest RespiBAN at 700 Hz and wrist Empatica E4. Labels: baseline, stress, amusement; meditation is dropped as a recovery state.
- **Windows:** 60 seconds at 50 percent overlap, kept only if at least 90 percent of samples share one label. 869 windows for the binary task.
- **Evaluation:** Leave-One-Subject-Out (train on 14 subjects, test on the held-out subject, rotate). Imputation, scaling, class balancing, and any recalibration are fit on training subjects only, so the held-out subject is never seen during fitting.
- **Transfer check:** PhysioNet Non-EEG (20 subjects) is used only to measure cross-dataset transfer on a shared 18-feature space.

## Metrics (binary, LOSO)

Accuracy and F1 are means over held-out subjects; balanced accuracy, AUROC and AUPRC use pooled held-out predictions.

| Metric | Value |
| --- | :-: |
| Accuracy | 0.913 |
| F1 (macro) | 0.898 |
| Balanced accuracy | 0.903 |
| AUROC | 0.973 |
| AUPRC | 0.960 |

Exploratory operating point (Random Forest, Youden J threshold 0.45): sensitivity 0.90, specificity 0.91, PPV 0.85, NPV 0.94. The threshold is selected and evaluated on the same pooled LOSO predictions, so these rates are not an independent validation of the chosen threshold.

Calibration on unseen subjects: ECE 0.070, cut to 0.025 by leakage-free isotonic recalibration. In the separate personalization evaluation, ECE falls from 0.151 to 0.084 with five enrollment windows and 0.071 with ten. A budget of 20 gives ECE 0.069 using the 14 to 15 windows available per subject. Enrollment and evaluation windows are disjoint; actual enrollment counts are recorded in the result metadata.

Three-class (baseline, stress, amusement) accuracy averages 0.67 over held-out subjects. No significant difference was found among the four binary feature models (Friedman p = 0.81). The 1D-CNN on raw signal is a weak baseline (0.718 binary, 0.626 three-class).

## Ethical considerations

- Physiological signals are sensitive personal data. Collect and keep only what an analysis needs.
- Stress inference can be misused for monitoring or coercion; deployment without consent is out of scope and discouraged.
- The dataset is small and demographically narrow, so subgroup performance is unknown and fairness is unverified.

## Limitations and caveats

- 15 lab subjects give wide confidence intervals and low power; no clinical claim is made.
- Transfer was rerun with per-second EDA/TEMP slopes and rebuilt caches: balanced accuracy is 0.557 from WESAD to Non-EEG and 0.494 in reverse. Corpus and label differences still confound the comparison. See [README limitations](../README.md#limitations).
- Ablation, calibration, and personalization are exploratory and not multiplicity-corrected.
- Accuracy and F1 use the classifier's default decision rule; the exploratory Youden-J operating point is selected separately.
- SHAP importance comes from XGBoost fit and explained on all binary windows. It is an exploratory training-data summary, not held-out importance or an explanation of the shipped Random Forest.
- The synthetic demo data is near-separable by design; only the real WESAD run is meaningful.

## References

- P. Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection," ICMI, 2018.
- M. Mitchell et al., "Model Cards for Model Reporting," FAT*, 2019.
