# Model Card: CalmSense binary stress classifier

This card describes the model shipped in this repository.
It follows the model-card format (Mitchell et al., 2019). Numbers are the committed WESAD
snapshot; see [results/README.md](../results/README.md) and results/provenance.json for provenance.

## Model details

- **Model:** Random Forest (200 trees, depth 10, class-balanced) on 58 physiological features.
- **Task:** Binary classification, baseline versus acute stress, from a 60 second window of wearable signals.
- **Inputs:** 58 features from ECG-derived HRV (time, frequency, nonlinear), electrodermal activity, skin temperature, respiration, and accelerometer motion.
- **Output:** An uncalibrated Random Forest probability of stress and a class label. The shipped artifact contains no isotonic or sigmoid calibrator.
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
- **Not validated in the field.** Trained and evaluated on 15 adults under lab-induced stress; behavior on free-living data, other populations, or other devices is unknown and expected to be worse (see transfer results below).

## Training and evaluation data

- **Dataset:** WESAD (Schmidt et al., 2018), 15 subjects, chest RespiBAN at 700 Hz and wrist Empatica E4. Labels: baseline, stress, amusement; meditation is dropped as a recovery state.
- **Windows:** 60 seconds at 50 percent overlap, kept only if at least 90 percent of samples share one label. 869 windows for the binary task.
- **Evaluation:** Leave-One-Subject-Out (train on 14 subjects, test on the held-out subject, rotate). Imputation, scaling, class balancing, and any recalibration are fit on training subjects only, so the held-out subject is never seen during fitting.
- **Transfer check:** PhysioNet Non-EEG (20 subjects) is used only to measure cross-dataset transfer on a shared 18-feature space.

## Metrics (binary, LOSO)

| Metric | Value | Aggregation |
| --- | :-: | --- |
| Accuracy | 0.913 | Mean over held-out subjects |
| F1 (macro) | 0.898 | Mean over held-out subjects |
| Balanced accuracy | 0.903 | Pooled default decisions |
| AUROC | 0.973 | Pooled probabilities, separate threshold pass |
| AUPRC (average precision) | 0.960 | Pooled probabilities, separate threshold pass |

Exploratory operating point (Random Forest, Youden J threshold 0.454): sensitivity 0.902, specificity 0.913, PPV 0.850, NPV 0.945. The threshold was selected on the evaluated predictions, not a separate validation set; these rates do not describe the artifact's default decisions.

Separate calibration experiment: full-window LOSO ECE 0.070, reduced to 0.025 by training-subject isotonic recalibration. Personalization uses subject means on a reserved non-overlapping evaluation half: requested budget 20 reaches ECE 0.069 without classifier retraining; actual enrollment may be smaller. These recalibration maps are not shipped in the model.

Three-class accuracy is 0.637 for RF and 0.670 for Logistic Regression; amusement is the hardest class. No significant binary accuracy difference was detected among the four feature models (Friedman p = 0.806); this does not establish equivalence. The raw-signal 1D-CNN is a weak baseline (0.718 binary, 0.626 three-class).

## Ethical considerations

- Physiological signals are sensitive personal data. Collect and keep only what an analysis needs.
- Stress inference can be misused for monitoring or coercion; deployment without consent is out of scope and discouraged.
- The dataset is small and demographically narrow, so subgroup performance is unknown and fairness is unverified.

## Limitations and caveats

- 15 lab subjects give wide confidence intervals and low power; no clinical claim is made.
- Cross-dataset transfer uses a separate wrist-feature RF, not the shipped chest model. Devices, stressors and labels differ; see [transfer results and history](../results/README.md).
- Ablation, calibration, and personalization are exploratory and not multiplicity-corrected.
- Class metrics use default classifier decisions; the Youden J rates use a separate, exploratory threshold.
- The synthetic demo data is near-separable by design; only the real WESAD run is meaningful.

## References

- P. Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection," ICMI, 2018.
- M. Mitchell et al., "Model Cards for Model Reporting," FAT*, 2019.
- Full method and citations: [README.md](../README.md).
