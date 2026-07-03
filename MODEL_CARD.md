# Model Card: CalmSense binary stress classifier

This card describes the model shipped in this repository and served by the browser demo.
It follows the model-card format (Mitchell et al., 2019). Numbers are the committed WESAD
snapshot; see [results/README.md](results/README.md) and results/provenance.json for provenance.

## Model details

- **Model:** Random Forest (200 trees, depth 10, class-balanced) on 58 physiological features.
- **Task:** Binary classification, baseline versus acute stress, from a 60 second window of wearable signals.
- **Inputs:** 58 features from ECG-derived HRV (time, frequency, nonlinear), electrodermal activity, skin temperature, respiration, and accelerometer motion.
- **Output:** A calibrated probability of stress, plus a class label.
- **Pipeline:** Median imputation, standardization, then the classifier, all fit inside each evaluation fold.
- **Export:** Trained with scikit-learn 1.6.1, exported to ONNX (opset via skl2onnx 1.20.0), and run in the browser with ONNX Runtime Web 1.19.2. The exported model is checked for parity with the scikit-learn pipeline to within 1e-4.
- **License:** MIT. **Version:** 1.0.0. **Contact:** github.com/urme-b/CalmSense.

## Intended use

- **Primary use:** A research benchmark for how much reported wearable-stress accuracy survives subject-independent evaluation, and a demonstration of calibrated, in-browser inference.
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

## Metrics (binary, LOSO, mean over held-out subjects)

| Metric | Value |
| --- | :-: |
| Accuracy | 0.913 |
| F1 (macro) | 0.898 |
| Balanced accuracy | 0.903 |
| AUROC | 0.973 |
| AUPRC | 0.960 |

Operating point (Random Forest, Youden J threshold 0.45): sensitivity 0.90, specificity 0.91, PPV 0.85, NPV 0.94.

Calibration on unseen subjects: ECE 0.070, cut to 0.025 by leakage-free isotonic recalibration; a 20-window per-subject enrollment reaches ECE 0.069 without retraining.

Three-class (baseline, stress, amusement) accuracy is far lower at 0.66, close to the majority class, and amusement is the hardest class. The four feature models are a statistical tie (Friedman p = 0.81), so the family is reported rather than a single winner. The 1D-CNN on raw signal is a weak baseline at 0.718.

## Ethical considerations

- Physiological signals are sensitive personal data. Collect and keep only what an analysis needs.
- Stress inference can be misused for monitoring or coercion; deployment without consent is out of scope and discouraged.
- The dataset is small and demographically narrow, so subgroup performance is unknown and fairness is unverified.

## Limitations and caveats

- 15 lab subjects give wide confidence intervals and low power; no clinical claim is made.
- A leakage-free model does not transfer to another dataset, falling to near chance (0.50 balanced accuracy).
- Ablation, calibration, and personalization are exploratory and not multiplicity-corrected.
- Metrics other than the threshold-free AUROC and AUPRC are reported at a fixed threshold.
- The synthetic demo data is near-separable by design; only the real WESAD run is meaningful.

## References

- P. Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection," ICMI, 2018.
- M. Mitchell et al., "Model Cards for Model Reporting," FAT*, 2019.
- Full method and citations: [PAPER.md](PAPER.md).
