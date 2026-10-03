# CalmSense

[Live demo](https://urmeo.github.io/CalmSense/) · [Colab](https://colab.research.google.com/github/urmeo/CalmSense/blob/main/notebooks/CalmSense.ipynb)

[![CalmSense dashboard](docs/demo.gif)](https://urmeo.github.io/CalmSense/)

## What this is

WESAD stress-vs-baseline benchmark using ECG, EDA, temperature, respiration and motion.
Leave-One-Subject-Out (LOSO): train on 14 subjects, test on the 15th, rotate.
Static dashboard shows committed results; make demo runs offline on synthetic signals.

## Results

Binary LOSO, means over 15 held-out subjects.

<table width="780">
<tr><th align="left" width="180">Model</th><th align="left" width="120">Accuracy</th><th align="left" width="120">F1 (macro)</th><th align="left" width="180">AUROC</th><th align="left" width="180">AUPRC</th></tr>
<tr><td>Random Forest</td><td>0.913</td><td>0.898</td><td>0.973</td><td>0.960</td></tr>
</table>

No significant difference among feature models (Friedman p = 0.81); RF accuracy 95% CI: [0.860, 0.960].
RF threshold (Youden J): 0.45 gives sensitivity 0.90, specificity 0.91, PPV 0.85, NPV 0.94.

- Leakage: same-person testing adds 13 points to 3-class accuracy (0.66 to 0.79); binary +5.7.
- Motion: dropping motion features changes accuracy from 0.913 to 0.901.
- Wrist: 0.893 vs chest 0.913 for the same model; best wrist 0.906, within noise.
- Transfer: recorded balanced accuracies 0.57/0.50; slope-unit mismatch needs a rerun.
- Calibration: isotonic recalibration lowers ECE from 0.070 to 0.025.
- Personalization: five enrollment windows beat global; 20 lower ECE from 0.146 to 0.069.

## Methods

58 features: HRV, EDA, temperature, respiration and motion. Logistic Regression, Random Forest, XGBoost and LightGBM use
median imputation and scaling fit per fold. NeuroKit2/SciPy, SHAP, PyTorch and React/TypeScript
support the pipeline and dashboard. [Model details](docs/MODEL_CARD.md).

## Limitations

- 15 lab subjects: wide confidence intervals, low power, no clinical claim.
- Ablation, calibration and personalization are exploratory, without multiplicity correction; the raw-signal 1D-CNN is a small baseline.
- Synthetic scores only exercise the pipeline; they are not evidence of performance on real subjects.
- WESAD/PhysioNet Non-EEG transfer uses one confounded pair. EDA/TEMP slopes are per sample at different rates (4 Hz vs 8 Hz); scores require recomputation with harmonized units and do not isolate dataset shift. See [provenance](README.md).

## Ethics & data use

Research only. Physiological signals are sensitive personal data: minimize collection and retention;
never monitor or penalize people without informed consent. Datasets retain their own
licenses and are not redistributed.

## License

[MIT License](LICENSE)

### Dataset and model lineage

WESAD is the primary benchmark dataset; PhysioNet Non-EEG is used only for the separate
cross-dataset transfer experiment. WESAD was introduced by Schmidt, Reiss, Duerichen,
Marberger, and Van Laerhoven (ICMI 2018); its attribution also remains in CITATION.cff.
Obtain WESAD from the [official UCI source](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
under its research agreement. The datasets are not redistributed here.

Run `make wesad` and `make data` to download the two datasets. The WESAD loader expects
`data/raw/WESAD/S2/S2.pkl` through `S17/S17.pkl`, excluding S12 (15 subjects; S1 is also absent).
The `latin1` pickles contain chest ACC/ECG/EMG/EDA/Temp/Resp and labels at 700 Hz;
wrist ACC at 32 Hz, BVP at 64 Hz, and EDA/TEMP at 4 Hz. The UCI distribution has no version tag
or official checksums. Run `python scripts/download_data.py --verify-wesad` to compare
the files against the committed reference hashes from the official Uni-Siegen distribution.

The shipped Random Forest at `outputs/models/stress_classifier.joblib` is refit on all
869 WESAD binary windows. Its performance is reported through a separate LOSO evaluation;
the shipped classifier has seen all subjects and uses no pretrained third-party weights.
