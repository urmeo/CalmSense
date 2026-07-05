# Provenance

Where the data and the shipped model come from.

## Datasets

### WESAD (primary)

Every headline result and the shipped classifier come from WESAD.

| Field | Value |
| ----- | ----- |
| Name | WESAD (Wearable Stress and Affect Detection) |
| Authors | Schmidt, Reiss, Duerichen, Marberger, Van Laerhoven (Uni Siegen) |
| Reference | Schmidt et al., ICMI 2018 (see `data/raw/README.md` for the BibTeX) |
| Subjects used | S2 to S17 (15 subjects; S1 and S12 do not exist upstream) |
| Source | UCI Machine Learning Repository, dataset 465 <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection> |
| License / access | Research-only; behind a one-time research agreement. Not redistributed in this repository. |
| Version | UCI distribution as of access; upstream ships no version tag or official checksum. |
| Format | Per-subject `S*.pkl` (latin1 pickle): chest signals at 700 Hz (ACC/ECG/EMG/EDA/Temp/Resp), wrist signals (ACC 32 Hz, BVP 64 Hz, EDA/TEMP 4 Hz), labels at 700 Hz. |

Download and layout instructions: `data/raw/README.md`. Because WESAD ships no
official checksum, the SHA-256 reference values for each `S*.pkl`, computed from
the official Uni-Siegen distribution, are listed there so a downloaded copy can be
verified before use.

### PhysioNet Non-EEG (cross-dataset transfer only)

Used only by `scripts/cross_dataset.py` for the transfer experiment, never for the
headline LOSO benchmark.

| Field | Value |
| ----- | ----- |
| Name | Non-EEG Dataset for Assessment of Neurological Status |
| Authors | Birjandtalab et al. |
| Source | PhysioNet (downloaded as a zip via `make data`; records read with `wfdb`) |
| Role | Second corpus for cross-dataset transfer; a separate, confounded pair, illustrative, not conclusive (see README Limitations). |

## Shipped model

The dashboard runs a Random Forest (the best of Logistic Regression, Random Forest, XGBoost, LightGBM by LOSO accuracy) refit on all 869 WESAD binary windows and exported to ONNX at `frontend/public/model.onnx`, with a `< 1e-4` sklearn-parity check enforced in CI.
It is a deployment model fit on all subjects for the browser demo; the reported performance is the separate LOSO evaluation (see README Results), and no pretrained third-party weights are used.
