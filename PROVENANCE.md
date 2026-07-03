# Provenance

Where the data and the shipped model come from, and how to verify them.

## Datasets

### WESAD (primary)

Every headline result and the shipped classifier come from WESAD.

| Field | Value |
| ----- | ----- |
| Name | WESAD (Wearable Stress and Affect Detection) |
| Authors | Schmidt, Reiss, Duerichen, Marberger, Van Laerhoven (Uni Siegen) |
| Reference | Schmidt et al., ICMI 2018 (see `data/raw/README.md` for the BibTeX) |
| Subjects used | S2–S17 (15 subjects; S1 and S12 do not exist upstream) |
| Source | UCI Machine Learning Repository, dataset 465 <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection> |
| License / access | Research-only; behind a one-time research agreement. Not redistributed in this repository. |
| Version | UCI distribution as of access; upstream ships no version tag or official checksum. |
| Format | Per-subject `S*.pkl` (latin1 pickle): chest signals at 700 Hz (ACC/ECG/EMG/EDA/Temp/Resp), wrist signals (ACC 32 Hz, BVP 64 Hz, EDA/TEMP 4 Hz), labels at 700 Hz. |

Download and layout instructions: `data/raw/README.md`. Because WESAD ships no
official checksum, the SHA-256 reference values for each `S*.pkl` — computed from
the official Uni-Siegen distribution — are listed there so a downloaded copy can be
verified before use.

### PhysioNet Non-EEG (cross-dataset transfer only)

Used only by `scripts/cross_dataset.py` for the transfer experiment, never for the
headline LOSO benchmark.

| Field | Value |
| ----- | ----- |
| Name | Non-EEG Dataset for Assessment of Neurological Status |
| Authors | Birjandtalab et al. |
| Source | PhysioNet (fetched with `wfdb` via `make data`) |
| Role | Second corpus for cross-dataset transfer; a separate, confounded pair — illustrative, not conclusive (see README Limitations). |

## Shipped model

The dashboard runs a single model in the browser. It is derived from the committed
scikit-learn bundle, which in turn is trained on WESAD.

### `outputs/models/stress_classifier.joblib`

- **What it is:** a scikit-learn `Pipeline` — `SimpleImputer(strategy="median")` →
  `StandardScaler` → classifier — bundled with its feature list and class names
  (`baseline`, `stress`).
- **Classifier:** whichever of {Logistic Regression, Random Forest, XGBoost, LightGBM}
  scored highest by mean LOSO accuracy on the binary task. In the committed snapshot
  that is **Random Forest** (200 trees, depth 10, class-balanced; see `results/metrics.json`).
- **How produced:** `scripts/run_experiment.py` — after the LOSO benchmark, the winning
  classifier is refit on **all** WESAD binary windows (58 features, 869 windows) and
  serialized (`run_experiment.py:420-425`). Seeded (`src/config.SEED`).
- **Trained on:** WESAD only, binary labels (baseline vs stress). No other data.
- **Not a held-out artifact:** this is the deployment model fit on all subjects for
  in-browser demo use. The reported *performance* numbers are the separate LOSO
  evaluation (train on 14, test on the 15th, rotate) — see README Results.

### `frontend/public/model.onnx`

- **How produced:** `scripts/export_onnx.py` converts the Random Forest step of the
  bundle to ONNX with `skl2onnx` (`target_opset=18`, `zipmap=False`). Imputation and
  scaling are **not** in the graph; the imputer medians and scaler mean/scale are
  exported to `frontend/src/model_meta.json` and applied in JavaScript before inference.
- **Trained on:** identical to the joblib bundle above (WESAD binary). ONNX is a format
  conversion, not a retrain.
- **Parity gate:** `export_onnx.py` asserts the ONNX probabilities match the full
  sklearn pipeline to `< 1e-4` on randomized inputs (NaNs included, to exercise the
  imputation path); `tests/test_onnx.py` re-checks parity and labels. This gate runs in
  CI (`.github/workflows/ci.yml`, test job), so a divergent export fails the build.

### Off-the-shelf components

- Feature extraction (HRV/EDA/etc.): NeuroKit2, SciPy.
- Models: scikit-learn, XGBoost, LightGBM, PyTorch (1D-CNN baseline).
- Browser runtime: ONNX Runtime Web (`frontend/public/ort/`), version-locked to the
  `onnx` opset used at export for parity.

No pretrained third-party weights are used; all models are trained from scratch on WESAD.

## Checksums

For the two committed model artifacts (SHA-256). Regenerating the model (`make reproduce`
then `make onnx`) will produce new bytes and new hashes; these pin the current snapshot.

```
624fe215181fd0410723d24bba648bb5eb524bc26fe6ba211f713559ef50cffb  outputs/models/stress_classifier.joblib
c3046cf3d953dc9fe40307858cfc73c4b24d1fdf7aa444588d25ae928085fd53  frontend/public/model.onnx
```

`outputs/models/stress_classifier.joblib.sha256` carries the joblib hash in-tree.
Verify with:

```bash
shasum -a 256 -c outputs/models/stress_classifier.joblib.sha256
shasum -a 256 frontend/public/model.onnx   # compare against the value above
```

## Result artifacts

The JSON/CSV files under `results/` are a fixed snapshot of the WESAD benchmark; each
carries (or gains on the next `make reproduce`) a `provenance` block recording the
`git_sha` and `generated_at` of the run that produced it. See `results/README.md` for
the per-file producer table and the leakage-safety argument for the committed
calibration and personalization snapshots.
