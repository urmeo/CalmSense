# CalmSense

[Live demo](https://urmeo.github.io/CalmSense/) · [Colab](https://colab.research.google.com/github/urmeo/CalmSense/blob/main/notebooks/CalmSense.ipynb)

[![CalmSense dashboard](docs/demo.gif)](https://urmeo.github.io/CalmSense/)

WESAD stress-vs-baseline research benchmark: 58 features from ECG, EDA, temperature,
respiration and motion. Leave-One-Subject-Out (LOSO) trains on 14 subjects and tests
on the 15th. The dashboard displays committed results; the offline demo uses synthetic signals.

## Run locally

Python 3.11 or 3.12, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
make demo
```

[Dataset setup](README.md) · [Frontend setup](frontend/README.md) · [Contributing and tests](CONTRIBUTING.md)

## Results

Binary LOSO: accuracy/F1 average 15 held-out subjects; AUROC/AUPRC pool held-out predictions.

| Model | Accuracy | Macro F1 | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| Random Forest | 0.913 | 0.898 | 0.973 | 0.960 |

RF accuracy 95% CI: [0.860, 0.960]. No significant difference among feature models (Friedman p = 0.81).

- Subject mixing inflates matched 3-class accuracy from 0.658 to 0.792; binary gains 5.7 points.
- Dropping motion features changes accuracy from 0.913 to 0.901; wrist RF reaches 0.893.
- Transfer balanced accuracy: 0.557 WESAD to PhysioNet, 0.494 in reverse, with consistent slope units.
- Isotonic recalibration lowers ECE from 0.070 to 0.025. Personalization with 14 to 15 enrollment windows lowers ECE from 0.151 to 0.069.

Median imputation and scaling fit inside each fold. [Model and methods](docs/MODEL_CARD.md) · [Result provenance](results/README.md).

## Limitations

Only 15 lab subjects; wide uncertainty, no clinical claim. Transfer compares different datasets
and stress tasks. Exploratory analyses lack multiplicity correction; the raw-signal CNN is a small baseline.
SHAP describes training-data XGBoost, and the shipped RF is uncalibrated. Synthetic scores test functionality,
not real-world performance.

Physiological data is sensitive: minimize collection and retention, and require informed consent.
Datasets retain their own licenses and are not redistributed. Code: [MIT License](LICENSE).
