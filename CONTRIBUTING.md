# Contributing to CalmSense

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/calibration.py --synthetic  # calibration smoke check; no download
```

## Before you open a PR

```bash
ruff format src/ tests/ scripts/
ruff check --fix src/ tests/ scripts/
ruff check src/ tests/ scripts/
mypy src/ --ignore-missing-imports
python -m pytest tests/ -q  # CI enforces ≥60% coverage on src/
```

- Target `main` with one focused change.
- Test behavior and methodology changes, including leakage, calibration, and windowing.
- Fit imputation, scaling, balancing, and calibration inside each LOSO training fold.
  Tests must verify that the held-out subject stays excluded from fitting.
- **Don't commit generated artifacts** in `outputs/generated/` or local datasets in `data/`.
  Committed `outputs/results/` and `outputs/figures/` preserve research snapshots;
  see [result provenance](README.md#result-provenance).
- Use short commit messages, such as `updated readme`, `updated features`, or `updated tests`.

## Adding a new dataset (for cross-dataset transfer)

Use [`src/datasets/non_eeg.py`](src/datasets/non_eeg.py) as the template. A dataset module needs one
function that returns a per-window `DataFrame`:

```python
def build(subjects: list[str] | None = None) -> pd.DataFrame:
    # one row per window, with the shared device-agnostic feature columns
    # plus "subject" and "label" (0 = non-stress, 1 = stress).
    ...
```

- Add the module to `scripts/cross_dataset.py` and its download to `scripts/download_data.py`,
  with [SHA-256 verification](README.md#dataset-download-and-integrity).
- Use shared HRV/EDA/TEMP/ACC summaries and binary stress/non-stress labels.
- Evaluate leave-one-dataset-out generalization on **≥3 corpora with matched stress constructs**;
  see [transfer limitations](README.md#cross-dataset-transfer).

## Reporting bugs

Open an issue with the command you ran, the expected vs actual behavior, and your OS/Python version.
For security concerns, see [SECURITY.md](SECURITY.md) instead.
