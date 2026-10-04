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
ruff check src/ tests/ scripts/
mypy src/ --ignore-missing-imports
python -m pytest tests/ -q  # CI enforces ≥60% coverage on src/
```

- Target `main` with one focused change.
- Test changed behavior, calibration, and windowing. Verify that held-out subjects stay
  excluded from fitting imputation, scaling, balancing, and calibration.
- Keep local data and `outputs/generated/` out of commits. Preserve published
  [research snapshots](README.md#result-provenance) unless explicitly updating results.
- Use short commit messages: `updated readme`, `updated features`, or `updated tests`.

## Adding a new dataset (for cross-dataset transfer)

Follow [`src/datasets/non_eeg.py`](src/datasets/non_eeg.py): return one `DataFrame` row per
window with the 18 shared HRV/EDA/TEMP/ACC features, `subject`, and `label`
(0 = non-stress, 1 = stress). Register the adapter in `scripts/cross_dataset.py` and
the download with SHA-256 verification in `scripts/download_data.py`.
For generalization claims, evaluate **≥3 corpora with matched stress constructs**;
see [transfer limitations](README.md#cross-dataset-transfer).

## Reporting bugs

Open an issue with the command you ran, the expected vs actual behavior, and your OS/Python version.
Report security vulnerabilities [privately](https://github.com/urmeo/CalmSense/security/advisories/new).
