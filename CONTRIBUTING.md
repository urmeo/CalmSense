# Contributing to CalmSense

## Setup and checks

Use Python 3.11+ from the repository root (macOS also needs `brew install libomp`).

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/calibration.py --synthetic
python -m black --check src/ tests/ scripts/
python -m pyflakes src/ tests/ scripts/
python -m pyright src/
python -m coverage run --source=src -m unittest discover -s tests
python -m coverage report --fail-under=60
cd frontend && node tooling.mjs test && node tooling.mjs build
```

## Pull requests

- Target `main` with a focused change, short commit message, and tests.
- Test calibration/windowing; keep held-out subjects out of preprocessing and calibration fits.
- Exclude local data and `outputs/generated/`; preserve [published results](README.md#protocol--provenance) unless updating them explicitly.
- New datasets: follow [the adapter](src/datasets/non_eeg.py), register in `scripts/cross_dataset.py`, and add SHA-256-verified downloads in `scripts/download_data.py`. Generalization claims need at least three matched corpora.

## Bug reports

Include the command, expected/actual behavior, and OS/Python version. Report security issues [privately](https://github.com/urmeo/CalmSense/security/advisories/new).
