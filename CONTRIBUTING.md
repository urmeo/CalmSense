# Contributing

Python 3.11+; macOS: `brew install libomp`.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m black --check src/ tests/ scripts/
python -m pyflakes src/ tests/ scripts/
python -m pyright src/
python -m coverage run --source=src -m unittest discover -s tests
python -m coverage report --fail-under=60
cd frontend && node tooling.mjs test && node tooling.mjs build
```

- Target `main`; submit one focused change with tests.
- Fit preprocessing and calibration on training subjects only.
- Exclude local data and `outputs/generated/`; preserve [published results](README.md#protocol--provenance).
- Datasets: verify download checksums; specify units and labels.

**Bug reports:** command, expected/actual result, OS/Python version.
[Report security issues privately](https://github.com/urmeo/CalmSense/security/advisories/new).
