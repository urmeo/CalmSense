# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[security advisories](https://github.com/urmeo/CalmSense/security/advisories/new) rather than opening a
public issue. Include steps to reproduce and the affected version or commit. You can expect an
acknowledgement within a few days.

## Threat model & known risks

CalmSense is research software, not a medical device or production service.

- **Pickle deserialization (WESAD).** WESAD subjects are distributed as Python pickles, and
  src/data/loader.py unpickles them (encoding="latin1"). Unpickling executes arbitrary code,
  **only load .pkl files you downloaded from the official WESAD source or generated yourself.** See
  [data/raw/README.md](data/raw/README.md).
- **Model deserialization (trust boundary).** The pipeline writes and reloads one model,
  outputs/models/stress_classifier.joblib (scripts/run_experiment.py). joblib
  uses pickle, so loading executes code: only load the model **this repo's own pipeline produced**. Its
  SHA-256 is pinned in
  [outputs/models/stress_classifier.joblib.sha256](outputs/models/stress_classifier.joblib.sha256);
  verify with shasum -a 256 -c outputs/models/stress_classifier.joblib.sha256, and never load a
  third-party .joblib. The public dashboard loads no model at all, it renders committed JSON.

## Static dashboard (no backend)

The dashboard has no server and runs no model: it renders the committed experiment output
(frontend/src/results.json, frontend/src/signals.json), so there is no server-side attack
surface and it accepts no user input or uploads.

## Supply chain & secrets

Dependency ranges are declared in requirements.txt with exact pins in requirements.lock, and audited in CI with pip-audit. The full git history
(all refs) is scanned for committed secrets with [gitleaks](https://github.com/gitleaks/gitleaks),
last run: **0 findings**.
