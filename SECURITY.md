# Security Policy

CalmSense is research software, not a medical device. Security fixes target current `main`;
older releases and experiment snapshots have no separate maintenance. CI tests Python **3.11 / 3.12**.

## Report privately

Use [Report a vulnerability](https://github.com/urmeo/CalmSense/security/advisories/new).
Keep exploit details out of public issues and pull requests.

Include:

- Affected commit/version, component, and environment.
- Minimal reproduction steps using synthetic data.
- Impact, prerequisites, and redacted evidence; a suggested fix if available.

Reports are handled on a best-effort basis. Use the private thread for updates and coordinate
disclosure after a fix or mitigation. No guaranteed response or fix deadline.

Report code execution, unsafe archive extraction, browser injection, credential exposure,
or unauthorized data disclosure. Ordinary bugs and research-result questions belong in
[Issues](https://github.com/urmeo/CalmSense/issues).

## Trusted files only

**Pickle and joblib can execute arbitrary code.** This applies to WESAD subjects, feature caches,
and model artifacts. Only load files from a trusted source or your own trusted pipeline.

- **WESAD:** follow the [dataset instructions](README.md#dataset-download-and-integrity).
  Verify all subjects before loading, including existing or manually extracted files.
  The loader does not check hashes automatically.
- **Feature caches:** regenerate them locally; do not load third-party caches.
- **Model:** use [`load_verified_joblib`](src/utils.py), which checks the SHA-256 sidecar before
  deserialization. Obtain both model and checksum from a trusted checkout.

From the repository root:

```bash
python scripts/download_data.py --verify-wesad
(cd outputs/models && shasum -a 256 -c stress_classifier.joblib.sha256)
```

**A checksum detects changed bytes; it does not establish trust or make a pickle safe.**
WESAD hashes are repository reference values, not publisher-issued checksums.
If verification fails, stop and obtain a trusted copy; do not replace the expected hash to bypass it.

## Dashboard and downloads

The dashboard displays precomputed results. It has no application backend, model inference,
or file uploads. Browser code, dependencies, and external resources remain part of its attack surface.

The downloader rejects archive path traversal and archives declaring more than **10 GiB**
uncompressed. These checks do not authenticate downloaded content.

## Security checks

[CI](.github/workflows/ci.yml) runs on pull requests to `main` and pushes to `main`:

| Check | Coverage |
| --- | --- |
| `pip-audit` | Known vulnerabilities in installed Python dependencies |
| `npm audit` | Frontend dependencies; moderate or higher findings fail CI |
| Gitleaks | Secret scanning with full-history checkout |

Passing checks is not a guarantee of security.

## Exposed secrets or participant data

Revoke or rotate exposed credentials immediately, then report privately. Deleting a file or
commit does not revoke a secret. Never include live credentials, identifying information,
or participant recordings in reports, issues, or commits; use redacted or synthetic examples.
