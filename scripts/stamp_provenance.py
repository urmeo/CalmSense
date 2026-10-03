"""Record the current environment and result hashes beside experiment outputs.

This snapshot does not prove which code or environment produced existing results.
Run immediately after experiments to record the environment used for that run.
"""

import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import WESAD_SHA256
from src.config import RESULTS_DIR, SEED
from src.utils import provenance, sha256_file, write_json

# Versions that move the numbers if they change; the model pickle is coupled to scikit-learn.
KEY_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "neurokit2",
    "xgboost",
    "lightgbm",
    "torch",
    "shap",
]


def _package_versions() -> dict:
    out = {}
    for pkg in KEY_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _dataset_fingerprint() -> dict:
    """Stable fingerprint of the WESAD subjects used, from their committed SHA-256 checksums.

    Hashes the checksum manifest (not the raw data), so it is reproducible without the
    ~2 GB download present.
    """
    manifest = json.dumps(WESAD_SHA256, sort_keys=True).encode()
    return {
        "dataset": "WESAD",
        "n_subjects": len(WESAD_SHA256),
        "checksum_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def run():
    prov = {
        **provenance(),
        "provenance_kind": "environment_snapshot",
        "seed": SEED,
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "data": _dataset_fingerprint(),
        "result_sha256": {
            str(path.relative_to(RESULTS_DIR)): sha256_file(path)
            for path in sorted(RESULTS_DIR.rglob("*"))
            if path.is_file() and path != RESULTS_DIR / "provenance.json"
        },
    }
    path = RESULTS_DIR / "provenance.json"
    write_json(path, prov)
    print(f"Wrote {path}")
    print(f"  git {prov['git_sha'][:10]}  seed {prov['seed']}  python {prov['python']}")
    print(
        f"  scikit-learn {prov['packages'].get('scikit-learn')}  torch {prov['packages'].get('torch')}"
    )


if __name__ == "__main__":
    run()
