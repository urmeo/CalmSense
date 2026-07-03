"""Write results/provenance.json: the exact context the committed numbers came from.

Closes the reproducibility loop for an auditor: which commit, which seed, which package
versions, which dataset. Run at the end of `make reproduce` (and standalone any time).
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import WESAD_SHA256
from src.config import PROJECT_ROOT, SEED

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
    "skl2onnx",
    "onnx",
    "onnxruntime",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


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
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "data": _dataset_fingerprint(),
    }
    path = PROJECT_ROOT / "results" / "provenance.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Wrote {path}")
    print(f"  git {prov['git_sha'][:10]}  seed {prov['seed']}  python {prov['python']}")
    print(f"  scikit-learn {prov['packages'].get('scikit-learn')}  torch {prov['packages'].get('torch')}")


if __name__ == "__main__":
    run()
