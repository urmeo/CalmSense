"""Write results/provenance.json after a full reproduction.

Run at the end of `make reproduce`; replacing an existing stamp requires --overwrite.
"""

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import WESAD_SHA256
from src.config import PROJECT_ROOT, SEED
from src.utils import provenance

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


def run(*, overwrite=False):
    path = PROJECT_ROOT / "results" / "provenance.json"
    if path.exists() and not overwrite:
        raise SystemExit(
            "Existing provenance preserved. Use --overwrite only after a full reproduction "
            "(the final step of 'make reproduce')."
        )
    prov = {
        **provenance(),
        "seed": SEED,
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "data": _dataset_fingerprint(),
    }
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Wrote {path}")
    print(f"  git {prov['git_sha'][:10]}  seed {prov['seed']}  python {prov['python']}")
    print(
        f"  scikit-learn {prov['packages'].get('scikit-learn')}  torch {prov['packages'].get('torch')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace lineage after a full reproduction"
    )
    run(overwrite=parser.parse_args().overwrite)
