import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Deterministic mode warns when an operation lacks a deterministic kernel.
    ``PYTHONHASHSEED`` applies to new child processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


@contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    from .logging_config import get_logger

    logger = get_logger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"{name} finished in {time.perf_counter() - start:.2f} seconds")


def ensure_directory(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance() -> dict:
    """Record Git HEAD, dirty state, feature schema, and UTC time."""
    import subprocess
    from datetime import datetime, timezone

    from .features.feature_pipeline import FEATURE_SCHEMA_VERSION

    root = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
        dirty = None
    return {
        "git_sha": sha,
        "working_tree_dirty": dirty,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def sha256_file(path: Union[str, Path]) -> str:
    """Hash a file without loading the entire recording into memory."""
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_verified_joblib(bundle, path: Union[str, Path]) -> None:
    """Write a joblib bundle and the SHA-256 sidecar used to verify its bytes."""
    from tempfile import TemporaryDirectory

    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Finish serialization before replacing any existing verified model.
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        checksum = temporary.with_name(path.name + ".sha256")
        joblib.dump(bundle, temporary)
        digest = sha256_file(temporary)
        checksum.write_text(f"{digest}  {path.name}\n")
        temporary.replace(path)
        checksum.replace(path.with_name(path.name + ".sha256"))


def load_verified_joblib(path: Union[str, Path]):
    """Load a joblib bundle only after its bytes match the committed SHA-256 sidecar.

    Unpickling executes arbitrary code, so the shipped model is checked against
    ``<path>.sha256`` before loading; a mismatch raises instead of trusting the file.
    """
    import hashlib
    import io
    import re

    import joblib

    path = Path(path)
    checksum = path.with_name(path.name + ".sha256").read_text().split()
    if not checksum or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum[0]):
        raise ValueError(f"Invalid SHA-256 sidecar for {path.name}")
    expected = checksum[0].lower()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: refusing to load an unverified pickle."
        )
    # Deserialize the checked snapshot, even if the original path is replaced meanwhile.
    return joblib.load(io.BytesIO(payload))


def paired_effect_size(a, b) -> dict:
    """Return paired d_z and g_z for at least three finite, aligned observations.

    Constant nonzero differences have undefined standardized effects, returned as None.
    """
    from math import exp, lgamma, log

    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 3:
        raise ValueError("Paired effect size requires at least three aligned 1D pairs")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired observations must be finite")
    diff = a - b
    n = len(diff)
    sd = diff.std(ddof=1)
    if not np.isfinite(diff).all() or not np.isfinite(sd):
        raise ValueError("Paired differences exceed the supported numerical range")
    if sd <= np.finfo(float).eps * max(1.0, float(np.abs(diff).max())):
        effect = 0.0 if np.all(diff == 0) else None
        return {"cohens_d": effect, "hedges_g": effect, "n": int(n)}
    d = float(diff.mean() / sd)
    # Paired differences estimate their SD with n-1 degrees of freedom.
    df = n - 1
    correction = exp(lgamma(df / 2) - 0.5 * log(df / 2) - lgamma((df - 1) / 2))
    g = d * correction
    return {"cohens_d": d, "hedges_g": float(g), "n": int(n)}
