import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """PYTHONHASHSEED affects child processes."""
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


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Replace complete text; preserve permissions."""
    from stat import S_IMODE
    from tempfile import TemporaryDirectory

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(text, encoding="utf-8")
        if path.exists():
            temporary.chmod(S_IMODE(path.stat().st_mode))
        temporary.replace(path)


def write_json(path: Union[str, Path], value) -> None:
    """Reject nonfinite JSON before replacement."""
    import json

    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False))


def provenance() -> dict:
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
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
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
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace_verified_pair(new_primary, new_sidecar, primary, sidecar) -> None:
    """Best-effort rollback across two files."""
    from shutil import copy2
    from tempfile import TemporaryDirectory

    primary, sidecar = Path(primary), Path(sidecar)
    with TemporaryDirectory(dir=primary.parent) as directory:
        backups = []
        for index, destination in enumerate((primary, sidecar)):
            backup = Path(directory) / str(index)
            if destination.exists():
                copy2(destination, backup)
            backups.append((destination, backup))
        Path(new_primary).replace(primary)
        try:
            Path(new_sidecar).replace(sidecar)
        except OSError:
            for destination, backup in backups:
                if backup.exists():
                    backup.replace(destination)
                else:
                    destination.unlink(missing_ok=True)
            raise


def save_verified_joblib(bundle, path: Union[str, Path]) -> None:
    from tempfile import TemporaryDirectory

    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        checksum = temporary.with_name(path.name + ".sha256")
        joblib.dump(bundle, temporary)
        digest = sha256_file(temporary)
        checksum.write_text(f"{digest}  {path.name}\n")
        replace_verified_pair(
            temporary, checksum, path, path.with_name(path.name + ".sha256")
        )


def load_verified_joblib(path: Union[str, Path]):
    """Hash-check trusted bundles before deserialization."""
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
    # Deserialize the verified snapshot.
    return joblib.load(io.BytesIO(payload))


def paired_effect_size(a, b) -> dict:
    """Paired effects; constant nonzero differences are undefined."""
    from math import exp, lgamma, log

    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 3:
        raise ValueError("Paired effect size requires at least three aligned 1D pairs")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired observations must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        diff = a - b
    n = len(diff)
    if not np.isfinite(diff).all():
        raise ValueError("Paired differences exceed the supported numerical range")
    magnitude = float(np.abs(diff).max())
    if magnitude == 0:
        return {"cohens_d": 0.0, "hedges_g": 0.0, "n": int(n)}
    # Scaling prevents underflow and overflow.
    scaled = diff / magnitude
    sd = scaled.std(ddof=1)
    if sd <= np.finfo(float).eps:
        return {"cohens_d": None, "hedges_g": None, "n": int(n)}
    d = float(scaled.mean() / sd)
    df = n - 1
    correction = exp(lgamma(df / 2) - 0.5 * log(df / 2) - lgamma((df - 1) / 2))
    g = d * correction
    return {"cohens_d": d, "hedges_g": float(g), "n": int(n)}
