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
        logger.info(f"{name} completed in {time.perf_counter() - start:.2f} seconds")


def ensure_directory(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance() -> dict:
    """Record Git HEAD and UTC time, excluding uncommitted code state."""
    import subprocess
    from datetime import datetime, timezone

    root = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        sha = "unknown"
    return {"git_sha": sha, "generated_at": datetime.now(timezone.utc).isoformat()}


def save_verified_joblib(bundle, path: Union[str, Path]) -> None:
    """Write a joblib bundle and the SHA-256 sidecar used to verify its bytes."""
    import hashlib

    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")


def load_verified_joblib(path: Union[str, Path]):
    """Load a joblib bundle only after its bytes match the committed SHA-256 sidecar.

    Unpickling executes arbitrary code, so the shipped model is checked against
    ``<path>.sha256`` before loading; a mismatch raises instead of trusting the file.
    """
    import hashlib

    import joblib

    path = Path(path)
    expected = path.with_name(path.name + ".sha256").read_text().split()[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: refusing to load an unverified pickle."
        )
    return joblib.load(path)


def paired_effect_size(a, b) -> dict:
    """Return paired Cohen's d, corrected Hedges' g, and sample count."""
    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    sd = diff.std(ddof=1)
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    g = d * (1 - 3 / (4 * n - 1)) if n > 1 else d  # Hedges correction for small n
    return {"cohens_d": d, "hedges_g": float(g), "n": int(n)}
