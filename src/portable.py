"""Shared feature space (EDA + TEMP + ACC + HR) for cross-dataset transfer.

The same statistics are computed for WESAD wrist signals and the PhysioNet
Non-EEG dataset, so a model trained on one can be tested on the other. This is
deliberately separate from the full WESAD chest model: cross-dataset transfer
only works on features both devices share.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FS, VALID_SUBJECTS
from .data.loader import WESADLoader

WINDOW_SEC = 60.0
OVERLAP = 0.5
PURITY = 0.9
PORTABLE_FEATURE_VERSION = 2


def _stats(
    x: np.ndarray, prefix: str, keys: List[str], sampling_rate: float = 1.0
) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).ravel()
    finite = np.isfinite(x)
    t = np.flatnonzero(finite) / sampling_rate
    x = x[finite]
    out = {f"{prefix}_{k}": np.nan for k in keys}
    if len(x) < 2:
        return out
    funcs: Dict[str, Callable[[np.ndarray], Any]] = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "range": np.ptp,
        "slope": lambda values: np.polyfit(t, values, 1)[0],
        "energy": lambda values: np.mean(values**2),
    }
    for k in keys:
        out[f"{prefix}_{k}"] = float(funcs[k](x))
    return out


def portable_features(eda, temp, acc_mag, hr, *, eda_fs: float, temp_fs: float) -> Dict[str, float]:
    """18 shared features, with EDA/TEMP slopes in signal units per second."""
    if not all(np.isfinite(rate) and rate > 0 for rate in (eda_fs, temp_fs)):
        raise ValueError("EDA and temperature sampling rates must be finite and positive")
    feats = {}
    feats.update(_stats(eda, "EDA", ["mean", "std", "min", "max", "range", "slope"], eda_fs))
    feats.update(_stats(temp, "TEMP", ["mean", "std", "min", "max", "slope"], temp_fs))
    feats.update(_stats(acc_mag, "ACC", ["mean", "std", "energy"]))
    feats.update(_stats(hr, "HR", ["mean", "std", "min", "max"]))
    return feats


def _window_label(labels: np.ndarray, keep: set) -> Optional[int]:
    values, counts = np.unique(labels, return_counts=True)
    dominant = values[counts.argmax()]
    if dominant not in keep:
        return None
    if counts.max() / counts.sum() < PURITY:
        return None
    return int(dominant)


def wesad_portable(subjects: Optional[List[str]] = None) -> pd.DataFrame:
    """WESAD wrist signals -> shared feature space, binary baseline(0)/stress(1)."""
    import neurokit2 as nk

    loader = WESADLoader()
    subjects = subjects or loader.subjects or VALID_SUBJECTS
    step = WINDOW_SEC * (1 - OVERLAP)
    rows = []

    for sid in subjects:
        data = loader.load_subject(sid)
        wrist = data["wrist"]
        labels = np.asarray(data["label"]).flatten()
        bvp = np.asarray(wrist["BVP"]).flatten()
        eda = np.asarray(wrist["EDA"]).flatten()
        temp = np.asarray(wrist["TEMP"]).flatten()
        acc_mag = np.sqrt(np.sum(np.asarray(wrist["ACC"]) ** 2, axis=1))

        peaks = np.asarray(
            nk.ppg_findpeaks(nk.ppg_clean(bvp, sampling_rate=64), sampling_rate=64)["PPG_Peaks"]
        )
        beat_hr = 60.0 / (np.diff(peaks) / FS.WRIST_BVP)
        beat_t = peaks[1:] / FS.WRIST_BVP

        duration = len(labels) / FS.CHEST
        t = 0.0
        while t + WINDOW_SEC <= duration:
            lab = _window_label(
                labels[int(t * FS.CHEST) : int((t + WINDOW_SEC) * FS.CHEST)], {1, 2}
            )
            if lab is not None:
                e0, e1 = int(t * FS.WRIST_EDA), int((t + WINDOW_SEC) * FS.WRIST_EDA)
                a0, a1 = int(t * FS.WRIST_ACC), int((t + WINDOW_SEC) * FS.WRIST_ACC)
                hr_win = beat_hr[(beat_t >= t) & (beat_t < t + WINDOW_SEC)]
                row: Dict[str, Any] = portable_features(
                    eda[e0:e1],
                    temp[e0:e1],
                    acc_mag[a0:a1],
                    hr_win,
                    eda_fs=FS.WRIST_EDA,
                    temp_fs=FS.WRIST_TEMP,
                )
                row["subject"] = sid
                row["label"] = 0 if lab == 1 else 1  # baseline=0, stress=1
                rows.append(row)
            t += step

    return pd.DataFrame(rows)
