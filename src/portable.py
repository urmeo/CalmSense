"""Shared feature space (EDA + TEMP + ACC + HR) for cross-dataset transfer.

The same statistics are computed for WESAD wrist signals and the PhysioNet
Non-EEG dataset, so a model trained on one can be tested on the other. This is
deliberately separate from the full WESAD chest model: cross-dataset transfer
only works on features both devices share.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import FS, VALID_SUBJECTS
from .data.loader import WESADLoader

WINDOW_SEC = 60.0
OVERLAP = 0.5
PURITY = 0.9
PORTABLE_SCHEMA_VERSION = 2
FEATURE_KEYS = {
    "EDA": ["mean", "std", "min", "max", "range", "slope"],
    "TEMP": ["mean", "std", "min", "max", "slope"],
    "ACC": ["mean", "std", "energy"],
    "HR": ["mean", "std", "min", "max"],
}
PORTABLE_FEATURE_COLUMNS = [
    f"{prefix}_{key}" for prefix, keys in FEATURE_KEYS.items() for key in keys
]


def _stats(
    x: np.ndarray, prefix: str, keys: List[str], sampling_rate: float = 1.0
) -> Dict[str, float]:
    if isinstance(sampling_rate, bool):
        raise ValueError("sampling_rate must be finite and positive")
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("sampling_rate must be finite and positive") from error
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    x = np.asarray(x, dtype=float).ravel()
    t = np.arange(len(x), dtype=float) / sampling_rate
    finite = np.isfinite(x)
    x, t = x[finite], t[finite]
    out = {f"{prefix}_{k}": np.nan for k in keys}
    if len(x) < 2:
        return out
    funcs = {
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "range": np.ptp(x),
        "slope": np.polyfit(t, x, 1)[0],
        "energy": np.mean(x**2),
    }
    for k in keys:
        out[f"{prefix}_{k}"] = float(funcs[k])
    return out


def portable_features(eda, temp, acc_mag, hr, *, eda_fs: float, temp_fs: float) -> Dict[str, float]:
    """18 shared features; EDA/TEMP slopes use seconds at explicit channel rates.

    Invalid signal samples are omitted without compressing their timestamps.
    """
    feats = {}
    feats.update(_stats(eda, "EDA", FEATURE_KEYS["EDA"], eda_fs))
    feats.update(_stats(temp, "TEMP", FEATURE_KEYS["TEMP"], temp_fs))
    feats.update(_stats(acc_mag, "ACC", FEATURE_KEYS["ACC"]))
    feats.update(_stats(hr, "HR", FEATURE_KEYS["HR"]))
    return feats


def _window_label(labels: np.ndarray, keep: set) -> Optional[int]:
    values, counts = np.unique(labels, return_counts=True)
    dominant = values[counts.argmax()]
    if dominant not in keep:
        return None
    if counts.max() / counts.sum() < PURITY:
        return None
    return int(dominant)


def wesad_portable(
    subjects: Optional[List[str]] = None, data_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """WESAD wrist signals -> shared feature space, binary baseline(0)/stress(1)."""
    import neurokit2 as nk

    loader = WESADLoader(data_path=data_path)
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
                t0, t1 = int(t * FS.WRIST_TEMP), int((t + WINDOW_SEC) * FS.WRIST_TEMP)
                a0, a1 = int(t * FS.WRIST_ACC), int((t + WINDOW_SEC) * FS.WRIST_ACC)
                hr_win = beat_hr[(beat_t >= t) & (beat_t < t + WINDOW_SEC)]
                row: Dict[str, Any] = portable_features(
                    eda[e0:e1],
                    temp[t0:t1],
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
