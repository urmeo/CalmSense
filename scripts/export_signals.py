"""Export real WESAD signal slices as a TypeScript module for the dashboard."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import resample

from src.config import FS, OUTPUT_DIR
from src.data.loader import WESADLoader

SUBJECTS = ["S2", "S3", "S4"]
CONDITIONS = {1: "Baseline", 2: "Stress", 3: "Amusement"}
SECONDS = 40  # per condition
OUT_FS = 30  # display rate
DASHBOARD_SIGNALS = OUTPUT_DIR / "dashboard" / "signals.ts"


def _slice(signal, labels, label, want):
    if len(signal) != len(labels):
        raise ValueError("Signal and label lengths differ")
    idx = np.flatnonzero(labels == label)
    if len(idx) == 0:
        return None
    # A label can occur in separate blocks; never bridge intervening conditions.
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    if len(run) < want:
        return None
    start = int(run[0]) + (len(run) - want) // 2
    return signal[start : start + want]


def run():
    loader = WESADLoader()
    fs = int(FS.CHEST)
    want = SECONDS * fs
    out_n = SECONDS * OUT_FS
    data = {}

    for sid in SUBJECTS:
        d = loader.load_subject(sid)
        chest = d["chest"]
        labels = np.asarray(d["label"]).flatten()
        ecg = chest["ECG"].flatten()
        eda = chest["EDA"].flatten()
        temp = chest["Temp"].flatten()
        acc = np.asarray(chest["ACC"])
        if acc.shape != (len(labels), 3):
            raise ValueError(f"{sid}: chest ACC must align with labels and contain three axes")

        chans = {"ecg": [], "eda": [], "temp": [], "accX": [], "accY": [], "accZ": []}
        conditions = []
        for label, name in CONDITIONS.items():
            sig_slices = {
                "ecg": _slice(ecg, labels, label, want),
                "eda": _slice(eda, labels, label, want),
                "temp": _slice(temp, labels, label, want),
                "accX": _slice(acc[:, 0], labels, label, want),
                "accY": _slice(acc[:, 1], labels, label, want),
                "accZ": _slice(acc[:, 2], labels, label, want),
            }
            if any(v is None for v in sig_slices.values()):
                continue
            for k, v in sig_slices.items():
                chans[k].extend(np.round(resample(v, out_n), 3).tolist())
            conditions.extend([name] * out_n)

        n = len(conditions)
        if n == 0:
            raise ValueError(f"{sid}: no condition contains a complete {SECONDS}s clip")
        data[sid] = {
            "time": np.round(np.arange(n) / OUT_FS, 3).tolist(),
            **chans,
            "conditions": conditions,
        }
        print(f"{sid}: {n} points ({n / OUT_FS:.0f}s real signal)")

    payload = json.dumps(data, allow_nan=False)
    out = DASHBOARD_SIGNALS
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"const data = {payload};\n\nexport default data;\n", encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    run()
