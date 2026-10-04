from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FS, PROCESSED_DATA_DIR, VALID_SUBJECTS
from .data.loader import WESADLoader
from .dataset import CONDITION_LABELS, window_label
from .features.feature_pipeline import FEATURE_SCHEMA_VERSION, FeatureExtractionPipeline
from .logging_config import LoggerMixin
from .preprocessing.ecg_processor import ECGProcessor
from .preprocessing.eda_processor import EDAProcessor
from .preprocessing.filters import _window_parameters


class WristDataset(LoggerMixin):
    def __init__(self, window_sec: float = 60.0, overlap: float = 0.5, purity: float = 0.9):
        _window_parameters(window_sec, overlap, purity, FS.WRIST_EDA)
        self.window_sec = float(window_sec)
        self.overlap = float(overlap)
        self.purity = float(purity)
        self.label_fs = FS.CHEST
        self.loader = WESADLoader()
        self.eda = EDAProcessor(sampling_rate=FS.WRIST_EDA)
        self.rr = ECGProcessor(sampling_rate=FS.WRIST_BVP)
        self.features = FeatureExtractionPipeline(
            chest_fs=FS.CHEST, wrist_eda_fs=FS.WRIST_EDA, wrist_acc_fs=FS.WRIST_ACC
        )

    def _window_label(self, labels: np.ndarray) -> Optional[int]:
        return window_label(labels, self.purity)

    def _bvp_peaks(self, bvp: np.ndarray) -> np.ndarray:
        import neurokit2 as nk

        clean = nk.ppg_clean(bvp, sampling_rate=int(FS.WRIST_BVP))
        info = nk.ppg_findpeaks(clean, sampling_rate=int(FS.WRIST_BVP))
        return np.asarray(info["PPG_Peaks"])

    def _process_subject(self, subject_id: str) -> Tuple[List[dict], List[int]]:
        data = self.loader.load_subject(subject_id)
        wrist = data["wrist"]
        labels = np.asarray(data["label"]).flatten()

        bvp = np.asarray(wrist["BVP"]).flatten()
        eda = np.asarray(wrist["EDA"]).flatten()
        temp = np.asarray(wrist["TEMP"]).flatten()
        acc = np.asarray(wrist["ACC"])
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))

        peaks = self._bvp_peaks(bvp)
        eda_filt = self.eda.lowpass_filter(eda)
        tonic, phasic = self.eda.decompose_eda(eda_filt)
        _, scr_features = self.eda.detect_scr_peaks(phasic)
        scr_idx = np.array([s["peak_idx"] for s in scr_features])

        step = self.window_sec * (1 - self.overlap)
        duration = len(labels) / self.label_fs
        windows, ys = [], []
        t = 0.0
        while t + self.window_sec <= duration:
            lab = self._window_label(
                labels[int(t * self.label_fs) : int((t + self.window_sec) * self.label_fs)]
            )
            if lab is not None:
                b0, b1 = int(t * FS.WRIST_BVP), int((t + self.window_sec) * FS.WRIST_BVP)
                in_win = (peaks >= b0) & (peaks < b1)
                rr = self.rr.extract_rr_intervals(peaks[in_win], unit="ms")
                _, valid = self.rr.remove_ectopic_beats(rr)
                rr_clean = self.rr.interpolate_artifacts(rr, valid)

                e0, e1 = int(t * FS.WRIST_EDA), int((t + self.window_sec) * FS.WRIST_EDA)
                a0, a1 = int(t * FS.WRIST_ACC), int((t + self.window_sec) * FS.WRIST_ACC)
                scr_in = [s for s, idx in zip(scr_features, scr_idx) if e0 <= idx < e1]

                windows.append(
                    {
                        "rr_intervals": rr_clean,
                        "eda_tonic": tonic[e0:e1],
                        "eda_phasic": phasic[e0:e1],
                        "eda_raw": eda_filt[e0:e1],
                        "scr_peaks": scr_in,
                        "temperature": temp[e0:e1],
                        "accelerometer": {"magnitude": acc_mag[a0:a1]},
                        "subject_id": subject_id,
                        "window_id": int(t * self.label_fs),
                        "label": lab,
                    }
                )
                ys.append(lab)
            t += step

        self.logger.info(f"{subject_id} (wrist): {len(ys)} windows")
        return windows, ys

    def build(self, subjects: Optional[List[str]] = None, cache: bool = True) -> pd.DataFrame:
        subjects = self.loader.subjects if subjects is None else subjects
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("subjects must be a nonempty list without duplicates")
        frames = []
        for s in subjects:
            windows, _ = self._process_subject(s)
            if windows:
                frames.append(self.features.extract_all_features(windows, show_progress=False))
            del windows
        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else self.features.extract_all_features([], show_progress=False)
        )
        df["label_name"] = df["label"].map(CONDITION_LABELS)
        df.attrs["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        df.attrs["dataset_parameters"] = {
            "window_sec": self.window_sec,
            "overlap": self.overlap,
            "purity": self.purity,
            "subjects": sorted(subjects),
        }
        if cache:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(PROCESSED_DATA_DIR / "features_wrist.parquet", index=False)
            self.logger.info(f"Cached wrist features ({len(df)} windows)")
        return df


def load_wrist() -> Optional[pd.DataFrame]:
    path = Path(PROCESSED_DATA_DIR) / "features_wrist.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    parameters = {
        "window_sec": 60.0,
        "overlap": 0.5,
        "purity": 0.9,
        "subjects": sorted(VALID_SUBJECTS),
    }
    if (
        frame.attrs.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or frame.attrs.get("dataset_parameters") != parameters
    ):
        return None
    columns = [
        column
        for column in frame
        if column not in {"subject_id", "window_id", "label", "label_name"}
    ]
    if columns != FeatureExtractionPipeline().get_feature_names() or not {
        "subject_id",
        "window_id",
        "label",
    } <= set(frame.columns):
        raise ValueError("Cached wrist feature columns do not match the current schema")
    return frame
