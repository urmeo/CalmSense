"""Window WESAD chest signals into features and raw CNN tensors."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import resample

from .config import FEATURE_PARAMS, FS, PROCESSED_DATA_DIR, VALID_SUBJECTS
from .data.loader import WESADLoader
from .features.feature_pipeline import FEATURE_SCHEMA_VERSION, FeatureExtractionPipeline
from .logging_config import LoggerMixin
from .preprocessing.ecg_processor import ECGProcessor
from .preprocessing.eda_processor import EDAProcessor
from .preprocessing.filters import SignalProcessor, _positive_integer, _window_parameters

# Conditions kept for classification
CONDITION_LABELS = {1: "baseline", 2: "stress", 3: "amusement"}
CNN_CHANNELS = ["ECG", "EDA", "Temp", "Resp", "ACC"]


def window_label(labels: np.ndarray, purity: float) -> Optional[int]:
    """Dominant condition of a window, or None if out-of-set or below `purity`."""
    if not np.isfinite(purity) or not 0 < purity <= 1:
        raise ValueError("purity must be finite and in (0, 1]")
    labels = np.asarray(labels).flatten()
    if not len(labels):
        return None
    values, counts = np.unique(labels, return_counts=True)
    dominant = values[counts.argmax()]
    if dominant not in CONDITION_LABELS:
        return None
    if counts.max() / counts.sum() < purity:
        return None
    return int(dominant)


class WindowedDataset(LoggerMixin):
    """Build chest features and aligned CNN tensors from each subject's signals.

    Filter each recording before windowing. Keep baseline, stress, or amusement
    windows meeting ``purity`` and preserve subject IDs for grouped evaluation.
    """

    def __init__(
        self,
        window_sec: float = FEATURE_PARAMS.WINDOW_SIZE_SEC,
        overlap: float = FEATURE_PARAMS.WINDOW_OVERLAP,
        purity: float = 0.9,
        cnn_length: int = 1024,
        fs: float = FS.CHEST,
        data_path: Optional[Union[str, Path]] = None,
    ):
        self.fs, self.window_samples, self.step = _window_parameters(
            window_sec, overlap, purity, fs
        )
        self.purity = float(purity)
        self.cnn_length = _positive_integer(cnn_length, "cnn_length")

        self.loader = WESADLoader(data_path=data_path)
        self.ecg = ECGProcessor(sampling_rate=fs)
        self.eda = EDAProcessor(sampling_rate=fs)
        self.sig = SignalProcessor(fs=fs)
        # All chest modalities share 700 Hz
        self.features = FeatureExtractionPipeline(chest_fs=fs, wrist_eda_fs=fs, wrist_acc_fs=fs)

    def _window_label(self, labels: np.ndarray) -> Optional[int]:
        return window_label(labels, self.purity)

    def _process_subject(self, subject_id: str) -> Tuple[List[Dict], List[np.ndarray], List[int]]:
        data = self.loader.load_subject(subject_id)
        chest = data["chest"]
        labels = np.asarray(data["label"]).flatten()

        ecg_filt = self.ecg.bandpass_filter(chest["ECG"])
        r_peaks = self.ecg.detect_r_peaks(ecg_filt)

        eda_filt = self.eda.remove_artifacts(self.eda.lowpass_filter(chest["EDA"]))
        tonic, phasic = self.eda.decompose_eda(eda_filt)
        _, scr_features = self.eda.detect_scr_peaks(phasic)
        scr_idx = np.array([s["peak_idx"] for s in scr_features])

        temp_filt = self.sig.process_temperature(chest["Temp"])
        resp_filt = self.sig.process_respiration(chest["Resp"])
        acc_mag = np.sqrt(np.sum(np.asarray(chest["ACC"]) ** 2, axis=1))

        n = min(len(ecg_filt), len(labels))
        windows, raws, ys = [], [], []

        for start in range(0, n - self.window_samples + 1, self.step):
            end = start + self.window_samples
            label = self._window_label(labels[start:end])
            if label is None:
                continue

            mask = (r_peaks >= start) & (r_peaks < end)
            # Both peaks must lie inside the window; do not include a boundary-spanning RR interval.
            rr = self.ecg.extract_rr_intervals(r_peaks[mask], unit="ms")
            _, valid = self.ecg.remove_ectopic_beats(rr)
            rr_clean = self.ecg.interpolate_artifacts(rr, valid)

            scr_in = [s for s, idx in zip(scr_features, scr_idx) if start <= idx < end]

            window = {
                "rr_intervals": rr_clean,
                "eda_tonic": tonic[start:end],
                "eda_phasic": phasic[start:end],
                "eda_raw": eda_filt[start:end],
                "scr_peaks": scr_in,
                "temperature": temp_filt[start:end],
                "respiration": resp_filt[start:end],
                "accelerometer": {"magnitude": acc_mag[start:end]},
                "subject_id": subject_id,
                "window_id": start,
                "label": label,
            }
            windows.append(window)
            raws.append(
                self._raw_tensor(
                    ecg_filt[start:end],
                    eda_filt[start:end],
                    temp_filt[start:end],
                    resp_filt[start:end],
                    acc_mag[start:end],
                )
            )
            ys.append(label)

        self.logger.info(f"{subject_id}: {len(ys)} windows")
        return windows, raws, ys

    def _raw_tensor(self, *channels: np.ndarray) -> np.ndarray:
        """Resample the same window across channels into a channels-by-time CNN input."""
        stacked = [resample(np.asarray(c, dtype=np.float32), self.cnn_length) for c in channels]
        return np.stack(stacked).astype(np.float32)

    def build(
        self, subjects: Optional[List[str]] = None, cache: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        subjects = self.loader.subjects if subjects is None else subjects
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("subjects must be a nonempty list without duplicates")

        all_windows, all_raw, all_y = [], [], []
        for subject_id in subjects:
            windows, raws, ys = self._process_subject(subject_id)
            all_windows.extend(windows)
            all_raw.extend(raws)
            all_y.extend(ys)

        features_df = self.features.extract_all_features(all_windows, show_progress=True)
        features_df["label_name"] = features_df["label"].map(CONDITION_LABELS)
        features_df.attrs["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        features_df.attrs["dataset_parameters"] = {
            "fs": self.fs,
            "window_samples": self.window_samples,
            "step": self.step,
            "purity": self.purity,
            "cnn_length": self.cnn_length,
            "subjects": sorted(subjects),
        }
        x_raw = np.stack(all_raw) if all_raw else np.empty((0, len(CNN_CHANNELS), self.cnn_length))

        if cache:
            self._save(features_df, x_raw)
        return features_df, x_raw, np.asarray(all_y)

    def _save(self, features_df: pd.DataFrame, x_raw: np.ndarray) -> None:
        # Feature rows and raw tensors use the same ordering for task filtering.
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(PROCESSED_DATA_DIR / "features.parquet", index=False)
        np.savez_compressed(
            PROCESSED_DATA_DIR / "raw_windows.npz",
            x=x_raw,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            subject=features_df["subject_id"].to_numpy(dtype=str),
            label=features_df["label"].to_numpy(dtype=int),
            window_id=features_df["window_id"].to_numpy(dtype=int),
        )
        self.logger.info(f"Cached dataset to {PROCESSED_DATA_DIR}")


def load_cached() -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    feat_path = PROCESSED_DATA_DIR / "features.parquet"
    raw_path = PROCESSED_DATA_DIR / "raw_windows.npz"
    if not feat_path.exists() or not raw_path.exists():
        return None
    features_df = pd.read_parquet(feat_path)
    default_parameters = {
        "fs": FS.CHEST,
        "window_samples": int(FEATURE_PARAMS.WINDOW_SIZE_SEC * FS.CHEST),
        "step": int(
            FEATURE_PARAMS.WINDOW_SIZE_SEC * FS.CHEST * (1 - FEATURE_PARAMS.WINDOW_OVERLAP)
        ),
        "purity": 0.9,
        "cnn_length": 1024,
        "subjects": sorted(VALID_SUBJECTS),
    }
    if (
        features_df.attrs.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or features_df.attrs.get("dataset_parameters") != default_parameters
    ):
        return None
    metadata = {"subject_id", "window_id", "label", "label_name"}
    columns = [column for column in features_df if column not in metadata]
    if columns != FeatureExtractionPipeline().get_feature_names() or not all(
        pd.api.types.is_numeric_dtype(features_df[column]) for column in columns
    ):
        raise ValueError("Cached feature columns do not match the current schema")
    with np.load(raw_path, allow_pickle=False) as raw:
        if (
            "feature_schema_version" not in raw
            or raw["feature_schema_version"].item() != FEATURE_SCHEMA_VERSION
        ):
            return None
        x_raw = raw["x"]
        aligned = (
            x_raw.shape == (len(features_df), len(CNN_CHANNELS), 1024)
            and np.isfinite(x_raw).all()
            and all(
                column in features_df
                and key in raw
                and np.array_equal(features_df[column].to_numpy(), raw[key])
                for column, key in (
                    ("subject_id", "subject"),
                    ("label", "label"),
                    ("window_id", "window_id"),
                )
            )
        )
        if not aligned:
            raise ValueError("Cached feature rows and raw CNN tensors are not aligned")
    return features_df, x_raw
