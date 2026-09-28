from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import FS
from ..logging_config import LoggerMixin
from ..utils import timer
from .accelerometer_features import AccelerometerFeatureExtractor
from .eda_features import EDAFeatureExtractor
from .hrv_frequency_domain import HRVFrequencyDomainExtractor
from .hrv_nonlinear import HRVNonlinearExtractor
from .hrv_time_domain import HRVTimeDomainExtractor
from .respiration_features import RespirationFeatureExtractor
from .temperature_features import TemperatureFeatureExtractor


class FeatureExtractionPipeline(LoggerMixin):
    """Turn one preprocessed window into a flat, prefixed feature dict.

    Composes seven per-modality extractors (HRV time/frequency/nonlinear, EDA,
    temperature, respiration, accelerometer) and namespaces their outputs with
    ``HRV_``/``EDA_``/``TEMP_``/``RESP_``/``ACC_`` prefixes. Groups can be toggled
    via ``feature_config``; disabled groups are omitted and unavailable groups yield
    NaN placeholders so columns stay stable across windows. Extraction never sees labels.
    """

    DEFAULT_CONFIG = {
        "hrv_time": True,
        "hrv_frequency": True,
        "hrv_nonlinear": True,
        "eda": True,
        "temperature": True,
        "respiration": True,
        "accelerometer": True,
    }

    def __init__(
        self,
        feature_config: Optional[Dict[str, bool]] = None,
        chest_fs: float = FS.CHEST,
        wrist_eda_fs: float = FS.WRIST_EDA,
        wrist_acc_fs: float = FS.WRIST_ACC,
    ):
        if feature_config:
            unknown = set(feature_config) - set(self.DEFAULT_CONFIG)
            if unknown:
                raise ValueError(
                    f"Unknown feature_config keys: {sorted(unknown)}. "
                    f"Valid keys: {sorted(self.DEFAULT_CONFIG)}"
                )
        self.feature_config = {**self.DEFAULT_CONFIG, **(feature_config or {})}

        self.extractors: Dict[str, Any] = {
            "hrv_time": HRVTimeDomainExtractor(),
            "hrv_frequency": HRVFrequencyDomainExtractor(),
            "hrv_nonlinear": HRVNonlinearExtractor(),
            "eda": EDAFeatureExtractor(sampling_rate=wrist_eda_fs),
            "temperature": TemperatureFeatureExtractor(
                sampling_rate=wrist_eda_fs
            ),  # wrist TEMP and EDA share 4 Hz
            "respiration": RespirationFeatureExtractor(sampling_rate=chest_fs),
            "accelerometer": AccelerometerFeatureExtractor(sampling_rate=wrist_acc_fs),
        }

        self.logger.info(
            f"FeatureExtractionPipeline initialized with config: "
            f"{sum(self.feature_config.values())} feature groups enabled"
        )

    def extract_window_features(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all enabled features for a single window.

        Args:
            window_data: One window's signals (``rr_intervals``, ``eda_tonic``,
                ``eda_phasic``, ``temperature``, ``respiration``, ``accelerometer``,
                ...). Missing modalities produce NaN placeholders.

        Returns:
            Prefixed feature-name to value mapping for this window.
        """
        rr = window_data.get("rr_intervals")

        features: Dict[str, float] = {}
        self._add(features, "hrv_time", "HRV_", lambda: self._hrv(rr, "hrv_time"))
        self._add(features, "hrv_frequency", "HRV_", lambda: self._hrv(rr, "hrv_frequency"))
        self._add(features, "hrv_nonlinear", "HRV_", lambda: self._hrv(rr, "hrv_nonlinear"))
        self._add(features, "eda", "EDA_", lambda: self._eda(window_data))
        self._add(features, "temperature", "TEMP_", lambda: self._temperature(window_data))
        self._add(features, "respiration", "RESP_", lambda: self._respiration(window_data))
        self._add(features, "accelerometer", "ACC_", lambda: self._accelerometer(window_data))
        return features

    def _add(self, features, group, prefix, compute) -> None:
        if not self.feature_config.get(group, True):
            return
        result = compute()
        if result is None:
            result = dict.fromkeys(self.extractors[group].get_feature_descriptions(), np.nan)
        features.update({k if k.startswith(prefix) else prefix + k: v for k, v in result.items()})

    def _hrv(self, rr, group):
        return None if rr is None else self.extractors[group].extract_all(rr)

    def _eda(self, w):
        tonic, raw = w.get("eda_tonic"), w.get("eda_raw")
        if tonic is None and raw is None:
            return None
        decomposed = {"tonic": tonic, "phasic": w.get("eda_phasic")}
        return self.extractors["eda"].extract_all(decomposed, w.get("scr_peaks"), raw)

    def _temperature(self, w):
        temp = w.get("temperature")
        return None if temp is None else self.extractors["temperature"].extract_all(temp)

    def _respiration(self, w):
        resp = w.get("respiration")
        if resp is None:
            return None
        return self.extractors["respiration"].extract_all(
            resp,
            breath_peaks=w.get("breath_peaks"),
            breath_troughs=w.get("breath_troughs"),
            breath_intervals=w.get("breath_intervals"),
        )

    def _accelerometer(self, w):
        acc = w.get("accelerometer")
        if acc is None:
            return None
        extractor = self.extractors["accelerometer"]
        if not isinstance(acc, dict):
            return extractor.extract_from_magnitude(acc)
        if "magnitude" in acc:
            return extractor.extract_from_magnitude(acc["magnitude"])
        empty = np.array([])
        return extractor.extract_all(acc.get("x", empty), acc.get("y", empty), acc.get("z", empty))

    def extract_all_features(
        self,
        processed_data: Union[pd.DataFrame, List[Dict]],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if isinstance(processed_data, pd.DataFrame):
            windows = processed_data.to_dict("records")
        else:
            windows = processed_data

        n_windows = len(windows)
        self.logger.info(f"Extracting features from {n_windows} windows")

        all_features = []

        with timer("Feature extraction"):
            for i, window_data in enumerate(windows):
                if show_progress and (i + 1) % 100 == 0:
                    self.logger.info(f"Processing window {i + 1}/{n_windows}")

                features = self.extract_window_features(window_data)

                for key in ["subject_id", "window_id", "label"]:
                    if key in window_data:
                        features[key] = window_data[key]

                all_features.append(features)

        features_df = pd.DataFrame(all_features)

        metadata_cols = ["subject_id", "window_id", "label"]
        existing_meta = [c for c in metadata_cols if c in features_df.columns]
        feature_cols = [c for c in features_df.columns if c not in metadata_cols]
        features_df = features_df[existing_meta + feature_cols]

        self.logger.info(
            f"Feature extraction complete: {len(features_df)} windows, {len(feature_cols)} features"
        )

        return features_df
