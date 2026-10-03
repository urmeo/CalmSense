from typing import Dict

import numpy as np

from ..config import FEATURE_PARAMS
from ..preprocessing.filters import _positive_integer
from .hrv_base import BaseHRVExtractor


class HRVTimeDomainExtractor(BaseHRVExtractor):
    def __init__(self, min_rr_count: int = 10):
        self.min_rr_count = _positive_integer(min_rr_count, "min_rr_count")
        self.logger.debug(f"HRVTimeDomainExtractor initialized, min_rr={min_rr_count}")

    def compute_mean_nn(self, rr: np.ndarray) -> float:
        return float(np.mean(rr))

    def compute_sdnn(self, rr: np.ndarray) -> float:
        return float(np.std(rr, ddof=1))

    def compute_rmssd(self, rr: np.ndarray) -> float:
        if len(rr) < 2:
            return np.nan
        diff = np.diff(rr)
        return float(np.sqrt(np.mean(diff**2)))

    def compute_pnn50(self, rr: np.ndarray) -> float:
        if len(rr) < 2:
            return np.nan
        diff = np.abs(np.diff(rr))
        nn50 = np.sum(diff > 50)
        return float(100.0 * nn50 / len(diff))

    def compute_pnn20(self, rr: np.ndarray) -> float:
        if len(rr) < 2:
            return np.nan
        diff = np.abs(np.diff(rr))
        nn20 = np.sum(diff > 20)
        return float(100.0 * nn20 / len(diff))

    def compute_median_nn(self, rr: np.ndarray) -> float:
        return float(np.median(rr))

    def compute_cvnn(self, rr: np.ndarray) -> float:
        mean_nn = np.mean(rr)
        if mean_nn < FEATURE_PARAMS.EPSILON:
            return np.nan
        return float(np.std(rr, ddof=1) / mean_nn)

    def compute_cvsd(self, rr: np.ndarray) -> float:
        mean_nn = np.mean(rr)
        if mean_nn < FEATURE_PARAMS.EPSILON:
            return np.nan
        rmssd = self.compute_rmssd(rr)
        return float(rmssd / mean_nn)

    def compute_madnn(self, rr: np.ndarray) -> float:
        median_nn = np.median(rr)
        return float(np.median(np.abs(rr - median_nn)))

    def compute_mcvnn(self, rr: np.ndarray) -> float:
        median_nn = np.median(rr)
        if median_nn < FEATURE_PARAMS.EPSILON:
            return np.nan
        madnn = self.compute_madnn(rr)
        return float(madnn / median_nn)

    def compute_iqrnn(self, rr: np.ndarray) -> float:
        q75, q25 = np.percentile(rr, [75, 25])
        return float(q75 - q25)

    def compute_hrvti(self, rr: np.ndarray, bin_width: float = 7.8125) -> float:
        if not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError("bin_width must be finite and positive")
        rr = np.asarray(rr, dtype=float).flatten()
        if not len(rr) or not np.isfinite(rr).all():
            return np.nan
        # Fixed 1/128-second bins anchored at zero match the stated histogram convention.
        _, counts = np.unique(np.floor(rr / bin_width), return_counts=True)
        return float(len(rr) / counts.max())

    def extract_all(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        features = dict.fromkeys(self.get_feature_descriptions(), np.nan)

        rr = self._validate_input(rr_intervals)
        if rr is None:
            self.logger.warning("Invalid input, returning NaN features")
            return features

        try:
            features["MeanNN"] = self.compute_mean_nn(rr)
            features["SDNN"] = self.compute_sdnn(rr)
            features["RMSSD"] = self.compute_rmssd(rr)
            features["pNN50"] = self.compute_pnn50(rr)
            features["pNN20"] = self.compute_pnn20(rr)
            features["MedianNN"] = self.compute_median_nn(rr)
            features["CVNN"] = self.compute_cvnn(rr)
            features["CVSD"] = self.compute_cvsd(rr)
            features["MadNN"] = self.compute_madnn(rr)
            features["MCVNN"] = self.compute_mcvnn(rr)
            features["IQRNN"] = self.compute_iqrnn(rr)
            features["HRVTI"] = self.compute_hrvti(rr)

            self.logger.debug(
                f"Extracted 12 time-domain features, "
                f"MeanNN={features['MeanNN']:.1f}ms, SDNN={features['SDNN']:.1f}ms"
            )
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "MeanNN": "Mean of NN intervals (ms)",
            "SDNN": "Standard deviation of NN intervals (ms)",
            "RMSSD": "Root mean square of successive differences (ms)",
            "pNN50": "Percentage of intervals differing >50ms (%)",
            "pNN20": "Percentage of intervals differing >20ms (%)",
            "MedianNN": "Median of NN intervals (ms)",
            "CVNN": "Coefficient of variation (SDNN/MeanNN)",
            "CVSD": "CV of successive differences (RMSSD/MeanNN)",
            "MadNN": "Median absolute deviation (ms)",
            "MCVNN": "MadNN/MedianNN ratio",
            "IQRNN": "Interquartile range (ms)",
            "HRVTI": "HRV Triangular Index",
        }
