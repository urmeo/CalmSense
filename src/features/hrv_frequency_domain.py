from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from ..preprocessing.filters import _positive_integer, _positive_number
from .hrv_base import BaseHRVExtractor

# numpy>=2 renamed trapz to trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class HRVFrequencyDomainExtractor(BaseHRVExtractor):
    # Task Force 1996 bands
    VLF_BAND = (0.0033, 0.04)
    LF_BAND = (0.04, 0.15)
    HF_BAND = (0.15, 0.40)

    def __init__(
        self,
        vlf_band: Tuple[float, float] = VLF_BAND,
        lf_band: Tuple[float, float] = LF_BAND,
        hf_band: Tuple[float, float] = HF_BAND,
        interpolation_rate: float = 4.0,
        min_rr_count: int = 30,
    ):
        self.vlf_band = vlf_band
        self.lf_band = lf_band
        self.hf_band = hf_band
        self.interpolation_rate = _positive_number(interpolation_rate, "interpolation_rate")
        self.min_rr_count = _positive_integer(min_rr_count, "min_rr_count")
        for band in (vlf_band, lf_band, hf_band):
            if len(band) != 2 or not 0 <= band[0] < band[1] < self.interpolation_rate / 2:
                raise ValueError(
                    "HRV frequency bands must be ordered below the interpolation Nyquist"
                )

        self.logger.debug(
            f"HRVFrequencyDomainExtractor initialized: "
            f"LF={lf_band}, HF={hf_band}, fs={interpolation_rate} Hz"
        )

    def _interpolate_rr(
        self, rr: np.ndarray, method: str = "cubic"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        t_rr = np.cumsum(rr) / 1000.0
        t_rr = np.insert(t_rr, 0, 0)
        t_rr = t_rr[:-1]

        duration = t_rr[-1] - t_rr[0]
        if duration * self.interpolation_rate < 10:
            return None, None

        # Welch's declared sampling rate must match the actual interpolation step.
        t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / self.interpolation_rate)

        try:
            interpolator = interp1d(
                t_rr, rr, kind=method, fill_value="extrapolate", bounds_error=False
            )
            rr_interpolated = interpolator(t_uniform)
        except Exception as e:
            self.logger.warning(f"Interpolation failed: {e}")
            return None, None

        return t_uniform, rr_interpolated

    def compute_psd(
        self, rr_intervals: np.ndarray, method: str = "welch"
    ) -> Tuple[np.ndarray, np.ndarray]:
        if method not in {"welch", "lomb"}:
            raise ValueError("PSD method must be 'welch' or 'lomb'")
        rr = self._validate_input(rr_intervals)
        if rr is None:
            return np.array([]), np.array([])

        if method == "lomb":
            return self._compute_psd_lomb(rr)
        else:
            return self._compute_psd_welch(rr)

    def _compute_psd_welch(self, rr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t_uniform, rr_interp = self._interpolate_rr(rr)
        if rr_interp is None:
            return np.array([]), np.array([])

        rr_detrend = rr_interp - np.mean(rr_interp)

        nperseg = min(256, len(rr_detrend) // 2)
        if nperseg < 16:
            return np.array([]), np.array([])

        freqs, psd = scipy_signal.welch(
            rr_detrend,
            fs=self.interpolation_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hann",
            scaling="density",
        )

        psd = np.maximum(psd, 0)

        return freqs, psd

    def _compute_psd_lomb(self, rr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.signal import lombscargle

        t = np.r_[0.0, np.cumsum(rr[:-1]) / 1000.0]

        rr_centered = rr - np.mean(rr)

        f_max = 0.5
        n_freqs = 512
        freqs = np.linspace(0.001, f_max, n_freqs)

        variance = float(np.var(rr_centered))
        if variance == 0:
            return freqs, np.zeros_like(freqs)

        angular_freqs = 2 * np.pi * freqs

        try:
            psd = np.maximum(lombscargle(t, rr_centered, angular_freqs, normalize=False), 0.0)
            area = float(_trapz(psd, freqs))
            if not np.isfinite(area) or area <= 0:
                return np.array([]), np.array([])
            # Match RR variance over this returned frequency grid; this bounded
            # normalization is a convention rather than Welch's full density.
            psd *= variance / area
        except Exception as e:
            self.logger.warning(f"Lomb-Scargle failed: {e}")
            return np.array([]), np.array([])

        return freqs, psd

    def _compute_band_power(
        self, freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]
    ) -> float:
        if len(freqs) == 0 or len(psd) == 0:
            return np.nan

        lo, hi = max(band[0], freqs[0]), min(band[1], freqs[-1])
        if lo >= hi:
            return 0.0
        inside = freqs[(freqs > lo) & (freqs < hi)]
        band_freqs = np.r_[lo, inside, hi]
        # Include interpolated endpoints so narrow bands do not collapse to zero bins.
        return float(_trapz(np.interp(band_freqs, freqs, psd), band_freqs))

    def compute_vlf_power(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        return self._compute_band_power(freqs, psd, self.vlf_band)

    def compute_lf_power(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        return self._compute_band_power(freqs, psd, self.lf_band)

    def compute_hf_power(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        return self._compute_band_power(freqs, psd, self.hf_band)

    def compute_total_power(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        vlf = self.compute_vlf_power(freqs, psd)
        lf = self.compute_lf_power(freqs, psd)
        hf = self.compute_hf_power(freqs, psd)

        powers = [vlf, lf, hf]
        valid_powers = [p for p in powers if np.isfinite(p)]

        if len(valid_powers) == 0:
            return np.nan

        return float(sum(valid_powers))

    def compute_lf_hf_ratio(self, lf_power: float, hf_power: float) -> float:
        if not np.isfinite(lf_power) or not np.isfinite(hf_power):
            return np.nan
        if hf_power < 1e-6:
            return np.nan
        return float(lf_power / hf_power)

    def compute_lfn(self, lf_power: float, hf_power: float) -> float:
        if not np.isfinite(lf_power) or not np.isfinite(hf_power):
            return np.nan
        total = lf_power + hf_power
        if total < 1e-6:
            return np.nan
        return float(100.0 * lf_power / total)

    def compute_hfn(self, lf_power: float, hf_power: float) -> float:
        if not np.isfinite(lf_power) or not np.isfinite(hf_power):
            return np.nan
        total = lf_power + hf_power
        if total < 1e-6:
            return np.nan
        return float(100.0 * hf_power / total)

    def compute_lf_peak_freq(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        if len(freqs) == 0 or len(psd) == 0:
            return np.nan

        mask = (freqs >= self.lf_band[0]) & (freqs <= self.lf_band[1])
        if not np.any(mask):
            return np.nan

        lf_freqs = freqs[mask]
        lf_psd = psd[mask]

        if not np.isfinite(lf_psd).all() or np.max(lf_psd) <= 0:
            return np.nan

        peak_idx = np.argmax(lf_psd)
        return float(lf_freqs[peak_idx])

    def extract_all(self, rr_intervals: np.ndarray, method: str = "welch") -> Dict[str, float]:
        features = dict.fromkeys(self.get_feature_descriptions(), np.nan)

        freqs, psd = self.compute_psd(rr_intervals, method=method)

        if len(freqs) == 0:
            self.logger.warning("PSD computation failed, returning NaN features")
            return features

        try:
            features["VLF_power"] = self.compute_vlf_power(freqs, psd)
            features["LF_power"] = self.compute_lf_power(freqs, psd)
            features["HF_power"] = self.compute_hf_power(freqs, psd)
            features["Total_power"] = self.compute_total_power(freqs, psd)

            features["LF_HF_ratio"] = self.compute_lf_hf_ratio(
                features["LF_power"], features["HF_power"]
            )
            features["LFn"] = self.compute_lfn(features["LF_power"], features["HF_power"])
            features["HFn"] = self.compute_hfn(features["LF_power"], features["HF_power"])
            features["LF_peak_freq"] = self.compute_lf_peak_freq(freqs, psd)

            if np.isfinite(features["LF_HF_ratio"]):
                self.logger.debug(
                    f"Extracted 8 frequency-domain features, LF/HF={features['LF_HF_ratio']:.2f}"
                )
            else:
                self.logger.debug("Extracted 8 frequency-domain features")
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "VLF_power": f"VLF power {self.vlf_band} Hz (ms²)",
            "LF_power": f"LF power {self.lf_band} Hz (ms²)",
            "HF_power": f"HF power {self.hf_band} Hz (ms²)",
            "Total_power": "Total power VLF+LF+HF (ms²)",
            "LF_HF_ratio": "Ratio of LF to HF spectral power",
            "LFn": "Normalized LF power (%)",
            "HFn": "Normalized HF power (%)",
            "LF_peak_freq": "Peak frequency in LF band (Hz)",
        }
