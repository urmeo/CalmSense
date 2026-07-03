"""Shared scaffolding for the HRV extractors.

The three HRV extractors (time-domain, frequency-domain, nonlinear) all begin
from the same RR-interval series and reject it the same way: drop non-finite
values, require a minimum count, and clip to a plausible physiological range.
That guard lives here so the three stay in lockstep — changing the accepted RR
range in one place changes it everywhere.
"""

from typing import Optional

import numpy as np

from ..logging_config import LoggerMixin

# Plausible human RR-interval range in ms (~24-300 bpm); values outside are artifacts.
RR_MIN_MS = 200.0
RR_MAX_MS = 2500.0


class BaseHRVExtractor(LoggerMixin):
    """Common RR-interval validation. Subclasses set ``self.min_rr_count``."""

    min_rr_count: int = 10

    def _validate_input(self, rr_intervals: np.ndarray) -> Optional[np.ndarray]:
        rr = np.asarray(rr_intervals).flatten()
        rr = rr[np.isfinite(rr)]

        if len(rr) < self.min_rr_count:
            self.logger.warning(f"Insufficient RR intervals: {len(rr)} < {self.min_rr_count}")
            return None

        rr = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]

        if len(rr) < self.min_rr_count:
            self.logger.warning("Too many invalid RR intervals removed")
            return None

        return rr
