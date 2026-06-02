"""
src/coupling/cross_correlation.py
───────────────────────────────────
Cross-correlation between instantaneous heart rate and respiratory rate
trend series. Quantifies the temporal coupling and lag between the two
oscillators over a sliding window.

Returns:
  - Peak cross-correlation coefficient (−1 to 1)
  - Lag at peak (seconds)
  - Full xcorr series for dashboard display
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from typing import List, Tuple
from loguru import logger

from ..core.config import TwinConfig


class HRRRCrossCorrelation:

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self._win_samples = int(cfg.coupling.cross_correlation.window_s)
        self._max_lag_s   = cfg.coupling.cross_correlation.max_lag_s

    def compute(
        self,
        hr_series: List[float],
        rr_series: List[float],   # respiratory rate, 1 Hz
    ) -> Tuple[float, float, List[float]]:
        """
        Parameters
        ----------
        hr_series   : instantaneous HR (bpm), 1-Hz trend series
        rr_series   : respiratory rate (br/min), 1-Hz trend series

        Returns
        -------
        (peak_xcorr, lag_seconds, full_xcorr_series)
        """
        null = (0.0, 0.0, [])

        n_needed = max(self._win_samples, 30)
        if len(hr_series) < n_needed or len(rr_series) < n_needed:
            return null

        # Take most recent window
        hr  = np.array(hr_series[-self._win_samples:], dtype=float)
        rr_ = np.array(rr_series[-self._win_samples:], dtype=float)

        # Z-score normalise
        hr_z  = self._znorm(hr)
        rr_z_ = self._znorm(rr_)

        # Compute normalised cross-correlation
        xcorr = sp_signal.correlate(hr_z, rr_z_, mode="full") / len(hr_z)
        lags  = sp_signal.correlation_lags(len(hr_z), len(rr_z_), mode="full")

        # Restrict to ±max_lag_s
        max_lag_samples = int(self._max_lag_s)   # 1 Hz series → 1 sample = 1 s
        valid_mask = np.abs(lags) <= max_lag_samples
        xcorr_valid = xcorr[valid_mask]
        lags_valid  = lags[valid_mask]

        # Find peak
        peak_idx   = int(np.argmax(np.abs(xcorr_valid)))
        peak_xcorr = float(xcorr_valid[peak_idx])
        peak_lag_s = float(lags_valid[peak_idx])

        # Return truncated series for display (last 200 points of valid range)
        display_series = list(xcorr_valid[-200:])

        return round(peak_xcorr, 3), round(peak_lag_s, 1), display_series

    @staticmethod
    def _znorm(x: np.ndarray) -> np.ndarray:
        std = np.std(x)
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - np.mean(x)) / std