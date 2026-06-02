"""
src/coupling/hypoxia_response.py
──────────────────────────────────
Hypoxia Response Monitor.

Physiological background:
  When SpO₂ falls, peripheral chemoreceptors (carotid bodies) trigger a
  sympathetic response that increases heart rate and respiratory rate.
  A blunted hypoxic HR response indicates autonomic dysfunction or sedation.

This module:
  1. Detects SpO₂ falling episodes (>2% drop from baseline)
  2. Measures the corresponding HR change
  3. Compares observed vs expected HR response
  4. Flags blunted response if HR rise < threshold per % SpO₂ drop
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from loguru import logger

from ..core.config import TwinConfig


class HypoxiaResponseMonitor:

    _MIN_SPO2_DROP = 2.0      # % — minimum SpO₂ drop to trigger response analysis
    _RESPONSE_WINDOW_S = 60   # seconds to measure HR response after SpO₂ drop

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self._expected_hr_rise = cfg.alerts.hypoxia_response.expected_hr_rise_per_spo2_fall

    def evaluate(
        self,
        hr_series: List[float],    # 1 Hz trend
        spo2_series: List[float],  # 1 Hz trend
    ) -> Tuple[float, bool]:
        """
        Compute observed HR response to SpO₂ drops.

        Returns
        -------
        (observed_hr_rise_per_pct_spo2_drop, blunted_response_flag)
        """
        if len(hr_series) < 30 or len(spo2_series) < 30:
            return 0.0, False

        hr   = np.array(hr_series[-300:],   dtype=float)   # Up to 5 min
        spo2 = np.array(spo2_series[-300:], dtype=float)

        n = min(len(hr), len(spo2))
        hr   = hr[-n:]
        spo2 = spo2[-n:]

        # Find SpO₂ drop episodes
        spo2_smooth = self._smooth(spo2, window=5)
        spo2_delta  = np.diff(spo2_smooth, prepend=spo2_smooth[0])

        responses = []

        i = 0
        while i < len(spo2_delta) - self._RESPONSE_WINDOW_S:
            # Detect onset of SpO₂ drop
            window = spo2_delta[i:i + 10]
            cumulative_drop = -np.sum(window[window < 0])

            if cumulative_drop >= self._MIN_SPO2_DROP:
                # Measure HR change over response window
                resp_end = min(i + self._RESPONSE_WINDOW_S, len(hr) - 1)
                hr_before = float(np.mean(hr[max(0, i-5):i+1]))
                hr_after  = float(np.mean(hr[i:resp_end+1]))
                hr_rise   = hr_after - hr_before

                if cumulative_drop > 0:
                    ratio = hr_rise / cumulative_drop
                    responses.append(ratio)

                i += self._RESPONSE_WINDOW_S   # Skip past this episode
            else:
                i += 1

        if not responses:
            return 0.0, False

        mean_response = float(np.mean(responses))
        blunted = mean_response < self._expected_hr_rise * 0.5   # <50% of expected

        return round(mean_response, 2), blunted

    @staticmethod
    def _smooth(x: np.ndarray, window: int) -> np.ndarray:
        if len(x) < window:
            return x.copy()
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="same")