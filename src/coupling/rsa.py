"""
src/coupling/rsa.py
────────────────────
Respiratory Sinus Arrhythmia (RSA) Analyzer.

RSA is the normal variation in heart rate that occurs during the
respiratory cycle: HR increases during inspiration, decreases during
expiration. It is a key marker of cardiac vagal tone and heart-lung coupling.

Algorithm:
  1. Interpolate RR intervals onto a uniform time grid (4 Hz)
  2. Bandpass filter the RR time series around the respiratory frequency
     (f_resp ± cfg.rsa.bandpass_around_resp Hz)
  3. RSA amplitude = peak-to-trough of the filtered RR variation (in bpm)
  4. RSA coherence = squared coherence between HR and RESP at f_resp
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from typing import List, Tuple
from loguru import logger

from ..core.config import TwinConfig


class RSAAnalyzer:

    _INTERP_FS = 4.0   # Interpolation rate for RR series

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self._rr_history: List[float] = []   # ms

    def compute(
        self,
        rr_intervals: List[float],
        resp_freq_hz: float,
    ) -> Tuple[float, float]:
        """
        Compute RSA amplitude (bpm) and coherence (0–1).

        Parameters
        ----------
        rr_intervals  : list of RR intervals in ms (most recent window)
        resp_freq_hz  : dominant respiratory frequency in Hz

        Returns
        -------
        (rsa_amplitude_bpm, rsa_coherence)
        """
        if len(rr_intervals) < 8 or resp_freq_hz < 0.05:
            return 0.0, 0.0

        self._rr_history.extend(rr_intervals)
        if len(self._rr_history) > 300:
            self._rr_history = self._rr_history[-300:]

        rr = np.array(self._rr_history, dtype=float)

        # Convert to HR (bpm) series for more intuitive amplitude
        hr = 60_000.0 / rr

        # Interpolate to uniform 4 Hz grid
        t_rr = np.cumsum(rr / 1000.0)
        t_grid = np.arange(t_rr[0], t_rr[-1], 1.0 / self._INTERP_FS)
        if len(t_grid) < 8:
            return 0.0, 0.0

        hr_interp = np.interp(t_grid, t_rr, hr)
        hr_detrend = sp_signal.detrend(hr_interp)

        # Bandpass around respiratory frequency
        lo = max(0.01, resp_freq_hz - self.cfg.coupling.rsa.bandpass_around_resp)
        hi = min(self._INTERP_FS / 2 - 0.01,
                 resp_freq_hz + self.cfg.coupling.rsa.bandpass_around_resp)

        if lo >= hi or hi >= self._INTERP_FS / 2:
            return 0.0, 0.0

        try:
            nyq = self._INTERP_FS / 2
            coeff = sp_signal.butter(3, [lo / nyq, hi / nyq], btype="band", output="ba")
            b_arr: np.ndarray = np.asarray(coeff[0])  # type: ignore[index]
            a_arr: np.ndarray = np.asarray(coeff[1])  # type: ignore[index]
            hr_rsa = sp_signal.filtfilt(b_arr, a_arr, hr_detrend)
        except Exception as e:
            logger.debug(f"RSA filter error: {e}")
            return 0.0, 0.0

        # RSA amplitude: peak-to-trough / 2 (half-amplitude)
        rsa_amp = float((np.max(hr_rsa) - np.min(hr_rsa)) / 2.0)
        rsa_amp = max(0.0, rsa_amp)

        # Coherence at resp frequency
        rsa_coh = self._coherence(hr_detrend, resp_freq_hz)

        return round(rsa_amp, 2), round(rsa_coh, 3)

    def _coherence(self, hr_series: np.ndarray, resp_freq: float) -> float:
        """
        Estimate coherence at the respiratory frequency using Welch cross-spectrum.
        Requires a matching respiratory signal; here we approximate with a
        synthetic sinusoid at resp_freq for the reference.
        """
        n = len(hr_series)
        t = np.arange(n) / self._INTERP_FS
        resp_synth = np.sin(2 * np.pi * resp_freq * t)

        nperseg = min(128, n // 2)
        if nperseg < 8:
            return 0.0

        try:
            coh_result = sp_signal.coherence(
                hr_series, resp_synth,
                fs=self._INTERP_FS,
                nperseg=nperseg,
            )
            freqs: np.ndarray = np.asarray(coh_result[0])
            coh: np.ndarray = np.asarray(coh_result[1])
            # Find coherence closest to resp_freq
            idx = int(np.argmin(np.abs(freqs - resp_freq)))
            return float(np.clip(coh[idx], 0, 1))
        except Exception:
            return 0.0