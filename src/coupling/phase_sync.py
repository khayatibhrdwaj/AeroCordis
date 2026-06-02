"""
src/coupling/phase_sync.py
───────────────────────────
Phase Synchronisation Index (PSI) between cardiac and respiratory oscillators.

Method (Hilbert transform approach — Tass et al. 1998):
  1. Extract instantaneous phase of ECG (via R-peak phase interpolation)
     and respiratory signal (via Hilbert transform)
  2. Compute relative phase: Δφ = φ_cardiac − n·φ_resp
     where n:m is the most common integer ratio (typically 4:1 or 5:1)
  3. PSI = |mean(exp(i·Δφ))| — circular mean of relative phase
     PSI = 1 → perfect phase locking
     PSI = 0 → no synchronisation

Also returns the instantaneous phase difference in radians and the n:m locking ratio.
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from typing import Tuple
from loguru import logger

from src.core.config import TwinConfig


class PhaseSynchronisation:

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self._win_s   = cfg.coupling.phase_sync.window_s
        self._n_bins  = cfg.coupling.phase_sync.n_bins
        self._method  = cfg.coupling.phase_sync.method

    def compute(
        self,
        ecg_buffer: np.ndarray,
        resp_buffer: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute Phase Synchronisation Index, instantaneous phase difference, and n:m ratio.

        Parameters
        ----------
        ecg_buffer  : raw/filtered ECG waveform (at ecg.sampling_rate)
        resp_buffer : filtered respiratory waveform (at respiratory.sampling_rate)

        Returns
        -------
        (psi, phase_difference_rad, n_ratio)
        """
        if len(ecg_buffer) < 32 or len(resp_buffer) < 32:
            return 0.0, 0.0, 0.0

        # Resample both to common 10 Hz grid for phase analysis
        target_fs = 10.0
        ecg_ds  = self._resample(ecg_buffer, self.cfg.ecg.sampling_rate,  target_fs)
        resp_ds = self._resample(resp_buffer, self.cfg.respiratory.sampling_rate, target_fs)

        # Truncate to same length
        n = min(len(ecg_ds), len(resp_ds))
        ecg_ds  = ecg_ds[-n:]
        resp_ds = resp_ds[-n:]

        if n < 20:
            return 0.0, 0.0, 0.0

        # Extract instantaneous phases via Hilbert transform
        try:
            # Detrend to remove DC offset and low-frequency drift
            ecg_detrended = sp_signal.detrend(ecg_ds)
            resp_detrended = sp_signal.detrend(resp_ds)
            
            # Hilbert transform to get analytic signal
            phi_ecg  = np.unwrap(np.angle(sp_signal.hilbert(ecg_detrended))) # type: ignore
            phi_resp = np.unwrap(np.angle(sp_signal.hilbert(resp_detrended))) # type: ignore
        except Exception as e:
            logger.debug(f"Hilbert error: {e}")
            return 0.0, 0.0, 0.0

        # Estimate n:m ratio (cardiac cycles per respiratory cycle)
        # Use ratio of mean instantaneous frequencies (phase velocities)
        # f = d(phi)/dt / (2*pi)
        # Ratio = (phi_ecg_end - phi_ecg_start) / (phi_resp_end - phi_resp_start)
        d_phi_ecg = phi_ecg[-1] - phi_ecg[0]
        d_phi_resp = phi_resp[-1] - phi_resp[0]
        
        if d_phi_resp > 0:
            n_ratio = d_phi_ecg / d_phi_resp
        else:
            n_ratio = self.cfg.coupling.phase_sync.n_ratio_default
            
        # For PSI calculation, we typically use the nearest integer ratio
        n_round = max(1, round(n_ratio))

        # Generalised phase difference: Δφ = φ_cardiac − n·φ_resp
        delta_phi = phi_ecg - n_round * phi_resp

        # PSI = circular mean vector length
        # PSI = |1/N * sum(exp(i * delta_phi))|
        psi = float(np.abs(np.mean(np.exp(1j * delta_phi))))
        psi = float(np.clip(psi, 0, 1))

        # Instantaneous phase difference (last value, wrapped to [−π, π])
        last_diff = float(((delta_phi[-1] + np.pi) % (2 * np.pi)) - np.pi)

        return round(psi, 3), round(last_diff, 4), round(n_ratio, 2)

    @staticmethod
    def _resample(sig: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
        if orig_fs == target_fs:
            return sig
        ratio = target_fs / orig_fs
        n_out = max(1, int(len(sig) * ratio))
        return np.asarray(sp_signal.resample(sig, n_out))
