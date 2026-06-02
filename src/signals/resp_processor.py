"""
src/signals/resp_processor.py
──────────────────────────────
Respiratory signal processing:
  1. Bandpass filter (0.1–1.0 Hz)
  2. Zero-crossing / peak-based breath detection
  3. Respiratory rate (breaths/min)
  4. I:E ratio
  5. Apnea detection
  6. Dominant frequency via FFT
  7. Relative tidal volume estimate
  8. Signal quality index
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq
from typing import List, Tuple
from loguru import logger

from ..core.config import TwinConfig
from ..core.twin_state import RespiratoryFeatures


class RespProcessor:

    _HIST_S = 120   # seconds of history for rate estimation

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self.fs  = cfg.respiratory.sampling_rate

        self._buf_len = int(self._HIST_S * self.fs)
        self._buf = np.zeros(self._buf_len)
        self._buf_filled = 0

        self._bp_b, self._bp_a = self._make_bp()

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, raw: np.ndarray) -> RespiratoryFeatures:
        if len(raw) == 0:
            return RespiratoryFeatures(signal_quality=0.0)

        # Filter
        filtered = self._filter(raw)

        # Update history buffer
        n = len(filtered)
        self._buf = np.roll(self._buf, -n)
        self._buf[-n:] = filtered
        self._buf_filled = min(self._buf_filled + n, self._buf_len)

        # Use filled portion only
        active = self._buf[-self._buf_filled:]

        # Dominant frequency
        dom_freq, dom_amp = self._dominant_frequency(active)

        # Breath detection
        peaks_insp, peaks_exp = self._detect_breaths(active)

        # Respiratory rate
        rr = self._respiratory_rate(peaks_insp, dom_freq)

        # I:E ratio
        insp_dur, exp_dur, ie = self._ie_ratio(peaks_insp, peaks_exp)

        # Apnea
        apnea, apnea_dur = self._apnea(peaks_insp)

        # Tidal volume (relative)
        tv_rel = self._tidal_volume_rel(active, peaks_insp, peaks_exp)

        # Quality
        sqI = self._signal_quality(filtered, dom_amp)

        return RespiratoryFeatures(
            respiratory_rate=rr,
            tidal_volume_rel=tv_rel,
            inspiration_duration_s=insp_dur,
            expiration_duration_s=exp_dur,
            ie_ratio=ie,
            apnea_detected=apnea,
            apnea_duration_s=apnea_dur,
            dominant_frequency_hz=dom_freq,
            signal_quality=sqI,
        )

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _make_bp(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = self.cfg.respiratory.bandpass_low  / (self.fs / 2)
        hi = self.cfg.respiratory.bandpass_high / (self.fs / 2)
        lo = max(0.001, min(lo, 0.499))
        hi = max(lo + 0.001, min(hi, 0.499))
        coeff = sp_signal.butter(3, [lo, hi], btype="band", output="ba")
        return np.asarray(coeff[0]), np.asarray(coeff[1])  # type: ignore[index]

    def _filter(self, raw: np.ndarray) -> np.ndarray:
        if len(raw) < 9:
            return raw.copy()
        try:
            return sp_signal.filtfilt(self._bp_b, self._bp_a, raw)
        except Exception:
            return raw.copy()

    # ── Dominant frequency ────────────────────────────────────────────────────

    def _dominant_frequency(self, sig: np.ndarray) -> Tuple[float, float]:
        if len(sig) < int(2 * self.fs):
            return 0.25, 0.0   # Fallback ~15 br/min

        N = len(sig)
        freqs: np.ndarray = np.asarray(rfftfreq(N, d=1.0 / self.fs))
        fft_mag: np.ndarray = np.abs(np.asarray(rfft(sig - np.mean(sig))))

        lo = self.cfg.respiratory.bandpass_low
        hi = self.cfg.respiratory.bandpass_high
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            return 0.25, 0.0

        idx = np.argmax(fft_mag[mask])
        dom_freq = float(freqs[mask][idx])
        dom_amp  = float(fft_mag[mask][idx])
        return max(dom_freq, 1e-4), dom_amp

    # ── Breath detection ──────────────────────────────────────────────────────

    def _detect_breaths(
        self, sig: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect inspiration peaks and expiration troughs."""
        if len(sig) < int(self.fs):
            return np.array([]), np.array([])

        min_dist = int(self.fs * 1.5)   # Minimum 1.5 s between breaths (40 br/min max)
        height_thresh = 0.1 * np.std(sig)

        peaks_result_insp = sp_signal.find_peaks(
            sig, distance=min_dist, height=height_thresh
        )
        peaks_result_exp = sp_signal.find_peaks(
            -sig, distance=min_dist, height=height_thresh
        )
        peaks_insp = np.asarray(peaks_result_insp[0], dtype=int)
        peaks_exp  = np.asarray(peaks_result_exp[0], dtype=int)
        return peaks_insp, peaks_exp

    # ── Respiratory rate ──────────────────────────────────────────────────────

    def _respiratory_rate(
        self, peaks_insp: np.ndarray, dom_freq: float
    ) -> float:
        # Primary: peak-based
        if len(peaks_insp) >= 3:
            intervals_s = np.diff(peaks_insp) / self.fs
            # Remove outliers
            med = np.median(intervals_s)
            ok  = np.abs(intervals_s - med) < 0.5 * med
            if ok.sum() >= 2:
                rr = 60.0 / float(np.mean(intervals_s[ok]))
                if 3 < rr < 60:
                    return round(rr, 1)

        # Fallback: dominant frequency
        return round(dom_freq * 60, 1)

    # ── I:E ratio ─────────────────────────────────────────────────────────────

    def _ie_ratio(
        self, peaks_insp: np.ndarray, peaks_exp: np.ndarray
    ) -> Tuple[float, float, float]:
        if len(peaks_insp) < 2 or len(peaks_exp) < 2:
            return 0.0, 0.0, 0.0

        insp_durs, exp_durs = [], []
        for i, p_insp in enumerate(peaks_insp[:-1]):
            # Find next expiration trough
            exp_after = peaks_exp[peaks_exp > p_insp]
            if len(exp_after) == 0:
                continue
            p_exp = exp_after[0]
            insp_dur = (p_exp - p_insp) / self.fs
            # Next inspiration starts next cycle
            if i + 1 < len(peaks_insp):
                exp_dur = (peaks_insp[i+1] - p_exp) / self.fs
            else:
                exp_dur = insp_dur * 2   # Approximate

            if 0.1 < insp_dur < 5 and 0.1 < exp_dur < 10:
                insp_durs.append(insp_dur)
                exp_durs.append(exp_dur)

        if not insp_durs:
            return 0.0, 0.0, 0.0

        mean_i = float(np.mean(insp_durs))
        mean_e = float(np.mean(exp_durs))
        ie = mean_i / mean_e if mean_e > 0 else 0.0
        return round(mean_i, 2), round(mean_e, 2), round(ie, 2)

    # ── Apnea ─────────────────────────────────────────────────────────────────

    def _apnea(self, peaks_insp: np.ndarray) -> Tuple[bool, float]:
        if len(peaks_insp) < 2:
            # No recent breaths detected — check entire buffer
            apnea_thresh_samples = int(
                self.cfg.respiratory.apnea_threshold_s * self.fs
            )
            return self._buf_filled > apnea_thresh_samples, float(self._buf_filled / self.fs)

        last_breath_sample = int(peaks_insp[-1])
        samples_since = self._buf_filled - last_breath_sample
        duration_s = samples_since / self.fs

        apnea = duration_s >= self.cfg.respiratory.apnea_threshold_s
        return apnea, round(float(duration_s), 1)

    # ── Relative tidal volume ─────────────────────────────────────────────────

    def _tidal_volume_rel(
        self,
        sig: np.ndarray,
        peaks_insp: np.ndarray,
        peaks_exp: np.ndarray,
    ) -> float:
        """
        Estimate relative tidal volume as peak-to-trough amplitude per breath,
        normalised to the grand mean amplitude.
        """
        if len(peaks_insp) < 2 or len(peaks_exp) < 2:
            return float(np.std(sig)) if len(sig) > 0 else 0.0

        amplitudes = []
        for p_insp in peaks_insp:
            exp_after = peaks_exp[peaks_exp > p_insp]
            if len(exp_after):
                amp = sig[p_insp] - sig[exp_after[0]]
                amplitudes.append(amp)

        if not amplitudes:
            return 0.0

        mean_amp = float(np.mean(amplitudes))
        # Normalise: baseline=1.0 is the historical mean
        return round(max(0.0, mean_amp), 3)

    # ── Signal quality ────────────────────────────────────────────────────────

    def _signal_quality(self, sig: np.ndarray, dom_amp: float) -> float:
        if len(sig) < int(self.fs):
            return 0.5
        if np.std(sig) < 1e-8:
            return 0.0   # Flatline
        snr_proxy = dom_amp / (np.std(sig) + 1e-9)
        return float(np.clip(snr_proxy / 5.0, 0, 1))