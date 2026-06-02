"""
src/signals/ecg_processor.py
─────────────────────────────
ECG signal processing pipeline:
  1. Bandpass + notch filter
  2. QRS detection (Pan-Tompkins via neurokit2)
  3. HRV time-domain (RMSSD, SDNN, pNN50)
  4. HRV frequency-domain (LF, HF, LF/HF via Welch PSD)
  5. Waveform morphology (QT, QTc Bazett, ST, P, QRS duration)
  6. Signal quality index
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from typing import List, Tuple, Optional
from loguru import logger

try:
    import neurokit2 as nk
    HAS_NK = True
except ImportError:
    HAS_NK = False
    logger.warning("neurokit2 not installed — using fallback QRS detector")

from ..core.config import TwinConfig
from ..core.twin_state import ECGFeatures


class ECGProcessor:
    """
    Processes raw ECG samples into ECGFeatures.

    Internal sliding buffers accumulate signal so HRV computations
    have enough data (5-minute window). Per-chunk features reflect
    the most recent estimate over the current window.
    """

    # HRV window in seconds
    _HRV_WIN_S = 300

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self.fs = cfg.ecg.sampling_rate

        # Build filters
        self._bp_filter = self._make_bandpass()
        self._notch_b, self._notch_a = self._make_notch()

        # Sliding ECG buffer for HRV (5 min)
        self._buf_len = int(self._HRV_WIN_S * self.fs)
        self._buf = np.zeros(self._buf_len)

        # Accumulated RR intervals (ms) — reset when buffer rolls
        self._rr_ms: List[float] = []
        self._last_r_sample: int = 0   # Global sample count

        self._total_samples = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, raw: np.ndarray) -> ECGFeatures:
        """Process one chunk of raw ECG samples."""
        if len(raw) == 0:
            return ECGFeatures(signal_quality=0.0)

        # 1. Filter
        filtered = self._filter(raw)

        # 2. Append to sliding buffer
        n = len(filtered)
        self._buf = np.roll(self._buf, -n)
        self._buf[-n:] = filtered
        self._total_samples += n

        # 3. Detect QRS on last chunk
        r_locs = self._detect_qrs(filtered)

        print(f"Detected peaks: {len(r_locs)}")
        print(f"Peak locations: {r_locs[:10]}")

        # 4. Convert local R-peak indices to RR intervals
        new_rr = self._rr_from_peaks(r_locs, n)

        # 5. HRV
        hrv = self._compute_hrv(new_rr)

        # 6. Morphology (from current buffer)
        morph = self._morphology(r_locs, filtered)

        # 7. Signal quality
        sqI = self._signal_quality(filtered)

        return ECGFeatures(
            heart_rate=60_000.0 / hrv["mean_rr"] if hrv["mean_rr"] > 0 else 0.0,
            rr_intervals=self._rr_ms[-60:],   # Last 60 beats
            rmssd=hrv["rmssd"],
            sdnn=hrv["sdnn"],
            pnn50=hrv["pnn50"],
            mean_rr=hrv["mean_rr"],
            lf_power=hrv["lf_power"],
            hf_power=hrv["hf_power"],
            lf_hf_ratio=hrv["lf_hf_ratio"],
            total_power=hrv["total_power"],
            qt_interval_ms=morph["qt_ms"],
            qtc_ms=morph["qtc_ms"],
            st_deviation_mv=morph["st_dev"],
            p_amplitude_mv=morph["p_amp"],
            qrs_duration_ms=morph["qrs_dur"],
            signal_quality=sqI,
        )

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _make_bandpass(self) -> Tuple[np.ndarray, np.ndarray]:
        low  = self.cfg.ecg.bandpass_low  / (self.fs / 2)
        high = self.cfg.ecg.bandpass_high / (self.fs / 2)
        low  = max(0.001, min(low,  0.999))
        high = max(0.001, min(high, 0.999))
        coeff = sp_signal.butter(4, [low, high], btype="band", output="ba")
        return np.asarray(coeff[0]), np.asarray(coeff[1])  # type: ignore[index]

    def _make_notch(self) -> Tuple[np.ndarray, np.ndarray]:
        f0 = self.cfg.ecg.notch_freq
        Q  = 30.0
        b, a = sp_signal.iirnotch(f0 / (self.fs / 2), Q)
        return np.asarray(b), np.asarray(a)

    def _filter(self, raw: np.ndarray) -> np.ndarray:
        b = np.asarray(self._bp_filter[0])
        a = np.asarray(self._bp_filter[1])
        try:
            filtered = sp_signal.filtfilt(b, a, raw)
            filtered = sp_signal.filtfilt(self._notch_b, self._notch_a, filtered)
        except ValueError:
            # Chunk too short for filtfilt
            filtered = raw.copy()
        return filtered

    # ── QRS detection ─────────────────────────────────────────────────────────

    def _detect_qrs(self, filtered: np.ndarray) -> np.ndarray:
        if len(filtered) < int(self.fs * 0.5):
            return np.array([], dtype=int)

        if HAS_NK:
            try:
                _, info = nk.ecg_peaks(
                    filtered,
                    sampling_rate=int(self.fs),
                    method="pantompkins1985",
                    correct_artifacts=True,
                )
                return np.asarray(info["ECG_R_Peaks"], dtype=int)
            except Exception as e:
                logger.debug(f"NK QRS fallback: {e}")

        # Fallback: simple derivative-based detector
        return self._simple_qrs(filtered)

    def _simple_qrs(self, ecg: np.ndarray) -> np.ndarray:
        """Pan-Tompkins simplified implementation."""
        # Derivative
        deriv = np.diff(ecg, prepend=ecg[0])
        # Square
        squared = deriv ** 2
        # Moving average (150 ms window)
        win = max(1, int(0.15 * self.fs))
        kernel = np.ones(win) / win
        mwa = np.convolve(squared, kernel, mode="same")
        # Threshold
        threshold = np.mean(mwa) + 0.5 * np.std(mwa)
        above = mwa > threshold
        # Find peaks
        min_dist = int(self.cfg.ecg.qrs_min_distance_ms / 1000 * self.fs)
        peaks = []
        i = 0
        while i < len(above):
            if above[i]:
                start = i
                while i < len(above) and above[i]:
                    i += 1
                segment = ecg[start:i]
                if len(segment) > 0:
                    peak = start + int(np.argmax(np.abs(segment)))
                    if not peaks or (peak - peaks[-1]) >= min_dist:
                        peaks.append(peak)
            i += 1
        return np.array(peaks, dtype=int)

    # ── RR interval computation ───────────────────────────────────────────────

    def _rr_from_peaks(self, r_locs: np.ndarray, chunk_len: int) -> List[float]:
        """Convert R-peak sample indices in the chunk to RR intervals (ms)."""
        new_rr = []
        if len(r_locs) == 0:
            self._total_samples += chunk_len
            return new_rr
        
        # Convert local chunk indices → global indices
        global_peaks = r_locs + (self._total_samples - chunk_len)

        # Compare first peak with previous chunk peak
        if self._last_r_sample > 0:
            rr = (global_peaks[0] - self._last_r_sample) / self.fs * 1000.0
            if 300 < rr < 2000:
                new_rr.append(rr)

        # RR intervals within current chunk
        for i in range(1, len(global_peaks)):
            rr = (global_peaks[i] - global_peaks[i-1]) / self.fs * 1000.0
            if 300 < rr < 2000:   # Physiologically valid RR (30–200 bpm)
                new_rr.append(rr)

    # Store latest peak for next chunk
        self._last_r_sample = global_peaks[-1]
        self._rr_ms.extend(new_rr)

        # Keep last 300 beats
        if len(self._rr_ms) > 300:
            self._rr_ms = self._rr_ms[-300:]

        return new_rr

    # ── HRV ──────────────────────────────────────────────────────────────────

    def _compute_hrv(self, new_rr: List[float]) -> dict:
        rr = np.array(self._rr_ms)
        null = {"mean_rr": 0, "rmssd": 0, "sdnn": 0, "pnn50": 0,
                "lf_power": 0, "hf_power": 0, "lf_hf_ratio": 0, "total_power": 0}

        if len(rr) < 4:
            return null

        mean_rr = float(np.mean(rr))
        sdnn    = float(np.std(rr, ddof=1))
        diff_rr = np.diff(rr)
        rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
        pnn50   = float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100)

        # Frequency domain via Welch PSD on interpolated RR series
        lf_p, hf_p, total_p = self._hrv_frequency(rr)

        return {
            "mean_rr": mean_rr,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnn50,
            "lf_power": lf_p,
            "hf_power": hf_p,
            "lf_hf_ratio": lf_p / hf_p if hf_p > 0 else 0.0,
            "total_power": total_p,
        }

    def _hrv_frequency(self, rr_ms: np.ndarray) -> Tuple[float, float, float]:
        """Welch PSD of the RR series (interpolated to 4 Hz)."""
        try:
            interp_fs = 4.0
            t_rr = np.cumsum(rr_ms / 1000.0)
            t_interp = np.arange(t_rr[0], t_rr[-1], 1.0 / interp_fs)
            if len(t_interp) < 8:
                return 0.0, 0.0, 0.0
            rr_interp = np.interp(t_interp, t_rr, rr_ms)
            rr_detrend = sp_signal.detrend(rr_interp)

            nperseg = min(256, len(rr_detrend))
            freqs, psd = sp_signal.welch(rr_detrend, fs=interp_fs, nperseg=nperseg)
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

            lf_lo, lf_hi = self.cfg.hrv.lf_band
            hf_lo, hf_hi = self.cfg.hrv.hf_band
            lf_mask = (freqs >= lf_lo) & (freqs < lf_hi)
            hf_mask = (freqs >= hf_lo) & (freqs < hf_hi)

            lf_p  = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0  # type: ignore[attr-defined]
            hf_p  = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0  # type: ignore[attr-defined]
            tot   = float(np.trapz(psd, freqs))  # type: ignore[attr-defined]

            return max(0, lf_p), max(0, hf_p), max(0, tot)
        except Exception as e:
            logger.debug(f"HRV freq error: {e}")
            return 0.0, 0.0, 0.0

    # ── Morphology ────────────────────────────────────────────────────────────

    def _morphology(self, r_locs: np.ndarray, ecg: np.ndarray) -> dict:
        null = {"qt_ms": 0.0, "qtc_ms": 0.0, "st_dev": 0.0, "p_amp": 0.0, "qrs_dur": 0.0}
        if len(r_locs) < 2 or len(self._rr_ms) < 2:
            return null

        mean_rr_s = np.mean(self._rr_ms) / 1000.0
        results = {"qt_ms": [], "st_dev": [], "p_amp": [], "qrs_dur": []}

        for r in r_locs:
            r = int(r)
            # QRS window: −40 ms to +60 ms around R
            q_start = max(0, r - int(0.04 * self.fs))
            s_end   = min(len(ecg)-1, r + int(0.06 * self.fs))
            qrs_dur = (s_end - q_start) / self.fs * 1000.0
            results["qrs_dur"].append(qrs_dur)

            # ST deviation: 80 ms after J-point (S-end)
            j_point = s_end
            st_sample = min(len(ecg)-1, j_point + int(0.08 * self.fs))
            if j_point < len(ecg) and st_sample < len(ecg):
                st_dev = float(ecg[st_sample]) * 1.0   # mV (assume unit scale)
                results["st_dev"].append(st_dev)

            # P-wave: −200 ms to −100 ms before R
            p_lo = max(0, r - int(0.20 * self.fs))
            p_hi = max(0, r - int(0.10 * self.fs))
            if p_hi > p_lo:
                p_amp = float(np.max(ecg[p_lo:p_hi]))
                results["p_amp"].append(p_amp)

            # T-wave end estimate: R + 400 ms (simplified)
            t_end = min(len(ecg)-1, r + int(0.40 * self.fs))
            qt_ms = (t_end - q_start) / self.fs * 1000.0
            results["qt_ms"].append(qt_ms)

        def safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        qt_ms  = safe_mean(results["qt_ms"])
        rr_s   = mean_rr_s if mean_rr_s > 0 else 1.0
        qtc_ms = qt_ms / np.sqrt(rr_s) if rr_s > 0 else 0.0

        return {
            "qt_ms":   qt_ms,
            "qtc_ms":  qtc_ms,
            "st_dev":  safe_mean(results["st_dev"]),
            "p_amp":   safe_mean(results["p_amp"]),
            "qrs_dur": safe_mean(results["qrs_dur"]),
        }

    # ── Signal quality ────────────────────────────────────────────────────────

    def _signal_quality(self, ecg: np.ndarray) -> float:
        """
        Simple signal quality index based on:
        - Power in physiological frequency band vs broadband
        - Flatline detection
        - Saturation detection
        """
        if len(ecg) < int(self.fs):
            return 0.5
        if np.std(ecg) < 1e-6:   # Flatline
            return 0.0

        total_power = np.mean(ecg ** 2)
        b = np.asarray(self._bp_filter[0])
        a = np.asarray(self._bp_filter[1])
        try:
            ecg_bp = sp_signal.filtfilt(b, a, ecg)
            bp_power = np.mean(ecg_bp ** 2)
            ratio = bp_power / (total_power + 1e-12)
            return float(np.clip(ratio * 2, 0, 1))
        except Exception:
            return 0.5