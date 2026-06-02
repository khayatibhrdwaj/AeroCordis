"""
src/signals/spo2_processor.py
──────────────────────────────
SpO₂ processing:
  1. Artifact rejection (physiologically implausible values)
  2. Perfusion index estimation from plethysmograph AC/DC ratio
  3. Rolling baseline with exponential smoothing
  4. Delta from baseline
  5. Signal quality index
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Deque
from loguru import logger

from ..core.config import TwinConfig
from ..core.twin_state import SpO2Features


class SpO2Processor:

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self.fs  = cfg.spo2.sampling_rate

        # Rolling window for baseline (e.g. 120 s)
        buf_len = int(cfg.spo2.baseline_window_s * self.fs)
        self._history: Deque[float] = deque(maxlen=buf_len)
        self._baseline = 98.0   # Initial assumed normal

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, raw: np.ndarray) -> SpO2Features:
        """
        Parameters
        ----------
        raw : np.ndarray
            If this is pleth waveform data (high frequency), the array contains
            the AC-coupled pulse waveform. If it is numerics (1 Hz), it contains
            SpO₂ percentages directly.
            The processor detects the mode by checking value range.
        """
        if len(raw) == 0:
            return SpO2Features(signal_quality=0.0)

        # Detect mode: numerics (50–100 %) vs pleth waveform (arbitrary units)
        is_numeric = (np.max(raw) <= 100.0 and np.min(raw) >= 50.0)

        if is_numeric:
            return self._process_numerics(raw)
        else:
            return self._process_pleth(raw)

    # ── Numerics mode (1 Hz SpO₂ values) ─────────────────────────────────────

    def _process_numerics(self, values: np.ndarray) -> SpO2Features:
        # Filter artifacts
        valid = values[(values >= self.cfg.spo2.min_valid) & (values <= 100.0)]
        if len(valid) == 0:
            return SpO2Features(spo2=self._baseline, signal_quality=0.1)

        current = float(np.median(valid))

        # Update history & baseline
        for v in valid:
            self._history.append(v)
        if len(self._history) >= 5:
            # Exponential moving average
            self._baseline = float(np.mean(list(self._history)[-60:])) \
                if len(self._history) >= 60 else float(np.mean(self._history))

        delta = current - self._baseline
        quality = len(valid) / len(values)   # Fraction of valid samples

        return SpO2Features(
            spo2=round(current, 1),
            spo2_baseline=round(self._baseline, 1),
            spo2_delta=round(delta, 2),
            perfusion_index=1.0,   # Not available in numerics mode
            signal_quality=round(float(quality), 2),
        )

    # ── Plethysmograph mode (high-frequency waveform) ─────────────────────────

    def _process_pleth(self, pleth: np.ndarray) -> SpO2Features:
        """
        Plethysmograph processing:
        - AC component = pulsatile (at heart rate frequency)
        - DC component = mean signal level
        - Perfusion Index = AC_peak / DC_mean × 100
        SpO₂ value is embedded as a numeric alongside; here we return
        PI and quality only, assuming SpO₂ numerics are processed separately.
        """
        if len(pleth) < 4:
            return SpO2Features(signal_quality=0.0)

        dc = float(np.mean(np.abs(pleth)))
        ac = float(np.max(pleth) - np.min(pleth))
        pi = (ac / dc * 100) if dc > 0 else 0.0

        # Quality: check for flatline or saturation
        if np.std(pleth) < 1e-8:
            quality = 0.0
        elif ac < 0.01 * dc:
            quality = 0.2   # Very low PI — poor perfusion or motion
        else:
            quality = float(np.clip(pi / 5.0, 0, 1))

        # Use last known spo2 from history
        spo2 = self._baseline

        return SpO2Features(
            spo2=round(spo2, 1),
            spo2_baseline=round(self._baseline, 1),
            spo2_delta=0.0,
            perfusion_index=round(pi, 2),
            signal_quality=round(quality, 2),
        )