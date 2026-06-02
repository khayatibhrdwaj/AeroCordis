"""
src/utils/math_helpers.py
──────────────────────────
Shared mathematical helpers used by the TwinEngine to compute
composite physiological indices.

Functions
---------
compute_o2_delivery   — Relative O₂ delivery index (Fick-based approximation)
compute_stability     — Cardiopulmonary stability index (0–1)
compute_stress        — Physiological stress load (0–100)
safe_normalise        — Clamp + normalise a value to [0, 1]
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from ..core.config import StabilityConfig, StressConfig


# ── O₂ Delivery ───────────────────────────────────────────────────────────────

def compute_o2_delivery(
    spo2: float,
    hr: float,
    hr_baseline: float = 72.0,
    sv_adjustment: bool = True,
    hrv_rmssd: float = 0.0,
    hgb: float = 14.0,
) -> float:
    """
    Relative oxygen delivery index.

    Simplified Fick principle (relative, not absolute):
        DO₂_rel = (SpO₂/100) × (HR/HR_baseline) × SV_factor

    SV_factor adjusts for stroke volume using HRV as a proxy:
        Higher RMSSD → better vagal tone → higher SV → SV_factor > 1
        (capped at ±20% of unity)

    Parameters
    ----------
    spo2         : SpO₂ in % (50–100)
    hr           : Heart rate in bpm (> 0)
    hr_baseline  : Reference HR for normalisation (default 72 bpm)
    sv_adjustment: Whether to apply SV correction via HRV
    hrv_rmssd    : RMSSD in ms (used for SV proxy)
    hgb          : Haemoglobin g/dL (not used in relative form; reserved)

    Returns
    -------
    float — relative O₂ delivery index (1.0 = normal)
    """
    if hr <= 0 or spo2 <= 0:
        return 0.0

    sao2 = np.clip(spo2 / 100.0, 0.0, 1.0)
    hr_ratio = hr / max(hr_baseline, 1.0)

    sv_factor = 1.0
    if sv_adjustment and hrv_rmssd > 0:
        # RMSSD 20–80 ms maps to SV factor 0.85–1.15
        rmssd_norm = np.clip((hrv_rmssd - 20.0) / 60.0, 0.0, 1.0)
        sv_factor = 0.85 + 0.30 * rmssd_norm

    do2_rel = sao2 * hr_ratio * sv_factor
    return float(round(np.clip(do2_rel, 0.0, 2.0), 4))


# ── Stability Index ────────────────────────────────────────────────────────────

def compute_stability(
    rmssd: float,
    psi: float,
    xcorr_peak: float,
    cfg: StabilityConfig,
    rmssd_ref: float = 40.0,
) -> float:
    """
    Cardiopulmonary Stability Index (0–1).

    A weighted composite of:
      - HRV RMSSD (normalised to a healthy reference value)
      - Phase Synchronisation Index (PSI) between heart and lung
      - Peak HR×RR cross-correlation magnitude

    Parameters
    ----------
    rmssd      : RMSSD in ms
    psi        : Phase synchronisation index (0–1)
    xcorr_peak : Peak cross-correlation coefficient (−1 to 1)
    cfg        : StabilityConfig with per-component weights
    rmssd_ref  : Healthy reference RMSSD for normalisation (ms)

    Returns
    -------
    float — stability index in [0, 1]
    """
    # Normalise RMSSD: sigmoid-like clamp to [0, 1]
    hrv_norm = float(np.clip(rmssd / rmssd_ref, 0.0, 1.0))

    # PSI already in [0, 1]
    psi_norm = float(np.clip(psi, 0.0, 1.0))

    # xcorr: take absolute value (anti-phase coupling also indicates coupling)
    xcorr_norm = float(np.clip(abs(xcorr_peak), 0.0, 1.0))

    stability = (
        cfg.hrv_weight   * hrv_norm  +
        cfg.psi_weight   * psi_norm  +
        cfg.xcorr_weight * xcorr_norm
    )
    return float(round(np.clip(stability, 0.0, 1.0), 4))


# ── Stress Load ───────────────────────────────────────────────────────────────

def compute_stress(
    hr: float,
    rr: float,
    spo2: float,
    rmssd: float,
    cfg: StressConfig,
) -> float:
    """
    Physiological Stress Load (0–100).

    Each component is scored 0–1 (0 = normal, 1 = maximally stressed)
    then combined with configured weights and scaled to 0–100.

    Components:
      HR   — deviation from normal range (60–100 bpm)
      RR   — deviation from normal range (12–20 br/min)
      SpO₂ — distance below 98% saturation
      HRV  — inverse RMSSD (low HRV = high stress)
    """
    # HR stress: 0 if in normal range, increases outside
    hr_lo, hr_hi = cfg.hr_ref_min, cfg.hr_ref_max
    if hr_lo <= hr <= hr_hi:
        hr_stress = 0.0
    elif hr < hr_lo:
        hr_stress = (hr_lo - hr) / hr_lo
    else:
        hr_stress = (hr - hr_hi) / hr_hi
    hr_stress = float(np.clip(hr_stress, 0.0, 1.0))

    # RR stress
    rr_lo, rr_hi = cfg.rr_ref_min, cfg.rr_ref_max
    if rr_lo <= rr <= rr_hi:
        rr_stress = 0.0
    elif rr < rr_lo and rr > 0:
        rr_stress = (rr_lo - rr) / rr_lo
    elif rr > rr_hi:
        rr_stress = (rr - rr_hi) / rr_hi
    else:
        rr_stress = 0.0
    rr_stress = float(np.clip(rr_stress, 0.0, 1.0))

    # SpO₂ stress: 0 at 98%, 1 at 88%
    spo2_stress = float(np.clip((98.0 - spo2) / 10.0, 0.0, 1.0))

    # HRV stress: 0 at RMSSD ≥ 40 ms, 1 at RMSSD = 0
    hrv_stress = float(np.clip(1.0 - (rmssd / 40.0), 0.0, 1.0))

    stress_raw = (
        cfg.hr_weight   * hr_stress   +
        cfg.rr_weight   * rr_stress   +
        cfg.spo2_weight * spo2_stress +
        cfg.hrv_weight  * hrv_stress
    )
    return float(round(np.clip(stress_raw * 100.0, 0.0, 100.0), 2))


# ── Utility ───────────────────────────────────────────────────────────────────

def safe_normalise(
    value: float,
    lo: float,
    hi: float,
    clip: bool = True,
) -> float:
    """Map value from [lo, hi] to [0, 1]."""
    if hi <= lo:
        return 0.0
    norm = (value - lo) / (hi - lo)
    if clip:
        norm = float(np.clip(norm, 0.0, 1.0))
    return float(norm)
