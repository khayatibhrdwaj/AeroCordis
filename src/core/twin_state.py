"""
src/core/twin_state.py
──────────────────────
The central state object for the digital twin.
Updated each processing cycle and passed to all downstream modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Deque, List
from collections import deque
import numpy as np


@dataclass
class ECGFeatures:
    """Derived features from the ECG signal."""
    heart_rate: float = 0.0           # bpm
    rr_intervals: List[float] = field(default_factory=list)   # ms
    # HRV time-domain
    rmssd: float = 0.0                # ms
    sdnn: float = 0.0                 # ms
    pnn50: float = 0.0                # %
    mean_rr: float = 0.0              # ms
    # HRV frequency-domain
    lf_power: float = 0.0             # ms²
    hf_power: float = 0.0            # ms²
    lf_hf_ratio: float = 0.0
    total_power: float = 0.0
    # Morphology
    qt_interval_ms: float = 0.0
    qtc_ms: float = 0.0               # Bazett-corrected
    st_deviation_mv: float = 0.0
    p_amplitude_mv: float = 0.0
    qrs_duration_ms: float = 0.0
    # Quality
    signal_quality: float = 1.0       # 0–1


@dataclass
class RespiratoryFeatures:
    """Derived features from the respiratory signal."""
    respiratory_rate: float = 0.0    # breaths/min
    tidal_volume_rel: float = 0.0    # Relative (no absolute without spirometry)
    inspiration_duration_s: float = 0.0
    expiration_duration_s: float = 0.0
    ie_ratio: float = 0.0            # I:E ratio
    apnea_detected: bool = False
    apnea_duration_s: float = 0.0
    dominant_frequency_hz: float = 0.0
    signal_quality: float = 1.0


@dataclass
class SpO2Features:
    """SpO2 and oxygen-related features."""
    spo2: float = 98.0               # %
    spo2_baseline: float = 98.0      # Rolling baseline
    spo2_delta: float = 0.0          # Change from baseline
    perfusion_index: float = 1.0     # PI from plethysmograph amplitude ratio
    signal_quality: float = 1.0


@dataclass
class CouplingFeatures:
    """Heart-lung coupling metrics."""
    # Respiratory Sinus Arrhythmia
    rsa_amplitude: float = 0.0       # bpm — peak HR variation per breath cycle
    rsa_coherence: float = 0.0       # 0–1 coherence between HR and resp

    # Cross-correlation HR × RR (respiratory rate)
    xcorr_peak: float = 0.0         # Peak cross-correlation coefficient (−1 to 1)
    xcorr_lag_s: float = 0.0        # Lag at peak (seconds)
    xcorr_series: List[float] = field(default_factory=list)  # Full xcorr series

    # Phase synchronisation index (Hilbert transform)
    psi: float = 0.0                 # 0–1 (1 = perfect sync)
    phase_difference: float = 0.0   # Radians, instantaneous
    n_ratio: float = 0.0            # n:m locking ratio

    # Hypoxia response coupling
    hypoxia_hr_response: float = 0.0  # Observed bpm per % SpO2 drop
    blunted_response: bool = False


@dataclass
class DerivedIndices:
    """Composite clinical indices."""
    o2_delivery_index: float = 1.0   # Relative O2 delivery (1.0 = normal)
    stability_index: float = 1.0     # 0–1 (1 = maximally stable)
    stress_load: float = 0.0         # 0–100

    # Trend slopes (per minute)
    hr_slope: float = 0.0
    rr_slope: float = 0.0
    spo2_slope: float = 0.0
    o2_slope: float = 0.0

    # ETA to threshold (seconds; None = no alert predicted)
    eta_spo2_critical_s: Optional[float] = None
    eta_hr_critical_s: Optional[float] = None
    eta_o2_critical_s: Optional[float] = None


@dataclass
class AlertEvent:
    """A single alert event."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: str = "info"           # "info" | "warning" | "critical"
    channel: str = ""                # "spo2" | "hr" | "rr" | "o2d" | "coupling"
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    eta_s: Optional[float] = None


@dataclass
class TwinState:
    """
    Complete state of the cardiopulmonary digital twin at one instant.
    Updated by the TwinEngine each processing cycle.
    """
    # Identity
    subject_id: Optional[str] = None
    stay_id: Optional[str] = None
    session_start: datetime = field(default_factory=datetime.utcnow)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    elapsed_s: float = 0.0

    # Feature groups
    ecg: ECGFeatures = field(default_factory=ECGFeatures)
    respiratory: RespiratoryFeatures = field(default_factory=RespiratoryFeatures)
    spo2: SpO2Features = field(default_factory=SpO2Features)
    coupling: CouplingFeatures = field(default_factory=CouplingFeatures)
    indices: DerivedIndices = field(default_factory=DerivedIndices)

    # Raw waveform buffers (for dashboard display)
    ecg_buffer: np.ndarray = field(default_factory=lambda: np.zeros(1250))   # 10s @ 125 Hz
    resp_buffer: np.ndarray = field(default_factory=lambda: np.zeros(625))   # 10s @ 62.5 Hz
    spo2_buffer: np.ndarray = field(default_factory=lambda: np.zeros(625))   # 10s pleth
    hrv_buffer: np.ndarray = field(default_factory=lambda: np.zeros(60))     # 60 RR intervals

    # Trend history (ring buffers, 1 Hz)
    trend_hr: Deque[float] = field(default_factory=lambda: deque(maxlen=3600))
    trend_rr: Deque[float] = field(default_factory=lambda: deque(maxlen=3600))
    trend_spo2: Deque[float] = field(default_factory=lambda: deque(maxlen=3600))
    trend_o2d: Deque[float] = field(default_factory=lambda: deque(maxlen=3600))
    trend_hrv: Deque[float] = field(default_factory=lambda: deque(maxlen=3600))
    trend_stress: Deque[float] = field(default_factory=lambda: deque(maxlen=3600))
    trend_psi: Deque[float] = field(default_factory=lambda: deque(maxlen=3600))
    trend_rsa: Deque[float] = field(default_factory=lambda: deque(maxlen=3600))

    # Active alerts
    active_alerts: List[AlertEvent] = field(default_factory=list)

    # Data quality flags
    ecg_artifact: bool = False
    resp_artifact: bool = False
    spo2_artifact: bool = False

    def update_trends(self) -> None:
        """Push current values into trend ring buffers."""
        self.trend_hr.append(self.ecg.heart_rate)
        self.trend_rr.append(self.respiratory.respiratory_rate)
        self.trend_spo2.append(self.spo2.spo2)
        self.trend_o2d.append(self.indices.o2_delivery_index)
        self.trend_hrv.append(self.ecg.rmssd)
        self.trend_stress.append(self.indices.stress_load)
        self.trend_psi.append(self.coupling.psi)
        self.trend_rsa.append(self.coupling.rsa_amplitude)

    def summary_dict(self) -> dict:
        """Flat dict for logging / export."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "subject_id": self.subject_id,
            "heart_rate": round(self.ecg.heart_rate, 1),
            "respiratory_rate": round(self.respiratory.respiratory_rate, 1),
            "spo2": round(self.spo2.spo2, 1),
            "rmssd": round(self.ecg.rmssd, 1),
            "sdnn": round(self.ecg.sdnn, 1),
            "lf_hf_ratio": round(self.ecg.lf_hf_ratio, 3),
            "o2_delivery_index": round(self.indices.o2_delivery_index, 3),
            "stability_index": round(self.indices.stability_index, 3),
            "stress_load": round(self.indices.stress_load, 1),
            "rsa_amplitude": round(self.coupling.rsa_amplitude, 2),
            "xcorr_peak": round(self.coupling.xcorr_peak, 3),
            "psi": round(self.coupling.psi, 3),
            "hypoxia_blunted": self.coupling.blunted_response,
            "eta_spo2_critical_s": self.indices.eta_spo2_critical_s,
            "eta_hr_critical_s": self.indices.eta_hr_critical_s,
            "n_alerts": len(self.active_alerts),
        }