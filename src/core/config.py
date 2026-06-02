"""
src/core/config.py
──────────────────
Loads twin_config.yaml into typed dataclasses.
All other modules import from here so there is a single source of truth.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import yaml


# ── Locate config directory ───────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ECGConfig:
    sampling_rate: float = 125.0
    lead: str = "II"
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    notch_freq: float = 60.0
    qrs_min_distance_ms: float = 250.0


@dataclass
class RespiratoryConfig:
    sampling_rate: float = 62.5
    bandpass_low: float = 0.1
    bandpass_high: float = 1.0
    apnea_threshold_s: float = 15.0


@dataclass
class SpO2Config:
    sampling_rate: float = 1.0
    min_valid: float = 50.0
    baseline_window_s: float = 120.0


@dataclass
class HRVConfig:
    window_s: int = 300
    overlap_s: int = 240
    rmssd_normal_min: float = 20.0
    sdnn_normal_min: float = 50.0
    pnn50_normal_min: float = 3.0
    lf_band: Tuple[float, float] = (0.04, 0.15)
    hf_band: Tuple[float, float] = (0.15, 0.40)
    ulf_band: Tuple[float, float] = (0.0, 0.0033)
    vlf_band: Tuple[float, float] = (0.0033, 0.04)


@dataclass
class RSAConfig:
    bandpass_around_resp: float = 0.02
    min_amplitude_bpm: float = 2.0


@dataclass
class CrossCorrConfig:
    max_lag_s: float = 30.0
    window_s: float = 120.0


@dataclass
class PhaseSyncConfig:
    method: str = "hilbert"
    n_bins: int = 20
    window_s: float = 60.0
    n_ratio_default: float = 4.0


@dataclass
class CouplingConfig:
    rsa: RSAConfig = field(default_factory=RSAConfig)
    cross_correlation: CrossCorrConfig = field(default_factory=CrossCorrConfig)
    phase_sync: PhaseSyncConfig = field(default_factory=PhaseSyncConfig)


@dataclass
class OxygenConfig:
    hr_baseline: float = 72.0
    sv_adjustment: bool = True
    hgb_assumed: float = 14.0
    critical_threshold: float = 0.60
    warning_threshold: float = 0.80
    optimal_threshold: float = 0.95


@dataclass
class StabilityConfig:
    hrv_weight: float = 0.30
    psi_weight: float = 0.40
    xcorr_weight: float = 0.30


@dataclass
class StressConfig:
    hr_weight: float = 0.30
    rr_weight: float = 0.20
    spo2_weight: float = 0.30
    hrv_weight: float = 0.20
    hr_ref_min: float = 60.0
    hr_ref_max: float = 100.0
    rr_ref_min: float = 12.0
    rr_ref_max: float = 20.0


@dataclass
class AlertThreshold:
    critical_low: Optional[float] = None
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    critical_high: Optional[float] = None
    critical: Optional[float] = None
    warning: Optional[float] = None
    prediction_horizon_s: float = 300.0


@dataclass
class HypoxiaResponseConfig:
    expected_hr_rise_per_spo2_fall: float = 1.5


@dataclass
class AlertsConfig:
    spo2: AlertThreshold = field(default_factory=lambda: AlertThreshold(critical=90, warning=94))
    heart_rate: AlertThreshold = field(default_factory=lambda: AlertThreshold(
        critical_low=40, warning_low=50, warning_high=110, critical_high=140))
    respiratory_rate: AlertThreshold = field(default_factory=lambda: AlertThreshold(
        critical_low=6, warning_low=8, warning_high=25, critical_high=30))
    o2_delivery: AlertThreshold = field(default_factory=lambda: AlertThreshold(critical=0.60, warning=0.80))
    hypoxia_response: HypoxiaResponseConfig = field(default_factory=HypoxiaResponseConfig)


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    sequence_length: int = 60
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32


@dataclass
class PredictionConfig:
    method: str = "combined"
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    linear_window_s: float = 30.0
    combined_weight_lstm: float = 0.7


@dataclass
class MIMICConfig:
    access_method: str = "physionet"
    base_url: str = "https://physionet.org/files/mimic4wdb/0.1.0/"
    waveform_channels: List[str] = field(default_factory=lambda: ["II", "RESP", "Pleth"])
    numerics_columns: List[str] = field(default_factory=lambda: ["heart rate", "respiratory rate", "spo2"])
    chunk_size_s: int = 3600


@dataclass
class DashboardConfig:
    update_interval_ms: int = 250
    waveform_window_s: int = 10
    trend_window_s: int = 300
    port: int = 8501


@dataclass
class TwinConfig:
    ecg: ECGConfig = field(default_factory=ECGConfig)
    respiratory: RespiratoryConfig = field(default_factory=RespiratoryConfig)
    spo2: SpO2Config = field(default_factory=SpO2Config)
    hrv: HRVConfig = field(default_factory=HRVConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    oxygen: OxygenConfig = field(default_factory=OxygenConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    mimic: MIMICConfig = field(default_factory=MIMICConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# ── Builder ───────────────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> TwinConfig:
    """Load configuration from YAML, falling back to defaults."""
    if path is None:
        path = CONFIG_DIR / "twin_config.yaml"

    if not path.exists():
        return TwinConfig()

    raw = _load_yaml(path)
    s = raw.get("signals", {})
    c = raw.get("coupling", {})
    a = raw.get("alerts", {})
    p = raw.get("prediction", {})
    m = raw.get("mimic", {})
    idx = raw.get("indices", {})

    ecg_raw = s.get("ecg", {})
    resp_raw = s.get("respiratory", {})
    spo2_raw = s.get("spo2", {})
    hrv_raw = raw.get("hrv", {})

    return TwinConfig(
        ecg=ECGConfig(**{k: v for k, v in ecg_raw.items() if k in ECGConfig.__dataclass_fields__}),
        respiratory=RespiratoryConfig(**{k: v for k, v in resp_raw.items() if k in RespiratoryConfig.__dataclass_fields__}),
        spo2=SpO2Config(**{k: v for k, v in spo2_raw.items() if k in SpO2Config.__dataclass_fields__}),
        hrv=HRVConfig(
            window_s=hrv_raw.get("window_s", 300),
            overlap_s=hrv_raw.get("overlap_s", 240),
            rmssd_normal_min=hrv_raw.get("rmssd_normal_min", 20.0),
            sdnn_normal_min=hrv_raw.get("sdnn_normal_min", 50.0),
            pnn50_normal_min=hrv_raw.get("pnn50_normal_min", 3.0),
            lf_band=tuple(hrv_raw.get("lf_band", [0.04, 0.15])),
            hf_band=tuple(hrv_raw.get("hf_band", [0.15, 0.40])),
        ),
        coupling=CouplingConfig(
            rsa=RSAConfig(**{k: v for k, v in c.get("rsa", {}).items() if k in RSAConfig.__dataclass_fields__}),
            cross_correlation=CrossCorrConfig(**{k: v for k, v in c.get("cross_correlation", {}).items() if k in CrossCorrConfig.__dataclass_fields__}),
            phase_sync=PhaseSyncConfig(**{k: v for k, v in c.get("phase_sync", {}).items() if k in PhaseSyncConfig.__dataclass_fields__}),
        ),
        oxygen=OxygenConfig(**{k: v for k, v in raw.get("oxygen", {}).items() if k in OxygenConfig.__dataclass_fields__}),
        stability=StabilityConfig(**{k: v for k, v in idx.get("stability", {}).items() if k in StabilityConfig.__dataclass_fields__}),
        stress=StressConfig(**{k: v for k, v in idx.get("stress_load", {}).items() if k in StressConfig.__dataclass_fields__}),
        alerts=AlertsConfig(
            spo2=AlertThreshold(**{k: v for k, v in a.get("spo2", {}).items() if k in AlertThreshold.__dataclass_fields__}),
            heart_rate=AlertThreshold(**{k: v for k, v in a.get("heart_rate", {}).items() if k in AlertThreshold.__dataclass_fields__}),
            respiratory_rate=AlertThreshold(**{k: v for k, v in a.get("respiratory_rate", {}).items() if k in AlertThreshold.__dataclass_fields__}),
            o2_delivery=AlertThreshold(**{k: v for k, v in a.get("o2_delivery", {}).items() if k in AlertThreshold.__dataclass_fields__}),
        ),
        prediction=PredictionConfig(
            method=p.get("method", "combined"),
            lstm=LSTMConfig(**{k: v for k, v in p.get("lstm", {}).items() if k in LSTMConfig.__dataclass_fields__}),
            linear_window_s=p.get("linear", {}).get("window_s", 30.0),
            combined_weight_lstm=p.get("combined_weight_lstm", 0.7),
        ),
        mimic=MIMICConfig(
            access_method=m.get("access_method", "physionet"),
            base_url=m.get("physionet", {}).get("base_url", "https://physionet.org/files/mimic4wdb/0.1.0/"),
            waveform_channels=m.get("waveform_channels", ["II", "RESP", "Pleth"]),
            numerics_columns=m.get("numerics_columns", ["heart rate", "respiratory rate", "spo2"]),
            chunk_size_s=m.get("chunk_size_s", 3600),
        ),
        dashboard=DashboardConfig(**{k: v for k, v in raw.get("dashboard", {}).items() if k in DashboardConfig.__dataclass_fields__}),
    )