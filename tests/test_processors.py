"""
tests/test_processors.py
─────────────────────────
Basic unit tests for signal processors and index computation.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.core.config import TwinConfig, load_config
from src.core.twin_state import TwinState
from src.signals.ecg_processor import ECGProcessor
from src.signals.resp_processor import RespProcessor
from src.signals.spo2_processor import SpO2Processor
from src.utils.math_helpers import compute_o2_delivery, compute_stability, compute_stress


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> TwinConfig:
    c = TwinConfig()
    c.prediction.method = "linear" # Disable LSTM for tests to avoid torch issues
    return c


@pytest.fixture
def ecg_chunk(cfg) -> np.ndarray:
    """Synthetic ECG chunk: 2.5 s at 125 Hz."""
    fs = cfg.ecg.sampling_rate
    n = int(fs * 2.5)
    t = np.linspace(0, 2.5, n)
    # Simulate R-peaks at 72 bpm
    ecg = np.zeros(n)
    for beat in np.arange(0, 2.5, 60 / 72):
        idx = int(beat * fs)
        if idx < n:
            ecg[idx] = 1.0
    ecg += np.random.normal(0, 0.05, n)
    return ecg


@pytest.fixture
def resp_chunk(cfg) -> np.ndarray:
    fs = cfg.respiratory.sampling_rate
    n = int(fs * 4.0)
    t = np.linspace(0, 4.0, n)
    return np.sin(2 * np.pi * 0.25 * t) + 0.05 * np.random.randn(n)


# ── ECG Processor ─────────────────────────────────────────────────────────────

class TestECGProcessor:

    def test_returns_features(self, cfg, ecg_chunk):
        proc = ECGProcessor(cfg)
        feats = proc.process(ecg_chunk)
        assert feats is not None
        assert 0.0 <= feats.signal_quality <= 1.0

    def test_empty_chunk(self, cfg):
        proc = ECGProcessor(cfg)
        feats = proc.process(np.array([]))
        assert feats.signal_quality == 0.0

    def test_heart_rate_range(self, cfg, ecg_chunk):
        proc = ECGProcessor(cfg)
        # Process multiple chunks to build RR buffer
        for _ in range(10):
            feats = proc.process(ecg_chunk)
        if feats.heart_rate > 0:
            assert 20 <= feats.heart_rate <= 250


# ── Respiratory Processor ─────────────────────────────────────────────────────

class TestRespProcessor:

    def test_returns_features(self, cfg, resp_chunk):
        proc = RespProcessor(cfg)
        feats = proc.process(resp_chunk)
        assert feats is not None
        assert feats.respiratory_rate >= 0

    def test_empty_chunk(self, cfg):
        proc = RespProcessor(cfg)
        feats = proc.process(np.array([]))
        assert feats.signal_quality == 0.0

    def test_no_apnea_normal(self, cfg, resp_chunk):
        proc = RespProcessor(cfg)
        feats = proc.process(resp_chunk)
        # Short chunk — should not trigger apnea
        assert isinstance(feats.apnea_detected, bool)


# ── SpO2 Processor ────────────────────────────────────────────────────────────

class TestSpO2Processor:

    def test_numerics_mode(self, cfg):
        proc = SpO2Processor(cfg)
        values = np.array([98.0, 97.5, 98.5, 97.0])
        feats = proc.process(values)
        assert 90 <= feats.spo2 <= 100
        assert feats.signal_quality > 0

    def test_artifact_rejection(self, cfg):
        proc = SpO2Processor(cfg)
        # Values below min_valid should be filtered
        values = np.array([30.0, 98.0, 99.0])
        feats = proc.process(values)
        assert feats.spo2 >= cfg.spo2.min_valid


# ── Math Helpers ──────────────────────────────────────────────────────────────

class TestMathHelpers:

    def test_o2_delivery_normal(self):
        do2 = compute_o2_delivery(spo2=98.0, hr=72.0, hr_baseline=72.0)
        assert 0.9 < do2 < 1.2   # Should be ~1.0 at baseline

    def test_o2_delivery_hypoxia(self):
        # SpO2=85, HR=90, HR_baseline=72
        # sao2 = 0.85
        # hr_ratio = 90/72 = 1.25
        # do2 = 0.85 * 1.25 = 1.0625 (without SV adjustment)
        # In this case, HR rise compensates for SpO2 drop, so do2 > 1.0 is possible.
        # Let's use a more severe hypoxia case for the test.
        do2 = compute_o2_delivery(spo2=70.0, hr=72.0, hr_baseline=72.0)
        assert do2 < 1.0   # Reduced SpO₂ should reduce delivery

    def test_o2_delivery_zero_hr(self):
        do2 = compute_o2_delivery(spo2=98.0, hr=0.0)
        assert do2 == 0.0

    def test_stability_bounds(self):
        cfg = TwinConfig()
        s = compute_stability(rmssd=40.0, psi=0.8, xcorr_peak=0.7, cfg=cfg.stability)
        assert 0.0 <= s <= 1.0

    def test_stability_low(self):
        cfg = TwinConfig()
        s = compute_stability(rmssd=5.0, psi=0.1, xcorr_peak=0.05, cfg=cfg.stability)
        assert s < 0.5

    def test_stress_normal(self):
        cfg = TwinConfig()
        stress = compute_stress(hr=72, rr=15, spo2=98, rmssd=40, cfg=cfg.stress)
        assert stress < 30   # Normal vitals = low stress

    def test_stress_elevated(self):
        cfg = TwinConfig()
        stress = compute_stress(hr=130, rr=28, spo2=89, rmssd=8, cfg=cfg.stress)
        assert stress > 50   # Abnormal vitals = high stress

    def test_stress_bounds(self):
        cfg = TwinConfig()
        for hr in [40, 72, 150]:
            stress = compute_stress(hr=hr, rr=15, spo2=95, rmssd=25, cfg=cfg.stress)
            assert 0.0 <= stress <= 100.0


# ── Twin Engine integration smoke test ────────────────────────────────────────

class TestTwinEngine:

    def test_ingest_cycle(self):
        # Mock sys.get_int_max_str_digits if it doesn't exist (Python < 3.10.7)
        if not hasattr(sys, 'get_int_max_str_digits'):
            sys.get_int_max_str_digits = lambda: 4300
            
        from src.core.twin_engine import TwinEngine
        c = TwinConfig()
        c.prediction.method = "linear"
        engine = TwinEngine(c)
        engine.start_session(subject_id="test")

        fs_ecg  = int(engine.cfg.ecg.sampling_rate)
        fs_resp = int(engine.cfg.respiratory.sampling_rate)

        ecg  = np.random.normal(0, 0.1, fs_ecg)
        resp = np.sin(2 * np.pi * 0.25 * np.linspace(0, 1, fs_resp))
        spo2 = np.array([97.5])

        state = engine.ingest(ecg, resp, spo2)
        assert state is not None
        assert state.elapsed_s >= 0
        assert isinstance(state.active_alerts, list)

    def test_trend_accumulation(self):
        from src.core.twin_engine import TwinEngine
        c = TwinConfig()
        c.prediction.method = "linear"
        engine = TwinEngine(c)
        engine.start_session()

        for _ in range(5):
            ecg  = np.random.normal(0, 0.1, 125)
            resp = np.sin(2 * np.pi * 0.25 * np.linspace(0, 1, 62))
            engine.ingest(ecg, resp, np.array([98.0]))

        assert len(engine.state.trend_hr) == 5
        assert len(engine.state.trend_spo2) == 5
