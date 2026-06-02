"""
src/core/twin_engine.py
───────────────────────
Central orchestrator. Accepts raw signal chunks, runs all
processing pipelines in order, updates TwinState, and emits alerts.

Processing order per cycle:
  1. ECG  → QRS detection → HRV → morphology features
  2. RESP → rate detection → apnea → I:E ratio
  3. SpO₂ → quality check → baseline → perfusion index
  4. Coupling → RSA → xcorr(HR,RR) → phase sync → hypoxia response
  5. Indices → O₂ delivery → stability → stress load
  6. Prediction → trend slopes → ETA to threshold
  7. Alerts → evaluate thresholds → emit events
  8. State → update trends → timestamp
"""

from __future__ import annotations
import time
from datetime import datetime
from typing import Optional
import numpy as np
from loguru import logger

from .config import TwinConfig, load_config
from .twin_state import TwinState, AlertEvent

from ..signals.ecg_processor import ECGProcessor
from ..signals.resp_processor import RespProcessor
from ..signals.spo2_processor import SpO2Processor
from ..coupling.rsa import RSAAnalyzer
from ..coupling.cross_correlation import HRRRCrossCorrelation
from ..coupling.phase_sync import PhaseSynchronisation
from ..coupling.hypoxia_response import HypoxiaResponseMonitor
from ..prediction.trend_predictor import TrendPredictor
from ..alerts.alert_engine import AlertEngine
from ..utils.math_helpers import compute_o2_delivery, compute_stability, compute_stress


class TwinEngine:
    """
    Main processing engine for the Cardiopulmonary Digital Twin.

    Usage:
        engine = TwinEngine(config)
        engine.start_session(subject_id="10000032", stay_id="30000426")

        # In your data loop:
        engine.ingest(ecg_chunk, resp_chunk, spo2_chunk, timestamp)
        state = engine.state  # Read updated state
    """

    def __init__(self, config: Optional[TwinConfig] = None):
        self.cfg = config or load_config()
        self.state = TwinState()

        # Initialise all processing modules
        self.ecg_proc  = ECGProcessor(self.cfg)
        self.resp_proc = RespProcessor(self.cfg)
        self.spo2_proc = SpO2Processor(self.cfg)

        self.rsa_analyzer   = RSAAnalyzer(self.cfg)
        self.xcorr_analyzer = HRRRCrossCorrelation(self.cfg)
        self.phase_sync     = PhaseSynchronisation(self.cfg)
        self.hypoxia_mon    = HypoxiaResponseMonitor(self.cfg)

        self.predictor    = TrendPredictor(self.cfg)
        self.alert_engine = AlertEngine(self.cfg)

        self._cycle_count = 0
        self._last_trend_update = 0.0

        logger.info("TwinEngine initialised with config.")

    # ── Session management ────────────────────────────────────────────────────

    def start_session(
        self,
        subject_id: Optional[str] = None,
        stay_id: Optional[str] = None,
    ) -> None:
        self.state = TwinState(
            subject_id=subject_id,
            stay_id=stay_id,
            session_start=datetime.utcnow(),
        )
        logger.info(f"Session started | subject={subject_id} | stay={stay_id}")

    # ── Main entry point ──────────────────────────────────────────────────────

    def ingest(
        self,
        ecg_chunk: np.ndarray,
        resp_chunk: np.ndarray,
        spo2_chunk: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> TwinState:
        """
        Process one chunk of raw signals and return updated TwinState.

        Parameters
        ----------
        ecg_chunk   : np.ndarray, shape (N,)  — raw ECG at cfg.ecg.sampling_rate Hz
        resp_chunk  : np.ndarray, shape (M,)  — raw RESP signal
        spo2_chunk  : np.ndarray, shape (K,)  — SpO₂ plethysmograph or numerics
        timestamp   : datetime, optional       — wall-clock time of end of chunk

        Returns
        -------
        TwinState — updated state (also accessible via self.state)
        """
        t0 = time.perf_counter()
        self.state.timestamp = timestamp or datetime.utcnow()
        self.state.elapsed_s = (
            self.state.timestamp - self.state.session_start
        ).total_seconds()

        # ── 1. ECG processing ─────────────────────────────────────────────
        try:
            self.state.ecg = self.ecg_proc.process(ecg_chunk)
            self.state.ecg_artifact = self.state.ecg.signal_quality < 0.4
            # Update display buffer
            n = min(len(ecg_chunk), len(self.state.ecg_buffer))
            self.state.ecg_buffer = np.roll(self.state.ecg_buffer, -n)
            self.state.ecg_buffer[-n:] = ecg_chunk[:n]
        except Exception as e:
            logger.warning(f"ECG processing error: {e}")
            self.state.ecg_artifact = True

        # ── 2. Respiratory processing ─────────────────────────────────────
        try:
            self.state.respiratory = self.resp_proc.process(resp_chunk)
            self.state.resp_artifact = self.state.respiratory.signal_quality < 0.4
            n = min(len(resp_chunk), len(self.state.resp_buffer))
            self.state.resp_buffer = np.roll(self.state.resp_buffer, -n)
            self.state.resp_buffer[-n:] = resp_chunk[:n]
        except Exception as e:
            logger.warning(f"RESP processing error: {e}")
            self.state.resp_artifact = True

        # ── 3. SpO₂ processing ────────────────────────────────────────────
        try:
            self.state.spo2 = self.spo2_proc.process(spo2_chunk)
            self.state.spo2_artifact = self.state.spo2.signal_quality < 0.4
            n = min(len(spo2_chunk), len(self.state.spo2_buffer))
            self.state.spo2_buffer = np.roll(self.state.spo2_buffer, -n)
            self.state.spo2_buffer[-n:] = spo2_chunk[:n]
        except Exception as e:
            logger.warning(f"SpO₂ processing error: {e}")
            self.state.spo2_artifact = True

        # ── 4. Coupling analysis ──────────────────────────────────────────
        try:
            self.state.coupling.rsa_amplitude, self.state.coupling.rsa_coherence = (
                self.rsa_analyzer.compute(
                    self.state.ecg.rr_intervals,
                    self.state.respiratory.dominant_frequency_hz,
                )
            )
            self.state.coupling.xcorr_peak, self.state.coupling.xcorr_lag_s, self.state.coupling.xcorr_series = (
                self.xcorr_analyzer.compute(
                    list(self.state.trend_hr),
                    list(self.state.trend_rr),
                )
            )
            self.state.coupling.psi, self.state.coupling.phase_difference, self.state.coupling.n_ratio = (
                self.phase_sync.compute(
                    self.state.ecg_buffer,
                    self.state.resp_buffer,
                )
            )
            hr_series  = list(self.state.trend_hr)
            spo2_series = list(self.state.trend_spo2)
            self.state.coupling.hypoxia_hr_response, self.state.coupling.blunted_response = (
                self.hypoxia_mon.evaluate(hr_series, spo2_series)
            )
        except Exception as e:
            logger.warning(f"Coupling analysis error: {e}")

        # ── 5. Derived indices ────────────────────────────────────────────
        self.state.indices.o2_delivery_index = compute_o2_delivery(
            spo2=self.state.spo2.spo2,
            hr=self.state.ecg.heart_rate,
            hr_baseline=self.cfg.oxygen.hr_baseline,
            sv_adjustment=self.cfg.oxygen.sv_adjustment,
            hrv_rmssd=self.state.ecg.rmssd,
        )
        self.state.indices.stability_index = compute_stability(
            rmssd=self.state.ecg.rmssd,
            psi=self.state.coupling.psi,
            xcorr_peak=self.state.coupling.xcorr_peak,
            cfg=self.cfg.stability,
        )
        self.state.indices.stress_load = compute_stress(
            hr=self.state.ecg.heart_rate,
            rr=self.state.respiratory.respiratory_rate,
            spo2=self.state.spo2.spo2,
            rmssd=self.state.ecg.rmssd,
            cfg=self.cfg.stress,
        )

        # ── 6. Trend update & prediction ─────────────────────────────────
        self.state.update_trends()

        (
            self.state.indices.hr_slope,
            self.state.indices.rr_slope,
            self.state.indices.spo2_slope,
            self.state.indices.o2_slope,
            self.state.indices.eta_spo2_critical_s,
            self.state.indices.eta_hr_critical_s,
            self.state.indices.eta_o2_critical_s,
        ) = self.predictor.predict(self.state)

        # ── 7. Alert evaluation ───────────────────────────────────────────
        self.state.active_alerts = self.alert_engine.evaluate(self.state)

        # ── 8. Bookkeeping ────────────────────────────────────────────────
        self._cycle_count += 1
        dt = time.perf_counter() - t0
        if self._cycle_count % 100 == 0:
            logger.debug(f"Cycle {self._cycle_count} | processing time {dt*1000:.1f}ms")

        return self.state

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
    
    def set_clinical_context(self, profile: dict):
        """Injects the static EHR data into the live math engine."""
        self.clinical_profile = profile

    def apply_metabolic_penalty(self, raw_o2_index: float) -> float:
        """Adjusts the O2 Delivery Index based on real patient labs and demographics."""
        penalty = 0.0

        if hasattr(self, 'clinical_profile'):
            p = self.clinical_profile
            
            # 1. Age Penalty (Physiological reserve drops slightly with age)
            if isinstance(p.get("age"), int) and p["age"] > 50:
                penalty += (p["age"] - 50) * 0.002

            # 2. Lactate Penalty (Normal < 2.0. High lactate = severe tissue hypoxia)
            lactate = p.get("latest_lactate")
            if isinstance(lactate, float) and lactate > 2.0:
                penalty += (lactate - 2.0) * 0.08  # Aggressive penalty for lactic acidosis

            # 3. pH Penalty (Normal 7.35 - 7.45. Drop in pH = metabolic failure)
            ph = p.get("latest_ph")
            if isinstance(ph, float) and ph < 7.35:
                penalty += (7.35 - ph) * 0.6       # Massive penalty for acidemia

        # Ensure the index doesn't drop below 0
        adjusted_index = max(0.0, raw_o2_index - penalty)
        return adjusted_index