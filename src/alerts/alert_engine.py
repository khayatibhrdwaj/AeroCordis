"""
src/alerts/alert_engine.py
───────────────────────────
Alert Engine: evaluates the current TwinState against all configured
thresholds and returns a list of AlertEvent objects.

Severity levels
  critical  — immediate clinical action required
  warning   — monitor closely, prepare intervention
  info      — informational / trend-based early warning
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from loguru import logger

from ..core.config import TwinConfig, AlertThreshold
from ..core.twin_state import TwinState, AlertEvent


class AlertEngine:

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg

    # ── Public ────────────────────────────────────────────────────────────────

    def evaluate(self, state: TwinState) -> List[AlertEvent]:
        """
        Evaluate all alert conditions on the current TwinState.

        Returns a list of AlertEvent objects (may be empty).
        Suppresses alerts on artifact-contaminated channels.
        """
        alerts: List[AlertEvent] = []
        ts = state.timestamp

        # ── SpO₂ ──────────────────────────────────────────────────────────
        if not state.spo2_artifact:
            alerts += self._check_low(
                value=state.spo2.spo2,
                channel="spo2",
                unit="%",
                critical_thr=self.cfg.alerts.spo2.critical,
                warning_thr=self.cfg.alerts.spo2.warning,
                eta_s=state.indices.eta_spo2_critical_s,
                ts=ts,
            )

        # ── Heart Rate ────────────────────────────────────────────────────
        if not state.ecg_artifact and state.ecg.heart_rate > 0:
            alerts += self._check_two_sided(
                value=state.ecg.heart_rate,
                channel="heart_rate",
                unit="bpm",
                thr=self.cfg.alerts.heart_rate,
                eta_s=state.indices.eta_hr_critical_s,
                ts=ts,
            )

        # ── Respiratory Rate ──────────────────────────────────────────────
        if not state.resp_artifact and state.respiratory.respiratory_rate > 0:
            alerts += self._check_two_sided(
                value=state.respiratory.respiratory_rate,
                channel="respiratory_rate",
                unit="br/min",
                thr=self.cfg.alerts.respiratory_rate,
                eta_s=None,
                ts=ts,
            )

        # ── O₂ Delivery Index ─────────────────────────────────────────────
        alerts += self._check_low(
            value=state.indices.o2_delivery_index,
            channel="o2_delivery",
            unit="index",
            critical_thr=self.cfg.alerts.o2_delivery.critical,
            warning_thr=self.cfg.alerts.o2_delivery.warning,
            eta_s=state.indices.eta_o2_critical_s,
            ts=ts,
        )

        # ── Apnea ─────────────────────────────────────────────────────────
        if state.respiratory.apnea_detected:
            alerts.append(AlertEvent(
                timestamp=ts,
                severity="critical",
                channel="respiratory_rate",
                message=f"Apnea detected — {state.respiratory.apnea_duration_s:.0f}s without breath",
                value=state.respiratory.apnea_duration_s,
                threshold=self.cfg.respiratory.apnea_threshold_s,
            ))

        # ── Blunted hypoxia response ───────────────────────────────────────
        if state.coupling.blunted_response:
            alerts.append(AlertEvent(
                timestamp=ts,
                severity="warning",
                channel="coupling",
                message=(
                    f"Blunted hypoxia response — observed HR rise "
                    f"{state.coupling.hypoxia_hr_response:.1f} bpm/% SpO₂ drop "
                    f"(expected ≥{self.cfg.alerts.hypoxia_response.expected_hr_rise_per_spo2_fall:.1f})"
                ),
                value=state.coupling.hypoxia_hr_response,
                threshold=self.cfg.alerts.hypoxia_response.expected_hr_rise_per_spo2_fall,
            ))

        # ── QTc prolongation ──────────────────────────────────────────────
        qtc = state.ecg.qtc_ms
        if qtc > 500:
            alerts.append(AlertEvent(
                timestamp=ts,
                severity="critical",
                channel="ecg_morphology",
                message=f"Critically prolonged QTc — {qtc:.0f} ms (>500 ms)",
                value=qtc,
                threshold=500.0,
            ))
        elif qtc > 460:
            alerts.append(AlertEvent(
                timestamp=ts,
                severity="warning",
                channel="ecg_morphology",
                message=f"Prolonged QTc — {qtc:.0f} ms (>460 ms)",
                value=qtc,
                threshold=460.0,
            ))

        # ── Predictive ETA alerts ─────────────────────────────────────────
        alerts += self._eta_alerts(state, ts)

        return alerts

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_low(
        self,
        value: float,
        channel: str,
        unit: str,
        critical_thr: Optional[float],
        warning_thr: Optional[float],
        eta_s: Optional[float],
        ts: datetime,
    ) -> List[AlertEvent]:
        events = []
        if critical_thr is not None and value <= critical_thr:
            events.append(AlertEvent(
                timestamp=ts,
                severity="critical",
                channel=channel,
                message=f"{channel} critically low — {value:.1f} {unit} (≤{critical_thr})",
                value=value,
                threshold=critical_thr,
                eta_s=eta_s,
            ))
        elif warning_thr is not None and value <= warning_thr:
            events.append(AlertEvent(
                timestamp=ts,
                severity="warning",
                channel=channel,
                message=f"{channel} low — {value:.1f} {unit} (≤{warning_thr})",
                value=value,
                threshold=warning_thr,
                eta_s=eta_s,
            ))
        return events

    def _check_two_sided(
        self,
        value: float,
        channel: str,
        unit: str,
        thr: AlertThreshold,
        eta_s: Optional[float],
        ts: datetime,
    ) -> List[AlertEvent]:
        events = []
        if thr.critical_high is not None and value >= thr.critical_high:
            events.append(AlertEvent(
                timestamp=ts, severity="critical", channel=channel,
                message=f"{channel} critically high — {value:.1f} {unit} (≥{thr.critical_high})",
                value=value, threshold=thr.critical_high, eta_s=eta_s,
            ))
        elif thr.warning_high is not None and value >= thr.warning_high:
            events.append(AlertEvent(
                timestamp=ts, severity="warning", channel=channel,
                message=f"{channel} high — {value:.1f} {unit} (≥{thr.warning_high})",
                value=value, threshold=thr.warning_high, eta_s=eta_s,
            ))
        elif thr.critical_low is not None and value <= thr.critical_low:
            events.append(AlertEvent(
                timestamp=ts, severity="critical", channel=channel,
                message=f"{channel} critically low — {value:.1f} {unit} (≤{thr.critical_low})",
                value=value, threshold=thr.critical_low, eta_s=eta_s,
            ))
        elif thr.warning_low is not None and value <= thr.warning_low:
            events.append(AlertEvent(
                timestamp=ts, severity="warning", channel=channel,
                message=f"{channel} low — {value:.1f} {unit} (≤{thr.warning_low})",
                value=value, threshold=thr.warning_low, eta_s=eta_s,
            ))
        return events

    def _eta_alerts(self, state: TwinState, ts: datetime) -> List[AlertEvent]:
        """Generate forward-looking (predictive) alerts from ETA estimates."""
        events = []
        horizon = self.cfg.alerts.spo2.prediction_horizon_s

        checks = [
            (state.indices.eta_spo2_critical_s, "spo2",
             f"SpO₂ predicted to reach critical threshold in {{eta:.0f}}s"),
            (state.indices.eta_hr_critical_s, "heart_rate",
             f"HR predicted to reach critical threshold in {{eta:.0f}}s"),
            (state.indices.eta_o2_critical_s, "o2_delivery",
             f"O₂ delivery predicted to reach critical threshold in {{eta:.0f}}s"),
        ]
        for eta, channel, msg_tpl in checks:
            if eta is not None and 0 < eta <= horizon:
                severity = "critical" if eta < 60 else "warning"
                events.append(AlertEvent(
                    timestamp=ts,
                    severity=severity,
                    channel=channel,
                    message=msg_tpl.format(eta=eta),
                    value=eta,
                    threshold=horizon,
                    eta_s=eta,
                ))
        return events
