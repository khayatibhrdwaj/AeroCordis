"""
src/prediction/trend_predictor.py
───────────────────────────────────
Predictive analytics for the digital twin.

Two-stage prediction:
  1. Linear extrapolation — fast, always available, interpretable
  2. LSTM model — trained online, more accurate for non-linear trajectories
  3. Combined output — weighted blend (configurable)

Outputs per cycle:
  - Slope for HR, RR, SpO₂, O₂ delivery (units per minute)
  - ETA (seconds) to each clinical threshold
    → None if no alert predicted in the configured horizon
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List
from collections import deque
from loguru import logger

from ..core.config import TwinConfig

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, Exception) as e:
    HAS_TORCH = False
    logger.warning(f"PyTorch initialization failed ({e}) — LSTM predictor disabled, using linear only.")


# ── LSTM Model definition ─────────────────────────────────────────────────────
# Defined conditionally to avoid NameError when torch is absent.

class _LSTMStub:
    """Stub used when PyTorch is not installed."""
    parameters: None = None

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("PyTorch is required for LSTM. Install with: pip install torch")

    def train(self, mode: bool = True) -> "_LSTMStub":  # noqa: ARG002
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError("PyTorch is required for LSTM.")


if HAS_TORCH:
    class CardiopulmonaryLSTM(nn.Module):  # type: ignore[misc]
        """
        Lightweight LSTM: takes [HR, RR, SpO₂, O₂D, HRV, Stress] sequences
        and predicts the next `horizon` steps for each channel.
        """
        def __init__(
            self,
            input_size: int = 6,
            hidden_size: int = 64,
            num_layers: int = 2,
            output_size: int = 6,
            horizon: int = 60,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, output_size * horizon)
            self.output_size = output_size
            self.horizon = horizon

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out.view(-1, self.horizon, self.output_size)

else:
    CardiopulmonaryLSTM = _LSTMStub  # type: ignore[misc,assignment]


# ── Main predictor ────────────────────────────────────────────────────────────

class TrendPredictor:

    # Channel indices in the feature vector
    _IDX_HR   = 0
    _IDX_RR   = 1
    _IDX_SPO2 = 2
    _IDX_O2D  = 3
    _IDX_HRV  = 4
    _IDX_STR  = 5

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self.pred_cfg = cfg.prediction

        self._lin_win = int(cfg.prediction.linear_window_s)
        self._horizon = cfg.prediction.lstm.sequence_length

        # Online training buffer
        seq_len = cfg.prediction.lstm.sequence_length
        self._feature_buf: deque = deque(maxlen=seq_len * 4)

        # LSTM (optional)
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_step = 0

        if HAS_TORCH and cfg.prediction.method in ("lstm", "combined"):
            self._build_lstm()

    def _build_lstm(self) -> None:
        if not HAS_TORCH:
            return
        cfg = self.pred_cfg.lstm
        self._model = CardiopulmonaryLSTM(
            input_size=6,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            output_size=6,
            horizon=60,
            dropout=cfg.dropout,
        )
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),  # type: ignore[union-attr]
            lr=cfg.learning_rate,
        )
        self._criterion = nn.MSELoss()
        logger.info("LSTM predictor initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, state) -> Tuple:
        """
        Returns:
          hr_slope, rr_slope, spo2_slope, o2_slope,
          eta_spo2_s, eta_hr_s, eta_o2_s
        """
        # Build feature vector for current timestep
        feat = np.array([
            state.ecg.heart_rate,
            state.respiratory.respiratory_rate,
            state.spo2.spo2,
            state.indices.o2_delivery_index,
            state.ecg.rmssd,
            state.indices.stress_load,
        ], dtype=float)
        self._feature_buf.append(feat)

        # 1. Linear slopes (always computed)
        hr_slope   = self._linear_slope(list(state.trend_hr),   self._lin_win)
        rr_slope   = self._linear_slope(list(state.trend_rr),   self._lin_win)
        spo2_slope = self._linear_slope(list(state.trend_spo2), self._lin_win)
        o2_slope   = self._linear_slope(list(state.trend_o2d),  self._lin_win)

        # 2. LSTM Prediction (if enabled)
        lstm_pred = None
        if (
            HAS_TORCH
            and self._model is not None
            and len(self._feature_buf) >= self.pred_cfg.lstm.sequence_length
        ):
            lstm_pred = self._generate_lstm_prediction()

        # 3. ETA to thresholds
        method = self.cfg.prediction.method
        
        if method == "lstm" and lstm_pred is not None:
            # Use LSTM trajectory for ETA
            eta_spo2 = self._eta_from_trajectory(lstm_pred[:, self._IDX_SPO2], self.cfg.alerts.spo2.critical, "below")
            eta_hr = self._eta_from_trajectory_two_sided(
                lstm_pred[:, self._IDX_HR], 
                self.cfg.alerts.heart_rate.critical_low, 
                self.cfg.alerts.heart_rate.critical_high
            )
            eta_o2 = self._eta_from_trajectory(lstm_pred[:, self._IDX_O2D], self.cfg.oxygen.critical_threshold, "below")
        else:
            # Default to linear extrapolation
            eta_spo2 = self._eta_to_threshold(
                current=state.spo2.spo2,
                slope=spo2_slope,
                threshold=self.cfg.alerts.spo2.critical,
                direction="below",
                horizon_s=self.cfg.alerts.spo2.prediction_horizon_s,
            )
            eta_hr = self._eta_to_threshold_two_sided(
                current=state.ecg.heart_rate,
                slope=hr_slope,
                lo=self.cfg.alerts.heart_rate.critical_low,
                hi=self.cfg.alerts.heart_rate.critical_high,
                horizon_s=self.cfg.alerts.heart_rate.prediction_horizon_s,
            )
            eta_o2 = self._eta_to_threshold(
                current=state.indices.o2_delivery_index,
                slope=o2_slope,
                threshold=self.cfg.oxygen.critical_threshold,
                direction="below",
                horizon_s=self.cfg.alerts.spo2.prediction_horizon_s,
            )

        # Online LSTM training (every 10 steps if enough data)
        if (
            HAS_TORCH
            and self._model is not None
            and len(self._feature_buf) >= self.pred_cfg.lstm.sequence_length + 1
        ):
            self._train_step += 1
            if self._train_step % 10 == 0:
                self._online_train()

        return (
            round(hr_slope,   3),
            round(rr_slope,   3),
            round(spo2_slope, 4),
            round(o2_slope,   4),
            eta_spo2,
            eta_hr,
            eta_o2,
        )

    def _generate_lstm_prediction(self) -> Optional[np.ndarray]:
        """Generate future trajectory using the LSTM model."""
        if not HAS_TORCH or self._model is None:
            return None
            
        try:
            seq_len = self.pred_cfg.lstm.sequence_length
            buf = np.array(list(self._feature_buf), dtype=np.float32)[-seq_len:]
            
            # Normalise
            mean = buf.mean(axis=0, keepdims=True)
            std = buf.std(axis=0, keepdims=True) + 1e-8
            buf_norm = (buf - mean) / std
            
            X = torch.tensor(buf_norm[np.newaxis], dtype=torch.float32)
            
            self._model.eval()
            with torch.no_grad():
                pred_norm = self._model(X).squeeze(0).cpu().numpy() # (horizon, output_size)
                
            # Denormalise
            pred = pred_norm * std + mean
            return pred
        except Exception as e:
            logger.warning(f"LSTM prediction error: {e}")
            return None

    @staticmethod
    def _eta_from_trajectory(
        trajectory: np.ndarray, 
        threshold: Optional[float], 
        direction: str
    ) -> Optional[float]:
        """Find first index in trajectory that crosses threshold."""
        if threshold is None:
            return None
            
        if direction == "below":
            indices = np.where(trajectory <= threshold)[0]
        else:
            indices = np.where(trajectory >= threshold)[0]
            
        if len(indices) > 0:
            return float(indices[0])
        return None

    @staticmethod
    def _eta_from_trajectory_two_sided(
        trajectory: np.ndarray, 
        lo: Optional[float], 
        hi: Optional[float]
    ) -> Optional[float]:
        """Find first index in trajectory that crosses either low or high threshold."""
        eta_lo = TrendPredictor._eta_from_trajectory(trajectory, lo, "below")
        eta_hi = TrendPredictor._eta_from_trajectory(trajectory, hi, "above")
        etas = [e for e in [eta_lo, eta_hi] if e is not None]
        return min(etas) if etas else None

    # ── Linear slope (per minute) ─────────────────────────────────────────────

    @staticmethod
    def _linear_slope(series: List[float], window: int) -> float:
        """Slope in units per minute via least-squares over the last `window` seconds."""
        if len(series) < 5:
            return 0.0
        y = np.array(series[-window:], dtype=float)
        n = len(y)
        x = np.arange(n, dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom  = np.sum((x - x_mean) ** 2)
        if denom < 1e-9:
            return 0.0
        slope_per_second = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        return slope_per_second * 60.0   # Convert to per minute

    # ── ETA to threshold ──────────────────────────────────────────────────────

    @staticmethod
    def _eta_to_threshold(
        current: float,
        slope: float,        # units per minute
        threshold: Optional[float],
        direction: str,      # "below" | "above"
        horizon_s: float,
    ) -> Optional[float]:
        """
        Linear extrapolation to predict seconds until `current` crosses `threshold`.
        Returns None if threshold is None or not predicted within horizon.
        """
        if threshold is None:
            return None

        if direction == "below":
            if current <= threshold:
                return 0.0
            if slope >= 0:
                return None
            delta = current - threshold
        else:
            if current >= threshold:
                return 0.0
            if slope <= 0:
                return None
            delta = threshold - current

        slope_per_s = abs(slope) / 60.0
        if slope_per_s < 1e-9:
            return None

        eta_s = delta / slope_per_s
        return round(eta_s, 0) if eta_s <= horizon_s else None

    @staticmethod
    def _eta_to_threshold_two_sided(
        current: float,
        slope: float,
        lo: Optional[float],
        hi: Optional[float],
        horizon_s: float,
    ) -> Optional[float]:
        """Returns the earlier of low / high breach ETAs."""
        eta_lo = TrendPredictor._eta_to_threshold(current, slope, lo, "below", horizon_s)
        eta_hi = TrendPredictor._eta_to_threshold(current, slope, hi, "above", horizon_s)
        etas = [e for e in [eta_lo, eta_hi] if e is not None]
        return min(etas) if etas else None

    # ── Online LSTM training ──────────────────────────────────────────────────

    def _online_train(self) -> None:
        """Train LSTM on the most recent sequence buffer (online learning)."""
        if not HAS_TORCH or self._model is None or self._optimizer is None or self._criterion is None:
            return

        seq_len = self.pred_cfg.lstm.sequence_length
        buf = np.array(list(self._feature_buf), dtype=np.float32)
        if len(buf) < seq_len + 1:
            return

        # Normalise
        buf_mean = buf.mean(axis=0, keepdims=True)
        buf_std  = buf.std(axis=0, keepdims=True) + 1e-8
        buf_norm = (buf - buf_mean) / buf_std

        X = torch.tensor(buf_norm[:seq_len][np.newaxis], dtype=torch.float32)

        horizon = min(60, len(buf) - seq_len)
        y_raw = buf_norm[seq_len:seq_len + horizon]
        if len(y_raw) < 1:
            return
        if len(y_raw) < 60:
            y_raw = np.pad(y_raw, ((0, 60 - len(y_raw)), (0, 0)), mode="edge")
        Y = torch.tensor(y_raw[np.newaxis], dtype=torch.float32)

        self._model.train()  # type: ignore[union-attr]
        self._optimizer.zero_grad()
        pred = self._model(X)  # type: ignore[operator]
        loss = self._criterion(pred, Y)
        loss.backward()
        self._optimizer.step()

        if self._train_step % 100 == 0:
            logger.debug(f"LSTM step {self._train_step} | loss={loss.item():.4f}")