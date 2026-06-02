"""
src/data/smartwatch_bridge.py
────────────────────────────────
API bridge for commercial wearables (Apple Watch, Garmin, WearOS).
Simulates BLE/HealthKit data ingestion for the Digital Twin.
"""
import time
import numpy as np

class SmartwatchBridge:
    def __init__(self, device_name="Apple Watch Series 9"):
        self.device_name = device_name
        self.is_connected = False
        self.baseline_hr = 72.0
        self.baseline_spo2 = 98.0
        
    def connect(self):
        """Simulates the BLE handshake and authentication."""
        time.sleep(1.5) # Simulate connection latency
        self.is_connected = True
        return True

    def stream_vitals(self, t: float, chunk_s: float, fs_ecg: int, fs_resp: int):
        """
        Generates waveform chunks mimicking optical wrist sensors.
        Wrist PPG is noisier than ICU chest leads.
        """
        if not self.is_connected:
            raise ConnectionError("Smartwatch not connected.")

        n_ecg = int(fs_ecg * chunk_s)
        n_resp = int(fs_resp * chunk_s)
        t_arr = np.linspace(t, t + chunk_s, n_ecg)
        
        # Add realistic "wrist movement" noise (baseline wander)
        movement_artifact = 0.15 * np.sin(2 * np.pi * 0.1 * t_arr) 
        
        # Heart Rate logic (drifts slightly over time)
        self.baseline_hr += np.random.normal(0, 0.5)
        self.baseline_hr = np.clip(self.baseline_hr, 60, 100)
        hr_hz = self.baseline_hr / 60.0
        
        # Simulated PPG waveform (looks different than strict ECG)
        ppg_sig = 0.8 * np.sin(2 * np.pi * hr_hz * t_arr)
        ppg_sig += 0.4 * np.sin(4 * np.pi * hr_hz * t_arr + 0.5)
        ppg_sig += movement_artifact + np.random.normal(0, 0.08, n_ecg)

        # Simulated Respiration (often derived from HR on watches)
        t_resp = np.linspace(t, t + chunk_s, n_resp)
        resp_hz = 0.25 + 0.02 * np.sin(t / 20)
        resp_sig = np.sin(2 * np.pi * resp_hz * t_resp) + np.random.normal(0, 0.05, n_resp)

        # SpO2 logic (Watch optical sensors sometimes drop data, simulated by variance)
        self.baseline_spo2 += np.random.normal(0, 0.1)
        self.baseline_spo2 = np.clip(self.baseline_spo2, 94.0, 100.0)
        spo2_val = np.array([self.baseline_spo2])

        return ppg_sig, resp_sig, spo2_val