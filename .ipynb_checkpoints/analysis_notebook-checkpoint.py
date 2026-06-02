"""
analysis_notebook.ipynb — Jupyter Notebook for Cardiopulmonary Analysis

This notebook demonstrates:
1. Engine initialization and synthetic data generation
2. Real-time physiological metrics extraction
3. HRV time-domain and frequency-domain analysis
4. Coupling metrics (RSA, PSI, cross-correlation)
5. Derived clinical indices
6. Alert detection and prediction
7. Comprehensive visualization

Instructions:
- Run cells in order with Shift+Enter
- Modify parameters to explore different scenarios
"""

# Cell 1: Setup and Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.twin_engine import TwinEngine
from src.core.config import load_config

print("✓ Imports successful")

# Cell 2: Initialize Engine
cfg = load_config()
engine = TwinEngine(cfg)
engine.start_session(subject_id="ANALYSIS_001", stay_id="SESSION_001")

print("✓ TwinEngine initialized")
print(f"  Config: {cfg}")

# Cell 3: Generate 5-Minute Synthetic Session
duration_s = 300  # 5 minutes
cycles = int(duration_s / 0.25)  # 250ms chunks

states_history = []
base_hr = 72.0
base_rr = 15.0
base_spo2 = 97.0

print(f"Generating {duration_s}s session ({cycles} cycles)...")

for cycle in range(cycles):
    t = cycle * 0.25
    
    # Modulate vitals
    hr_mod = 5.0 * np.sin(2 * np.pi * t / 60.0)
    rr_mod = 2.0 * np.sin(2 * np.pi * t / 45.0)
    spo2_mod = 1.5 * np.sin(2 * np.pi * t / 90.0)
    
    # ECG waveform
    fs_ecg = int(cfg.ecg.sampling_rate)
    n_ecg = int(fs_ecg * 0.25)
    t_ecg = np.linspace(t, t + 0.25, n_ecg)
    ecg_hr = base_hr + hr_mod
    phase_ecg = 2 * np.pi * ecg_hr / 60.0 * t_ecg
    ecg_chunk = (
        1.0 * np.sin(phase_ecg) +
        0.3 * np.sin(2 * phase_ecg) +
        0.05 * np.random.randn(n_ecg)
    )
    
    # Respiratory waveform
    fs_resp = int(cfg.respiratory.sampling_rate)
    n_resp = int(fs_resp * 0.25)
    t_resp = np.linspace(t, t + 0.25, n_resp)
    resp_rate = base_rr + rr_mod
    phase_resp = 2 * np.pi * resp_rate / 60.0 * t_resp
    resp_chunk = (
        np.sin(phase_resp) +
        0.03 * np.random.randn(n_resp)
    )
    
    # SpO2
    spo2_value = base_spo2 + spo2_mod + 0.1 * np.random.randn()
    spo2_chunk = np.array([np.clip(spo2_value, 50.0, 100.0)])
    
    # Process
    state = engine.ingest(ecg_chunk, resp_chunk, spo2_chunk)
    states_history.append(state)

print(f"✓ Generated {len(states_history)} states")

# Cell 4: Extract Time Series Data
df_data = []
for i, state in enumerate(states_history):
    df_data.append({
        'cycle': i,
        'time_s': state.elapsed_s,
        'hr': state.ecg.heart_rate,
        'rr': state.respiratory.respiratory_rate,
        'spo2': state.spo2.spo2,
        'rmssd': state.ecg.rmssd,
        'sdnn': state.ecg.sdnn,
        'lf_power': state.ecg.lf_power,
        'hf_power': state.ecg.hf_power,
        'lf_hf_ratio': state.ecg.lf_hf_ratio,
        'o2_delivery': state.indices.o2_delivery_index,
        'stability': state.indices.stability_index,
        'stress_load': state.indices.stress_load,
        'rsa_amp': state.coupling.rsa_amplitude,
        'psi': state.coupling.psi,
        'xcorr_peak': state.coupling.xcorr_peak,
        'n_alerts': len(state.active_alerts),
    })

df = pd.DataFrame(df_data)
print(f"✓ Created dataframe with {len(df)} rows")
print("\nDataframe summary:")
print(df[['time_s', 'hr', 'rr', 'spo2', 'o2_delivery', 'stress_load']].describe())

# Cell 5: Plot Vital Signs Trends
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Heart Rate', 'Respiratory Rate', 'SpO₂'),
    vertical_spacing=0.1
)

fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['hr'], name='HR', line=dict(color='#ff6644')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['rr'], name='RR', line=dict(color='#4488ff')),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['spo2'], name='SpO₂', line=dict(color='#44cc88')),
    row=3, col=1
)

fig.update_layout(height=600, title='Vital Signs Over Time', showlegend=True)
fig.update_yaxes(title_text='bpm', row=1, col=1)
fig.update_yaxes(title_text='br/min', row=2, col=1)
fig.update_yaxes(title_text='%', row=3, col=1)
fig.update_xaxes(title_text='Time (s)', row=3, col=1)
fig.show()

# Cell 6: HRV Analysis
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Time-Domain HRV (RMSSD, SDNN)', 'LF/HF Ratio'),
    vertical_spacing=0.15
)

fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['rmssd'], name='RMSSD', line=dict(color='#cc88ff')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['sdnn'], name='SDNN', line=dict(color='#88ccff')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['lf_hf_ratio'], name='LF/HF', line=dict(color='#ffaa44')),
    row=2, col=1
)

fig.update_layout(height=600, title='Heart Rate Variability (HRV) Analysis', showlegend=True)
fig.update_yaxes(title_text='ms', row=1, col=1)
fig.update_yaxes(title_text='ratio', row=2, col=1)
fig.update_xaxes(title_text='Time (s)', row=2, col=1)
fig.show()

# Cell 7: Coupling Metrics
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('RSA Amplitude', 'Phase Synchronisation Index (PSI)', 'Cross-Correlation Peak'),
    vertical_spacing=0.15
)

fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['rsa_amp'], name='RSA', line=dict(color='#ff6644', width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['psi'], name='PSI', line=dict(color='#44cc88', width=2)),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['xcorr_peak'], name='XCorr', line=dict(color='#4488ff', width=2)),
    row=3, col=1
)

fig.update_layout(height=600, title='Heart-Lung Coupling Metrics', showlegend=True)
fig.update_yaxes(title_text='bpm', row=1, col=1)
fig.update_yaxes(title_text='0-1', row=2, col=1)
fig.update_yaxes(title_text='-1 to 1', row=3, col=1)
fig.update_xaxes(title_text='Time (s)', row=3, col=1)
fig.show()

# Cell 8: Derived Indices & Stress
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('O₂ Delivery Index', 'Stability Index', 'Stress Load'),
    vertical_spacing=0.15
)

fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['o2_delivery'], name='O₂D', line=dict(color='#44cc88', width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['stability'], name='Stability', line=dict(color='#88ccff', width=2)),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=df['time_s'], y=df['stress_load'], name='Stress', line=dict(color='#ff8844', width=2)),
    row=3, col=1
)

fig.update_layout(height=600, title='Clinical Derived Indices', showlegend=True)
fig.update_yaxes(title_text='index', row=1, col=1)
fig.update_yaxes(title_text='0-1', row=2, col=1)
fig.update_yaxes(title_text='0-100', row=3, col=1)
fig.update_xaxes(title_text='Time (s)', row=3, col=1)
fig.show()

# Cell 9: Alert Summary
final_state = states_history[-1]
print("=" * 70)
print("FINAL STATE SUMMARY")
print("=" * 70)
print(f"Elapsed Time: {final_state.elapsed_s:.1f} seconds")
print(f"\nVitals:")
print(f"  Heart Rate: {final_state.ecg.heart_rate:.1f} bpm")
print(f"  Respiratory Rate: {final_state.respiratory.respiratory_rate:.1f} br/min")
print(f"  SpO₂: {final_state.spo2.spo2:.1f}%")
print(f"\nHRV:")
print(f"  RMSSD: {final_state.ecg.rmssd:.2f} ms (normal: >20 ms)")
print(f"  SDNN: {final_state.ecg.sdnn:.2f} ms (normal: >50 ms)")
print(f"  LF/HF Ratio: {final_state.ecg.lf_hf_ratio:.2f} (normal: 0.5-2.0)")
print(f"\nCoupling:")
print(f"  RSA Amplitude: {final_state.coupling.rsa_amplitude:.2f} bpm")
print(f"  Phase Sync Index: {final_state.coupling.psi:.3f} (0-1)")
print(f"  HR-RR Cross-Corr: {final_state.coupling.xcorr_peak:.3f}")
print(f"\nIndices:")
print(f"  O₂ Delivery Index: {final_state.indices.o2_delivery_index:.3f}")
print(f"  Stability Index: {final_state.indices.stability_index:.3f} (0-1)")
print(f"  Stress Load: {final_state.indices.stress_load:.1f}/100")
print(f"\nAlerts:")
if final_state.active_alerts:
    for alert in final_state.active_alerts:
        print(f"  [{alert.severity.upper()}] {alert.message}")
else:
    print("  None (patient stable)")
print("=" * 70)

# Cell 10: Prediction Analysis
print("\nPredictive Analytics:")
print(f"  HR Slope: {final_state.indices.hr_slope:.3f} bpm/min")
print(f"  RR Slope: {final_state.indices.rr_slope:.3f} br/min/min")
print(f"  SpO₂ Slope: {final_state.indices.spo2_slope:.4f} %/min")
print(f"  ETA to SpO₂ Critical: {final_state.indices.eta_spo2_critical_s}")
print(f"  ETA to HR Critical: {final_state.indices.eta_hr_critical_s}")
print(f"  ETA to O₂D Critical: {final_state.indices.eta_o2_critical_s}")

print("\n✓ Analysis complete! Export this notebook as HTML to share.")
