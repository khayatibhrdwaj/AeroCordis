#!/usr/bin/env python3
"""
demo_engine.py
──────────────
Complete working demo of the Cardiopulmonary Digital Twin.

Run:
    python demo_engine.py --mode synthetic --cycles 60
    python demo_engine.py --mode mimic --subject 10000032 --stay 30000426 --chunks 10
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.twin_engine import TwinEngine
from src.core.config import load_config


def synthetic_demo(cycles: int = 120):
    """
    Run demo with synthetic waveforms.
    
    Parameters
    ----------
    cycles : int
        Number of 250ms cycles to simulate (120 = 30 seconds)
    """
    print("\n" + "=" * 80)
    print("CARDIOPULMONARY DIGITAL TWIN — Synthetic Data Demo")
    print("=" * 80)
    print(f"Duration: {cycles * 0.25:.1f} seconds ({cycles} cycles @ 250ms each)\n")
    
    cfg = load_config()
    engine = TwinEngine(cfg)
    engine.start_session(subject_id="SYNTHETIC_001", stay_id="DEMO_STAY_001")
    
    # Synthetic parameters: slow trends
    base_hr = 72.0
    base_rr = 15.0
    base_spo2 = 97.0
    
    print(f"{'Cycle':>5} {'HR':>7} {'RR':>7} {'SpO₂':>7} {'O₂D':>7} {'Stress':>8} {'Alerts':>7} {'Time (s)':>8}")
    print("-" * 80)
    
    alerts_triggered = []
    
    for cycle in range(cycles):
        t = cycle * 0.25  # seconds
        
        # Modulate vitals with slow sine waves
        hr_mod = 5.0 * np.sin(2 * np.pi * t / 60.0)  # ±5 bpm over 60s
        rr_mod = 2.0 * np.sin(2 * np.pi * t / 45.0)  # ±2 br/min over 45s
        spo2_mod = 1.5 * np.sin(2 * np.pi * t / 90.0)  # ±1.5% over 90s
        
        # Cardiac waveform (realistic synthetic ECG with sharp R peaks)
        fs_ecg = int(cfg.ecg.sampling_rate)
        n_ecg = int(fs_ecg * 0.25)
        t_ecg = np.linspace(0, 0.25, n_ecg)

        ecg_hr = base_hr + hr_mod
        beat_interval = 60.0 / ecg_hr

        ecg_chunk = np.zeros(n_ecg)

    # Add synthetic R peaks
        for beat_time in np.arange(0, 0.25, beat_interval):
            peak_idx = int(beat_time * fs_ecg)
    
        if peak_idx < n_ecg:
            width = int(0.02 * fs_ecg)  # narrow QRS width
        
            for i in range(-width, width):
                idx = peak_idx + i
                if 0 <= idx < n_ecg:
                    ecg_chunk[idx] += np.exp(-(i**2)/(2*(width/3)**2)) * 2.0

    # baseline drift + noise
        ecg_chunk += 0.1 * np.sin(2*np.pi*0.5*t_ecg)
        ecg_chunk += 0.03 * np.random.randn(n_ecg)
        
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
        
        # SpO₂ (1 Hz numerics)
        spo2_value = base_spo2 + spo2_mod + 0.1 * np.random.randn()
        spo2_chunk = np.array([np.clip(spo2_value, 50.0, 100.0)])
        
        # Process
        state = engine.ingest(ecg_chunk, resp_chunk, spo2_chunk)
        
        # Print every 10 cycles (2.5 seconds)
        if cycle % 10 == 0:
            print(f"{cycle:5d} {state.ecg.heart_rate:7.1f} {state.respiratory.respiratory_rate:7.1f} "
                  f"{state.spo2.spo2:7.1f} {state.indices.o2_delivery_index:7.3f} "
                  f"{state.indices.stress_load:8.1f} {len(state.active_alerts):7d} {t:8.2f}")
            
            # Collect alerts
            for alert in state.active_alerts:
                if alert not in alerts_triggered:
                    alerts_triggered.append(alert)
                    print(f"  ⚠️  [{alert.severity.upper()}] {alert.message}")
    
    print("-" * 80)
    print(f"\n✓ Demo completed successfully!")
    print(f"  Cycles: {cycles}")
    print(f"  Duration: {cycles * 0.25:.1f} seconds")
    print(f"  Final HR: {state.ecg.heart_rate:.1f} bpm")
    print(f"  Final RR: {state.respiratory.respiratory_rate:.1f} br/min")
    print(f"  Final SpO₂: {state.spo2.spo2:.1f}%")
    print(f"  O₂ Delivery Index: {state.indices.o2_delivery_index:.3f}")
    print(f"  Stability Index: {state.indices.stability_index:.3f}")
    print(f"  Stress Load: {state.indices.stress_load:.1f}/100")
    print(f"  Total Alerts Triggered: {len(alerts_triggered)}")
    
    if state.active_alerts:
        print(f"\n  Active Alerts:")
        for alert in state.active_alerts[:3]:
            print(f"    • [{alert.severity}] {alert.message}")
    
    print("\n" + "=" * 80)


def mimic_demo(subject_id: str, stay_id: str, max_chunks: int = 10):
    """
    Run demo with real MIMIC-IV data.
    
    Parameters
    ----------
    subject_id : str
        MIMIC-IV subject ID
    stay_id : str
        ICU stay ID
    max_chunks : int
        Maximum chunks to load
    """
    print("\n" + "=" * 80)
    print("CARDIOPULMONARY DIGITAL TWIN — MIMIC-IV Data Demo")
    print("=" * 80)
    print(f"Subject: {subject_id} | Stay: {stay_id} | Max chunks: {max_chunks}\n")
    
    try:
        from src.data.mimic_loader import MIMICLoader
    except ImportError as e:
        print(f"✗ MIMIC loader not available: {e}")
        print("  Try: pip install wfdb")
        return
    
    cfg = load_config()
    loader = MIMICLoader(cfg)
    
    # Authenticate
    if not loader.authenticate():
        print("✗ MIMIC authentication failed")
        return
    
    # Initialize engine
    engine = TwinEngine(cfg)
    engine.start_session(subject_id=subject_id, stay_id=stay_id)
    
    print(f"{'Chunk':>5} {'HR':>7} {'RR':>7} {'SpO₂':>7} {'O₂D':>7} {'Stress':>8} {'Alerts':>7}")
    print("-" * 80)
    
    chunk_count = 0
    alerts_total = 0
    
    try:
        for chunk in loader.stream_waveforms(subject_id, stay_id, max_chunks=max_chunks):
            state = engine.ingest(chunk.ecg, chunk.resp, chunk.spo2, chunk.timestamp)
            chunk_count += 1
            alerts_total += len(state.active_alerts)
            
            print(f"{chunk_count:5d} {state.ecg.heart_rate:7.1f} {state.respiratory.respiratory_rate:7.1f} "
                  f"{state.spo2.spo2:7.1f} {state.indices.o2_delivery_index:7.3f} "
                  f"{state.indices.stress_load:8.1f} {len(state.active_alerts):7d}")
            
            if state.active_alerts:
                for alert in state.active_alerts[:1]:
                    print(f"  ⚠️  [{alert.severity.upper()}] {alert.message}")
    
    except Exception as e:
        print(f"\n✗ MIMIC streaming error: {e}")
        if chunk_count == 0:
            print("  Could not load any chunks. Check subject/stay IDs and credentials.")
            return
    
    print("-" * 80)
    if chunk_count > 0:
        print(f"\n✓ MIMIC demo completed!")
        print(f"  Chunks loaded: {chunk_count}")
        print(f"  Total alerts: {alerts_total}")
        print(f"  Final metrics:")
        print(f"    HR: {state.ecg.heart_rate:.1f} bpm")
        print(f"    RR: {state.respiratory.respiratory_rate:.1f} br/min")
        print(f"    SpO₂: {state.spo2.spo2:.1f}%")
        print(f"    Stress: {state.indices.stress_load:.1f}/100")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Cardiopulmonary Digital Twin Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic 30-second demo
  python demo_engine.py --mode synthetic --cycles 120

  # MIMIC real data (requires credentials)
  python demo_engine.py --mode mimic --subject 10000032 --stay 30000426 --chunks 20
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["synthetic", "mimic"],
        default="synthetic",
        help="Demo mode (default: synthetic)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=120,
        help="For synthetic mode: number of 250ms cycles (default: 120 = 30 seconds)"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="10000032",
        help="For MIMIC mode: subject ID"
    )
    parser.add_argument(
        "--stay",
        type=str,
        default="30000426",
        help="For MIMIC mode: stay ID"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=10,
        help="For MIMIC mode: max chunks to load"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "synthetic":
            synthetic_demo(cycles=args.cycles)
        else:
            mimic_demo(
                subject_id=args.subject,
                stay_id=args.stay,
                max_chunks=args.chunks
            )
    except KeyboardInterrupt:
        print("\n\n✗ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Demo failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
