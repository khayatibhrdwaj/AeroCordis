#!/usr/bin/env python3
"""
scripts/run_twin.py
────────────────────
CLI entry point: run the Cardiopulmonary Digital Twin on a
specific MIMIC-IV subject and stay.

Usage:
    python scripts/run_twin.py --subject-id 10000032 --stay-id 30000426
    python scripts/run_twin.py --subject-id 10000032 --demo
    python scripts/run_twin.py --subject-id 10000032 --max-chunks 100 --output results.csv
"""

from __future__ import annotations
import sys
import csv
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
from loguru import logger

from src.core.config import load_config
from src.core.twin_engine import TwinEngine
from src.data.mimic_loader import MIMICLoader


@click.command()
@click.option("--subject-id", default=None, help="MIMIC-IV subject_id")
@click.option("--stay-id", default=None, help="MIMIC-IV ICU stay_id")
@click.option("--record", default=None, help="Specific waveform record name")
@click.option("--start-s", default=0.0, type=float, help="Start offset in seconds")
@click.option("--max-chunks", default=None, type=int, help="Maximum chunks to process")
@click.option("--output", default=None, help="CSV output path for state snapshots")
@click.option("--demo", is_flag=True, help="Run with synthetic demo data")
@click.option("--config", default=None, help="Path to custom twin_config.yaml")
def main(
    subject_id: str,
    stay_id: str,
    record: str,
    start_s: float,
    max_chunks: int,
    output: str,
    demo: bool,
    config: str,
):
    # ── Load config ───────────────────────────────────────────────────────
    cfg_path = Path(config) if config else None
    cfg = load_config(cfg_path)

    # ── Pull settings from config ─────────────────────────────────────────
    sim_cfg = getattr(cfg, 'simulation', {})
    mode = sim_cfg.get('mode', 'demo')
    
    # If the user passed flags in the terminal, those override the config
    active_max_chunks = max_chunks or sim_cfg.get('max_chunks', 1200)
    active_output = output or sim_cfg.get('output_file', 'session_log.csv')
    is_demo = demo or (mode == 'demo')

    # ── The IoT Abstraction Layer ─────────────────────────────────────────
    if mode == "iot_mock":
        logger.info("📡 MOCK IoT MODE ACTIVATED: Listening for incoming sensor stream...")
        active_subject = "10014354"
        active_stay = "81739927"
    elif mode == "mimic":
        active_subject = subject_id or sim_cfg.get('subject_id')
        active_stay = stay_id or sim_cfg.get('stay_id')
        if not active_subject:
            logger.error("MIMIC mode requires a subject_id.")
            sys.exit(1)
    else:
        active_subject = None
        active_stay = None

    # ── Initialize Engine ─────────────────────────────────────────────────
    engine = TwinEngine(cfg)
    if active_subject:
        engine.start_session(subject_id=active_subject, stay_id=active_stay)

    # ── CSV writer ────────────────────────────────────────────────────────
    csv_file = None
    writer   = None
    if active_output:
        csv_file = open(active_output, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=list(engine.state.summary_dict().keys()))
        writer.writeheader()
        logger.info(f"Writing output to: {active_output}")

    # ── Run the Simulation ────────────────────────────────────────────────
    try:
        if mode == "iot_mock" or mode == "mimic":
            _run_mimic(engine, cfg, active_subject, active_stay, record, start_s, active_max_chunks, writer)
        else:
            _run_demo(engine, active_max_chunks, writer)
    finally:
        if csv_file:
            csv_file.close()
            logger.info(f"Output saved to: {active_output}")

def _run_demo(engine: TwinEngine, max_chunks: int, writer):
    """Run on synthetic data for testing."""
    logger.info("Running in DEMO mode with synthetic data")
    fs_ecg  = engine.cfg.ecg.sampling_rate
    fs_resp = engine.cfg.respiratory.sampling_rate
    chunk_s = 0.25

    for i in range(max_chunks):
        t = i * chunk_s
        n_ecg  = int(fs_ecg * chunk_s)
        n_resp = int(fs_resp * chunk_s)

        ecg_chunk  = _synthetic_ecg(n_ecg, fs_ecg, t)
        resp_chunk = np.sin(2 * np.pi * 0.25 * np.linspace(t, t + chunk_s, n_resp))
        spo2_chunk = np.array([98.0 - 0.3 * np.sin(t / 60)])

        state = engine.ingest(ecg_chunk, resp_chunk, spo2_chunk)

        if i % 40 == 0:  # Log every 10s
            d = state.summary_dict()
            logger.info(
                f"[{i*chunk_s:.0f}s] HR={d['heart_rate']} | "
                f"RR={d['respiratory_rate']} | SpO₂={d['spo2']} | "
                f"O₂D={d['o2_delivery_index']} | Alerts={d['n_alerts']}"
            )

        if writer:
            writer.writerow(state.summary_dict())

        for alert in state.active_alerts:
            logger.warning(f"  ALERT [{alert.severity}] {alert.message}")

    logger.info(f"Demo complete | {engine.cycle_count} cycles processed")


def _run_mimic(engine, cfg, subject_id, stay_id, record, start_s, max_chunks, writer):
    """Run on real MIMIC-IV waveform data."""
    loader = MIMICLoader(cfg)

    if not loader.authenticate():
        logger.error("Authentication failed. Check config/credentials.yaml")
        sys.exit(1)

    logger.info(f"Streaming MIMIC-IV | subject={subject_id} | stay={stay_id}")

    chunk_count = 0
    for chunk in loader.stream_waveforms(
        subject_id=subject_id,
        stay_id=stay_id,
        record_name=record,
        start_s=start_s,
        max_chunks=max_chunks,
    ):
        state = engine.ingest(
            ecg_chunk=chunk.ecg,
            resp_chunk=chunk.resp,
            spo2_chunk=chunk.spo2,
            timestamp=chunk.timestamp,
        )

        if chunk_count % 4 == 0:  # Log every second (4 × 250ms chunks)
            d = state.summary_dict()
            logger.info(
                f"[{state.elapsed_s:.0f}s] HR={d['heart_rate']} | "
                f"RR={d['respiratory_rate']} | SpO₂={d['spo2']} | "
                f"Stability={d['stability_index']:.2f} | Alerts={d['n_alerts']}"
            )

        if writer:
            writer.writerow(state.summary_dict())

        for alert in state.active_alerts:
            logger.warning(f"  ALERT [{alert.severity}] {alert.message}")

        chunk_count += 1

    logger.info(f"Processing complete | {engine.cycle_count} cycles")


def _synthetic_ecg(n: int, fs: float, t_offset: float) -> np.ndarray:
    t = np.linspace(t_offset, t_offset + n / fs, n)
    ecg = np.zeros(n)
    hr = 72.0
    for i, ti in enumerate(t):
        phase = (ti * hr / 60.0) % 1.0
        if 0.0 < phase < 0.05:
            ecg[i] = np.exp(-((phase - 0.025) ** 2) / 0.0001)
        elif 0.75 < phase < 0.90:
            ecg[i] = 0.15 * np.sin(np.pi * (phase - 0.75) / 0.15)
        elif 0.10 < phase < 0.35:
            ecg[i] = 0.3 * np.sin(np.pi * (phase - 0.10) / 0.25)
    return ecg + np.random.normal(0, 0.02, n)


if __name__ == "__main__":
    main()
