"""
src/visualization/dashboard.py
AeroCordis — Cardiopulmonary Digital Twin · Clinical Dashboard
"""

from __future__ import annotations

import os
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from auth import register_patient, authenticate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroCordis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #f0f2f5 !important; color: #334155; }
header { visibility: hidden; }
footer { visibility: hidden; }

.block-container { padding: 12px 20px 72px 20px !important; max-width: 100% !important; }
div[data-testid="stVerticalBlock"] { gap: 8px !important; }
div[data-testid="stHorizontalBlock"] { gap: 12px !important; align-items: stretch !important; }

section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e6eb;
    width: 200px !important;
}
section[data-testid="stSidebar"] > div { background: #ffffff !important; }

/* ── Metric card ── */
.ac-card {
    background: #ffffff;
    border: 1px solid #e2e6eb;
    border-top: 3px solid transparent;
    border-radius: 12px;
    padding: 18px 20px;
    transition: border-color 0.2s;
}
.ac-card.teal  { border-top-color: #0d9488; }
.ac-card.red   { border-top-color: #dc2626; }
.ac-card.blue  { border-top-color: #2563eb; }
.ac-card.amber { border-top-color: #d97706; }
.ac-card.violet{ border-top-color: #7c3aed; }
.ac-card.slate { border-top-color: #64748b; }
.ac-card.green { border-top-color: #16a34a; }
.ac-card.warn  { border-top-color: #d97706; background: #fffbeb; }
.ac-card.crit  { border-top-color: #dc2626; background: #fef2f2; animation: crit-pulse 1.8s ease-in-out infinite; }
@keyframes crit-pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
    50%      { box-shadow: 0 0 16px 3px rgba(220,38,38,0.12); }
}

.ac-label {
    color: #94a3b8;
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-family: 'DM Mono', monospace;
}
.ac-value {
    font-size: 2.4rem;
    font-weight: 600;
    line-height: 1;
    color: #1e293b;
    font-family: 'DM Mono', monospace;
}
.ac-unit { font-size: 0.82rem; color: #94a3b8; margin-left: 3px; font-weight: 400; }
.ac-sub  { font-size: 0.72rem; color: #94a3b8; margin-top: 6px; }
.ac-sub.ok   { color: #16a34a; }
.ac-sub.warn { color: #d97706; }
.ac-sub.crit { color: #dc2626; }

/* ── Section heading ── */
.ac-section {
    color: #64748b;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    border-bottom: 1px solid #e2e6eb;
    padding-bottom: 8px;
    margin: 22px 0 14px;
    font-family: 'DM Mono', monospace;
}

/* ── Coupling bar ── */
.coup-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.coup-lbl { color: #94a3b8; font-size: 0.67rem; width: 52px; flex-shrink: 0;
            letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600;
            font-family: 'DM Mono', monospace; }
.coup-track { flex: 1; height: 5px; background: #f1f5f9; border-radius: 3px; overflow: hidden; }
.coup-fill  { height: 100%; border-radius: 3px; }
.coup-val   { font-size: 0.78rem; font-weight: 600; width: 38px; text-align: right;
              font-family: 'DM Mono', monospace; }

/* ── Alert row ── */
.ac-alert {
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.78rem;
    border-radius: 8px;
    border: 1px solid;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ac-alert.ok   { background: #f0fdf4; border-color: #bbf7d0; color: #15803d; }
.ac-alert.warn { background: #fffbeb; border-color: #fde68a; color: #92400e; }
.ac-alert.crit { background: #fef2f2; border-color: #fca5a5; color: #dc2626; }

/* ── ETA card ── */
.eta-card { border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; border: 1px solid; }
.eta-card.ok   { background: #f0fdf4; border-color: #bbf7d0; }
.eta-card.warn { background: #fffbeb; border-color: #fde68a; }
.eta-card.crit { background: #fef2f2; border-color: #fca5a5; }
.eta-title { font-size: 0.63rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
             margin-bottom: 2px; font-family: 'DM Mono', monospace; }
.eta-value { font-size: 1.2rem; font-weight: 700; line-height: 1.1; font-family: 'DM Mono', monospace; }
.eta-sub   { font-size: 0.65rem; color: #94a3b8; margin-top: 2px; }

/* ── Patient sidebar card ── */
.patient-card {
    background: #f8fafc;
    border: 1px solid #e2e6eb;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 14px;
}
.patient-name { color: #1e293b; font-size: 0.92rem; font-weight: 600; }
.patient-meta { color: #94a3b8; font-size: 0.73rem; margin-top: 3px; }
.nav-label {
    color: #94a3b8; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase; margin: 14px 0 4px;
    font-family: 'DM Mono', monospace;
}

/* ── Bottom status bar ── */
.bot-bar {
    position: fixed; bottom: 0; left: 200px; right: 0; z-index: 100;
    background: #ffffff;
    border-top: 1px solid #e2e6eb;
    padding: 8px 28px;
    display: flex; align-items: center; gap: 22px;
    font-size: 0.72rem; color: #64748b;
}
.bot-item { display: flex; align-items: center; gap: 6px; }
.bot-dot  { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.bot-val  { color: #1e293b; font-weight: 600; font-family: 'DM Mono', monospace; }

/* ── Login box ── */
.login-box {
    max-width: 360px; margin: 80px auto;
    background: #ffffff;
    border: 1px solid #e2e6eb; border-radius: 16px; padding: 44px 36px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}

/* ── Login / Register: white inputs, black text ── */
.login-box input[type="text"],
.login-box input[type="password"],
.stTextInput input, .stNumberInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cccccc !important;
}
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #888888 !important;
}
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stNumberInput label {
    color: #000000 !important;
    font-weight: 500 !important;
}

/* ── Register button: green → blue on hover ── */
div[data-testid="stButton"]:has(button[kind="secondary"]).register-btn-wrap button {
    background-color: #28a745 !important;
    color: #ffffff !important;
    border: none !important;
    transition: background-color 0.25s ease !important;
}
div[data-testid="stButton"]:has(button[kind="secondary"]).register-btn-wrap button:hover {
    background-color: #1a73e8 !important;
}

/* ── Back to Login: blue with white text ── */
div[data-testid="stButton"].back-btn-wrap button {
    background-color: #1a73e8 !important;
    color: #ffffff !important;
    border: none !important;
}
div[data-testid="stButton"].back-btn-wrap button:hover {
    background-color: #1558b0 !important;
}

/* ── Watch panel ── */
.watch-panel {
    background: #f8fafc;
    border: 1px solid #e2e6eb; border-radius: 12px; padding: 20px;
}

/* ── Chart wrapper ── */
.chart-card {
    background: #ffffff;
    border: 1px solid #e2e6eb; border-radius: 12px; padding: 14px 16px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f5f6f8; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 2px; }

/* ── Sidebar nav radio ── */
div[role="radiogroup"] label {
    color: #1e293b !important; font-size: 0.82rem !important;
    padding: 7px 10px !important; border-radius: 7px !important; transition: all 0.15s;
}
div[role="radiogroup"] label:hover { background: #f1f5f9 !important; color: #0f172a !important; }
div[role="radiogroup"] label p { color: #1e293b !important; }

/* ── Sidebar general text ── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #1e293b !important; }
section[data-testid="stSidebar"] .nav-label { color: #64748b !important; }
section[data-testid="stSidebar"] select { color: #1e293b !important; background: #ffffff !important; }
section[data-testid="stSidebar"] .stButton button {
    color: #334155 !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e6eb !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #e2e8f0 !important;
    border-color: #cbd5e1 !important;
    color: #1e293b !important;
}
/* Selectbox in sidebar */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #f1f5f9 !important;
    border: 1px solid #e2e6eb !important;
    color: #334155 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stSelectbox svg { color: #64748b !important; fill: #64748b !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SCENARIOS: Dict[str, Dict[str, float]] = {
    "Normal":      {"hr": 74,  "spo2": 98,  "rr_hz": 0.25, "rmssd": 42, "rsa": 4.2, "psi": 0.68, "xcorr": 0.72, "stress": 18},
    "Hypoxia":     {"hr": 108, "spo2": 87,  "rr_hz": 0.32, "rmssd": 18, "rsa": 2.1, "psi": 0.42, "xcorr": 0.48, "stress": 62},
    "Tachycardia": {"hr": 138, "spo2": 95,  "rr_hz": 0.27, "rmssd": 12, "rsa": 1.4, "psi": 0.38, "xcorr": 0.41, "stress": 75},
    "Apnea":       {"hr": 55,  "spo2": 91,  "rr_hz": 0.07, "rmssd": 55, "rsa": 8.2, "psi": 0.31, "xcorr": 0.28, "stress": 58},
}

WATCH_MODELS = [
    "Apple Watch Series 9", "Apple Watch Ultra 2", "Samsung Galaxy Watch 6",
    "Garmin Fenix 7", "Fitbit Sense 2", "Polar H10 + Band",
    "Withings ScanWatch 2", "WHOOP 4.0", "Oura Ring Gen 3",
]

# ── All valid tab names ───────────────────────────────────────────────────────
ALL_TABS = [
    "🫀 Cardio Vitals",
    "🫁 Pulmonary Metrics",
    "⚡ ECG Coupling",
    "✍️ Manual Assessment",
    "⌚ Smartwatch",
    "📊 Twin History",
]

# ── Session State ─────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "logged_in": False, "registered": False,
        "patient_id": "", "patient_name": "",
        "patient_age": 0, "patient_uid": "",
        "current_tab": "🫀 Cardio Vitals", "live_feed": True,
        "t": 0.0, "scenario": "Normal", "cycle": 0,
        "session_start": datetime.utcnow(),
        "hr_hist":     [74.0] * 60, "spo2_hist":   [98.0] * 60,
        "rr_hist":     [16.0] * 60, "o2d_hist":    [1.00] * 60,
        "rmssd_hist":  [42.0] * 60, "psi_hist":    [0.68] * 60,
        "rsa_hist":    [4.2]  * 60, "stress_hist": [18.0] * 60,
        "hr": 74.0, "spo2": 98.0, "rr": 16.0, "ef": 56.0,
        "co": 5.2,  "tv": 480.0,  "rmssd": 42.0, "sdnn": 58.0,
        "qtc": 412.0, "o2d": 1.00, "stability": 0.82, "stress": 18.0,
        "rsa": 4.2, "psi": 0.68, "xcorr": 0.72, "coh": 0.61,
        "lf_hf": 1.2, "pnn50": 8.2,
        "eta_spo2": None, "eta_hr": None, "eta_o2d": None, "alerts": [],
        "watch_connected": False, "watch_model": "Apple Watch Series 9",
        "watch_last_sync": None, "watch_battery": 82,
        "watch_hr": 0.0, "watch_spo2": 0.0, "watch_steps": 0,
        "watch_calories": 0, "watch_hrv": 0.0, "watch_skin_temp": 0.0,
        "watch_activity": "Resting",
        "watch_hr_hist":   [0.0] * 60,
        "watch_spo2_hist": [0.0] * 60,
        "twin_history": [],
        "manual_results": [],
        "show_register": False,
        "just_logged_in": False,
        # nav state — track last selected values to detect changes
        "_last_nav": "🫀 Cardio Vitals",
        "_last_tool": "✍️ Manual Assessment",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Layout helpers ─────────────────────────────────────────────────────────────
def _base_layout(**extra: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "font": {"color": "#64748b", "size": 11, "family": "DM Mono, monospace"},
        "margin": {"l": 40, "r": 10, "t": 30, "b": 24},
        "showlegend": False,
    }
    base.update(extra)
    return base

def _ax(title: str = "") -> Dict[str, Any]:
    return {
        "showgrid": True,
        "gridcolor": "rgba(226,230,235,0.8)",
        "zeroline": False,
        "color": "#94a3b8",
        "linecolor": "#e2e6eb",
        "title": title,
    }

def _sp(px: int = 12) -> None:
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

def ac_card(label: str, value: str, unit: str, color: str,
            sub: str = "", sub_class: str = "ok",
            variant: str = "normal") -> str:
    card_class = {"warn": "warn", "crit": "crit"}.get(variant, color)
    return (
        f'<div class="ac-card {card_class}">'
        f'<div class="ac-label">{label}</div>'
        f'<div class="ac-value" style="color:{color};">{value}'
        f'<span class="ac-unit">{unit}</span></div>'
        f'<div class="ac-sub {sub_class}">{sub}</div>'
        f'</div>'
    )

def coupling_bar(label: str, val: float, lo: float, hi: float, color: str) -> str:
    pct = max(0.0, min(100.0, (val - lo) / (hi - lo + 1e-9) * 100))
    vstr = f"{val:.2f}" if hi <= 1.0 else f"{val:.1f}"
    return (
        f'<div class="coup-row">'
        f'<div class="coup-lbl">{label}</div>'
        f'<div class="coup-track"><div class="coup-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
        f'<div class="coup-val" style="color:{color}">{vstr}</div>'
        f'</div>'
    )

def _hex_to_rgba(hex_color: str, alpha: float = 0.07) -> str:
    """Convert #rrggbb hex to rgba(r,g,b,alpha) string safe for Plotly."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(13,148,136,{alpha})"  # fallback teal


def plot_trend(
    title: str,
    data: List[float],
    color: str,
    y_range: List[float],
    thresholds: Optional[List[Tuple[float, str]]] = None,
    data2: Optional[List[float]] = None,
    color2: str = "#7c3aed",
    label2: str = "",
) -> go.Figure:
    fig = go.Figure()
    if "rgba" in color or "rgb(" in color:
        fill_color = color.replace(")", ",0.06)").replace("rgb(", "rgba(")
    else:
        fill_color = _hex_to_rgba(color, 0.07)
    fig.add_trace(go.Scatter(
        x=list(range(len(data))), y=data, mode="lines",
        line={"color": color, "width": 2.0},
        fill="tozeroy",
        fillcolor=fill_color,
        name=title,
    ))
    if data2:
        fig.add_trace(go.Scatter(
            x=list(range(len(data2))), y=data2, mode="lines",
            line={"color": color2, "width": 1.6, "dash": "dot"},
            name=label2,
        ))
    if thresholds:
        for tv, tc in thresholds:
            fig.add_hline(y=tv, line_dash="dot", line_color=tc, line_width=1.0)
    ax = _ax()
    fig.update_layout(
        **_base_layout(
            title={"text": title, "font": {"size": 11, "color": "#94a3b8"}},
            height=200,
            xaxis={**ax, "showticklabels": False},
            yaxis={**ax, "range": y_range},
            showlegend=bool(data2),
            legend={"orientation": "h", "y": 1.1, "font": {"size": 9, "color": "#94a3b8"}, "bgcolor": "rgba(0,0,0,0)"},
        )
    )
    return fig

def gauge_circular(value: float, max_val: float, color: str, unit: str, height: int = 150) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"font": {"size": 26, "color": "#1e293b"}, "suffix": unit},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#94a3b8", "tickfont": {"size": 9, "color": "#94a3b8"}},
            "bar":  {"color": color, "thickness": 0.24},
            "bgcolor": "rgba(241,245,249,0.8)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_val * 0.5], "color": "rgba(241,245,249,0.5)"},
                {"range": [max_val * 0.5, max_val * 0.8], "color": "rgba(226,232,240,0.5)"},
            ],
            "threshold": {"line": {"color": "#94a3b8", "width": 2}, "thickness": 0.8, "value": max_val * 0.9},
        },
    ))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e293b"},
        margin={"l": 20, "r": 20, "t": 20, "b": 10},
    )
    return fig

def gauge_donut(value: float, max_val: float, color: str, unit: str) -> go.Figure:
    pct = min(1.0, value / max_val)
    fig = go.Figure(go.Pie(
        values=[pct, 1 - pct], hole=0.72, rotation=90, showlegend=False,
        marker={"colors": [color, "rgba(241,245,249,0.8)"], "line": {"width": 0}},
        hoverinfo="skip", textinfo="none",
    ))
    fig.add_annotation(
        text=f"<b>{value:.0f}</b><br><span style='font-size:9px'>{unit}</span>",
        x=0.5, y=0.5, showarrow=False,
        font={"size": 20, "color": "#1e293b"},
        align="center",
    )
    fig.update_layout(
        height=140,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Engine step ───────────────────────────────────────────────────────────────
def _step_engine() -> None:
    sc = SCENARIOS.get(st.session_state.scenario, SCENARIOS["Normal"])
    t: float = st.session_state.t

    def lerp(key: str, target: float, noise: float, lr: float = 0.05) -> float:
        cur: float = st.session_state[key]
        new = cur + lr * (target - cur) + np.random.normal(0, noise)
        st.session_state[key] = float(new)
        return float(new)

    hr     = float(np.clip(lerp("hr",    sc["hr"],    1.5), 30, 200))
    spo2   = float(np.clip(lerp("spo2",  sc["spo2"],  0.2), 70, 100))
    rmssd  = float(np.clip(lerp("rmssd", sc["rmssd"], 1.0), 5, 250))
    psi    = float(np.clip(lerp("psi",   sc["psi"],   0.02), 0, 1))
    rsa    = float(np.clip(lerp("rsa",   sc["rsa"],   0.1), 0, 20))
    xcorr  = float(np.clip(lerp("xcorr", sc["xcorr"], 0.02), -1, 1))
    stress = float(np.clip(lerp("stress", sc["stress"], 1.0), 0, 100))

    rr_bpm = float(np.clip(sc["rr_hz"] * 60 + np.random.normal(0, 0.3), 3, 50))
    o2d    = float(np.clip((spo2 / 100) * (hr / 74) * 0.88 + 0.12, 0.3, 1.5))
    stab   = float(np.clip(rmssd / 42 * 0.3 + psi * 0.4 + xcorr * 0.3, 0, 1))
    ef     = float(np.clip(st.session_state.ef + np.random.normal(0, 0.3), 40, 75))
    co     = float(np.clip(hr * ef / 100 * 0.07, 2.5, 10.0))
    tv     = float(np.clip(st.session_state.tv + np.random.normal(0, 8), 300, 700))
    sdnn   = float(np.clip(rmssd * 1.4, 5, 200))
    qtc    = float(np.clip(st.session_state.qtc + np.random.normal(0, 1.5), 320, 560))
    lf_hf  = float(np.clip(st.session_state.lf_hf + np.random.normal(0, 0.05), 0.1, 6))
    pnn50  = float(np.clip(rmssd / 2 - 5, 0, 40))
    coh    = float(np.clip(st.session_state.coh + np.random.normal(0, 0.01), 0, 1))

    st.session_state.update({
        "hr": hr, "spo2": spo2, "rr": rr_bpm, "rmssd": rmssd,
        "psi": psi, "rsa": rsa, "xcorr": xcorr, "stress": stress,
        "o2d": o2d, "stability": stab, "ef": ef, "co": co,
        "tv": tv, "sdnn": sdnn, "qtc": qtc, "lf_hf": lf_hf,
        "pnn50": pnn50, "coh": coh, "t": t + 2.0,
        "cycle": st.session_state.cycle + 1,
    })

    def push(key: str, val: float) -> None:
        st.session_state[key].pop(0)
        st.session_state[key].append(float(val))

    push("hr_hist", hr); push("spo2_hist", spo2); push("rr_hist", rr_bpm)
    push("o2d_hist", o2d); push("rmssd_hist", rmssd)
    push("psi_hist", psi); push("rsa_hist", rsa); push("stress_hist", stress)

    if st.session_state.watch_connected:
        _step_watch(hr, spo2, rmssd)

    # ── Alerts ────────────────────────────────────────────────────────────────
    alerts: List[Tuple[str, str, str]] = []
    if spo2 < 90:  alerts.append(("critical", "spo2", f"SpO₂ CRITICAL — {spo2:.0f}%"))
    elif spo2 < 94: alerts.append(("warning",  "spo2", f"SpO₂ low — {spo2:.0f}%"))
    if hr > 130:   alerts.append(("critical", "hr",   f"Tachycardia — {hr:.0f} bpm"))
    elif hr > 100: alerts.append(("warning",  "hr",   f"HR elevated — {hr:.0f} bpm"))
    if hr < 45:    alerts.append(("critical", "hr",   f"Bradycardia — {hr:.0f} bpm"))
    if o2d < 0.6:  alerts.append(("critical", "o2d",  f"O₂ delivery CRITICAL — {o2d:.3f}"))
    elif o2d < 0.8: alerts.append(("warning", "o2d",  f"O₂ delivery low — {o2d:.3f}"))
    if qtc > 500:  alerts.append(("critical", "qtc",  f"QTc critically prolonged — {qtc:.0f} ms"))
    elif qtc > 460: alerts.append(("warning", "qtc",  f"QTc prolonged — {qtc:.0f} ms"))
    st.session_state.alerts = alerts

    # ── ETA ───────────────────────────────────────────────────────────────────
    def eta_below(hist: List[float], thr: float) -> Optional[str]:
        if len(hist) < 10: return None
        slope = (hist[-1] - hist[-10]) / 10.0
        if slope >= 0: return None
        val = hist[-1]
        if val <= thr: return "NOW"
        s = abs((val - thr) / slope)
        return f"{s:.0f}s" if s < 300 else None

    def eta_above(hist: List[float], thr: float) -> Optional[str]:
        if len(hist) < 10: return None
        slope = (hist[-1] - hist[-10]) / 10.0
        if slope <= 0: return None
        val = hist[-1]
        if val >= thr: return "NOW"
        s = abs((thr - val) / slope)
        return f"{s:.0f}s" if s < 300 else None

    st.session_state.eta_spo2 = eta_below(st.session_state.spo2_hist, 90)
    st.session_state.eta_hr   = eta_above(st.session_state.hr_hist, 140)
    st.session_state.eta_o2d  = eta_below(st.session_state.o2d_hist, 0.6)

    if st.session_state.cycle % 30 == 0:
        _save_history_snapshot()


def _step_watch(twin_hr: float, twin_spo2: float, twin_rmssd: float) -> None:
    s = st.session_state
    w_hr   = float(np.clip(twin_hr   + np.random.normal(0, 2.5), 30, 210))
    w_spo2 = float(np.clip(twin_spo2 + np.random.normal(0, 0.5), 70, 100))
    w_hrv  = float(np.clip(twin_rmssd + np.random.normal(0, 3),  5, 200))
    w_temp = float(np.clip(36.6 + np.random.normal(0, 0.15), 35.0, 40.0))
    activity_map = {
        "Normal": "Resting", "Hypoxia": "Resting",
        "Tachycardia": "Active", "Apnea": "Sleeping",
    }
    s.watch_hr         = w_hr
    s.watch_spo2       = w_spo2
    s.watch_hrv        = w_hrv
    s.watch_skin_temp  = w_temp
    s.watch_steps      = int(s.watch_steps + random.randint(0, 3))
    s.watch_calories   = int(s.watch_calories + random.randint(0, 1))
    s.watch_activity   = activity_map.get(s.scenario, "Resting")
    s.watch_last_sync  = datetime.utcnow()
    s.watch_battery    = max(1, int(s.watch_battery - 0.01))
    s.watch_hr_hist.pop(0);   s.watch_hr_hist.append(w_hr)
    s.watch_spo2_hist.pop(0); s.watch_spo2_hist.append(w_spo2)


def _save_history_snapshot() -> None:
    s = st.session_state
    phase = (
        "CRITICAL" if any(a[0] == "critical" for a in s.alerts) else
        "WARNING"  if any(a[0] == "warning"  for a in s.alerts) else
        "STABLE"
    )
    entry = {
        "ts": datetime.utcnow(), "scenario": s.scenario,
        "hr": round(s.hr, 1), "spo2": round(s.spo2, 1), "rr": round(s.rr, 1),
        "o2d": round(s.o2d, 3), "psi": round(s.psi, 3), "rmssd": round(s.rmssd, 1),
        "stress": round(s.stress, 1), "stability": round(s.stability, 3),
        "phase": phase, "cycle": s.cycle,
        "watch_hr": round(s.watch_hr, 1) if s.watch_connected else None,
    }
    s.twin_history.insert(0, entry)
    if len(s.twin_history) > 200:
        s.twin_history = s.twin_history[:200]


# ── ETA helper ────────────────────────────────────────────────────────────────
def _compute_eta_seconds(hist: List[float], threshold: float, direction: str = "below") -> Optional[float]:
    if len(hist) < 10:
        return None
    slope = (hist[-1] - hist[-10]) / 10.0
    val = hist[-1]
    if direction == "below":
        if slope >= 0: return None
        if val <= threshold: return 0.0
        return abs((val - threshold) / slope) * 2.0
    else:
        if slope <= 0: return None
        if val >= threshold: return 0.0
        return abs((threshold - val) / slope) * 2.0


def _eta_card(label: str, metric: str, value: float, threshold: float,
              unit: str, eta_secs: Optional[float], direction: str = "below") -> str:
    if eta_secs is not None and eta_secs == 0.0:
        cls, color = "crit", "#dc2626"
        eta_text = "THRESHOLD CROSSED"
        eta_sub  = f"{metric} = {value:.1f}{unit} · threshold {threshold}{unit}"
    elif eta_secs is not None:
        mins = int(eta_secs // 60); secs = int(eta_secs % 60)
        if eta_secs < 60:
            cls, color = "crit", "#dc2626"; eta_text = f"{secs}s"
        elif eta_secs < 180:
            cls, color = "warn", "#d97706"; eta_text = f"{mins}m {secs:02d}s"
        else:
            cls, color = "warn", "#d97706"; eta_text = f"~{mins}m"
        eta_sub = f"Threshold: {threshold}{unit} · Current: {value:.1f}{unit}"
    else:
        cls, color = "ok", "#16a34a"
        eta_text = "Stable"
        eta_sub  = f"Not trending toward {threshold}{unit}"

    return (
        f'<div class="eta-card {cls}">'
        f'<div class="eta-title" style="color:{color}">{label}</div>'
        f'<div class="eta-value" style="color:{color}">{eta_text}</div>'
        f'<div class="eta-sub">{eta_sub}</div>'
        f'</div>'
    )


# ── UID helper ────────────────────────────────────────────────────────────────
def _gen_uid() -> str:
    return f"{random.randint(10000,99999)}-{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"


# ── REGISTER ──────────────────────────────────────────────────────────────────
def view_register() -> None:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align:center;margin-bottom:24px">
                <div style="font-size:2.5rem">🫀</div>
                <div style="color:#1e293b;font-size:1.2rem;font-weight:700;margin-top:8px">
                    PATIENT REGISTRATION</div>
                <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px;letter-spacing:1.5px">
                    AEROCORDIS MONITORING SYSTEM</div>
            </div>""", unsafe_allow_html=True)

        full_name = st.text_input("Full Name", placeholder="e.g. Jane Smith", key="reg_name")
        age       = st.number_input("Age", min_value=1, max_value=120, value=30, key="reg_age")
        st.selectbox("Gender", ["Male", "Female", "Other / Prefer not to say"], key="reg_gender")
        pwd1 = st.text_input("Create Access Key", type="password", placeholder="Min 4 characters", key="reg_pwd1")
        pwd2 = st.text_input("Confirm Access Key", type="password", placeholder="Repeat access key", key="reg_pwd2")

        if st.button("✅ REGISTER & CREATE TWIN", use_container_width=True, type="primary"):
            if not full_name.strip():
                st.error("Please enter your full name.")
            elif len(pwd1) < 4:
                st.error("Access key must be at least 4 characters.")
            elif pwd1 != pwd2:
                st.error("Access keys do not match.")
            else:
                uid = _gen_uid()
                register_patient(
                    uid=uid,
                    full_name=full_name.strip(),
                    age=int(age),
                    gender=st.session_state.reg_gender,
                    password=pwd1,
                )
                st.session_state.patient_name = full_name.strip()
                st.session_state.patient_age  = int(age)
                st.session_state.patient_uid  = uid
                st.session_state.patient_id   = uid
                st.session_state.registered   = True
                st.markdown(f"""
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                            border-radius:10px;padding:14px;margin-top:10px;text-align:center">
                    <div style="color:#16a34a;font-weight:700;font-size:0.9rem">Registration Successful!</div>
                    <div style="color:#334155;font-size:0.8rem;margin-top:6px">
                        Patient ID: <b style="color:#1e293b;font-size:1rem">{uid}</b>
                    </div>
                    <div style="color:#94a3b8;font-size:0.72rem;margin-top:4px">Save this ID — you'll need it to log in.</div>
                </div>""", unsafe_allow_html=True)
                time.sleep(2)
                st.rerun()

        _sp(10)
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.show_register = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ── LOGIN ─────────────────────────────────────────────────────────────────────
def view_login() -> None:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:

        # ── Logo (outside the login box, no white gap) ────────────────────────
        logo_candidates = [
            Path(__file__).parent / "aerocordis_logo.png",
            Path(__file__).parent.parent / "aerocordis_logo.png",
            Path.cwd() / "aerocordis_logo.png",
        ]
        logo_path = next((p for p in logo_candidates if p.exists()), None)
        if logo_path:
            st.image(str(logo_path), use_container_width=True)

        # ── Login card ────────────────────────────────────────────────────────
        st.markdown("""
            <div style="text-align:center;margin-bottom:24px;margin-top:8px">
                <div style="color:#1e293b;font-size:1.2rem;font-weight:700;">
                    AEROCORDIS</div>
                <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px;letter-spacing:1.5px">
                    CARDIOPULMONARY MONITORING SYSTEM</div>
            </div>""", unsafe_allow_html=True)

        if st.session_state.get("registered"):
            uid = st.session_state.patient_uid
            st.markdown(f"""
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                            border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:0.78rem">
                    <b style="color:#16a34a">✓ Registration complete!</b><br>
                    <span style="color:#334155">Patient ID: <b style="color:#1e293b">{uid}</b></span>
                </div>""", unsafe_allow_html=True)

        pid = st.text_input("Patient ID", placeholder="Your system-generated ID", key="login_pid")
        pwd = st.text_input("Access Key", type="password", placeholder="Enter access key", key="login_pwd")

        if st.button("AUTHENTICATE & LAUNCH TWIN", use_container_width=True, type="primary"):
            if pid and pwd:
                ok, record = authenticate(pid, pwd)
                if ok and record:
                    st.session_state.logged_in      = True
                    st.session_state.patient_id     = pid
                    st.session_state.patient_uid    = pid
                    st.session_state.patient_name   = record["full_name"]
                    st.session_state.patient_age    = record["age"]
                    st.session_state.current_tab    = "🫀 Cardio Vitals"
                    st.session_state.session_start  = datetime.utcnow()
                    st.session_state.just_logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid Patient ID or Access Key. Please check your credentials.")
            else:
                st.error("Enter both Patient ID and Access Key.")

        _sp(8)
        if st.button("📋 New Patient? Register Here", use_container_width=True):
            st.session_state.show_register = True
            st.rerun()

        st.markdown("""
            <div style="text-align:center;margin-top:14px;color:#94a3b8;
                        font-size:0.72rem;letter-spacing:1px">
                MIMIC-IV CREDENTIALED ACCESS · PhysioNet
            </div>""", unsafe_allow_html=True)


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def view_dashboard() -> None:
    s = st.session_state

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div class="patient-card">
            <div style="color:#0d9488;font-size:0.65rem;font-weight:700;letter-spacing:2px;margin-bottom:4px">PATIENT</div>
            <div class="patient-name">{s.patient_name}</div>
            <div class="patient-meta">Age: {s.patient_age} · ID: {s.patient_uid}</div>
            <div class="patient-meta" style="margin-top:5px">
                Status: <span style="color:#16a34a;font-weight:600">● Active</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Unified single-radio navigation ──────────────────────────────────
        # Find current tab index so the radio reflects the active tab
        current_idx = ALL_TABS.index(s.current_tab) if s.current_tab in ALL_TABS else 0

        st.markdown('<div class="nav-label">Monitoring</div>', unsafe_allow_html=True)
        selected_monitoring = st.radio(
            "monitoring_nav",
            ["🫀 Cardio Vitals", "🫁 Pulmonary Metrics", "⚡ ECG Coupling"],
            index=current_idx if current_idx < 3 else 0,
            key="nav_monitoring",
            label_visibility="collapsed",
        )

        st.markdown('<div class="nav-label">Tools</div>', unsafe_allow_html=True)
        tools_options = [
            "✍️ Manual Assessment",
            "⌚ Smartwatch",
            "📊 Twin History",
                ]
        tools_idx = (current_idx - 3) if 3 <= current_idx <= 6 else 0
        selected_tools = st.radio(
            "tools_nav",
            tools_options,
            index=tools_idx,
            key="nav_tools",
            label_visibility="collapsed",
        )

        # Determine which group was most recently interacted with by comparing
        # to stored previous values, then update current_tab accordingly.
        prev_monitoring = s.get("_last_nav", selected_monitoring)
        prev_tools      = s.get("_last_tool", selected_tools)

        if selected_monitoring != prev_monitoring:
            # Monitoring radio changed
            s.current_tab   = selected_monitoring
            s["_last_nav"]  = selected_monitoring
        elif selected_tools != prev_tools:
            # Tools radio changed
            s.current_tab   = selected_tools
            s["_last_tool"] = selected_tools

        st.markdown("---")
        st.markdown('<div class="nav-label">Controls</div>', unsafe_allow_html=True)
        st.toggle("🔴 Live Feed Auto-Sync", key="live_feed")
        st.selectbox("Scenario", list(SCENARIOS.keys()), key="scenario")

        if st.button("🔄 Manual Sync", use_container_width=True):
            _step_engine()
            st.rerun()
        if st.button("💾 Save Snapshot", use_container_width=True):
            _save_history_snapshot()
            st.success("Snapshot saved!")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

        st.markdown("---")
        elapsed = int((datetime.utcnow() - s.session_start).total_seconds())
        h, m, sec = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        watch_str  = "🟢 Connected" if s.watch_connected else "⚫ No Watch"
        watch_col  = "#16a34a" if s.watch_connected else "#94a3b8"
        st.markdown(f"""
        <div style="font-size:0.7rem;color:#94a3b8;text-align:center;line-height:1.8">
            <div style="color:#0d9488;font-weight:700;font-family:'DM Mono',monospace">SESSION</div>
            {h:02d}:{m:02d}:{sec:02d} · Cycle #{s.cycle}<br>
            <span style="color:{watch_col}">{watch_str}</span>
        </div>""", unsafe_allow_html=True)

    # ── Top bar ───────────────────────────────────────────────────────────────
    now_str = datetime.utcnow().strftime("%H:%M UTC · %d %b %Y")
    watch_line = ""
    if s.watch_connected:
        parts  = s.watch_model.split()
        wname  = " ".join(parts[:2]) if len(parts) >= 2 else s.watch_model
        watch_line = (
            f'<span style="color:#7c3aed;font-size:0.72rem;font-weight:600;">'
            f'⌚ {wname} · 🔋{s.watch_battery}%</span>'
            f'<span style="color:#e2e6eb;margin:0 8px;">|</span>'
        )
    st.markdown(
        '<div style="background:#ffffff;padding:10px 20px 8px;'
        'border-bottom:1px solid #e2e6eb;display:flex;align-items:center;'
        'justify-content:space-between;margin-bottom:12px;">'
        '<div>'
        '<span style="color:#1e293b;font-size:1.05rem;font-weight:700;letter-spacing:2px;'
        'font-family:\'DM Mono\',monospace;">AEROCORDIS</span>'
        f'<span style="color:#94a3b8;font-size:0.74rem;margin-left:14px;">'
        f'PATIENT {s.patient_name.upper()} · ID: {s.patient_uid} · MIMIC-IV</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:0;">'
        + watch_line
        + f'<span style="color:#94a3b8;font-size:0.74rem;">{now_str}</span>'
        '<span style="color:#e2e6eb;margin:0 8px;">|</span>'
        '<span style="color:#0d9488;font-size:0.74rem;font-weight:700;">● SYNC</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Alert banner ──────────────────────────────────────────────────────────
    if s.alerts:
        cols = st.columns(min(len(s.alerts), 3))
        for i, (sev, ch, msg) in enumerate(s.alerts[:3]):
            with cols[i]:
                ico  = "🔴" if sev == "critical" else "🟡"
                eta  = s.eta_spo2 if ch == "spo2" else (s.eta_hr if ch == "hr" else s.eta_o2d)
                eta_s = f" · ETA {eta}" if eta else ""
                clr   = "#dc2626" if sev == "critical" else "#d97706"
                bg    = "#fef2f2" if sev == "critical" else "#fffbeb"
                bd    = "#fca5a5" if sev == "critical" else "#fde68a"
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {bd};border-radius:8px;'
                    f'padding:9px 14px;font-size:0.78rem;display:flex;align-items:center;gap:8px;">'
                    f'{ico} <div><b style="color:{clr}">[{sev.upper()}]</b> {msg}{eta_s}</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Route tab ─────────────────────────────────────────────────────────────
    tab = s.current_tab
    if   tab == "🫀 Cardio Vitals":       _tab_cardio()
    elif tab == "🫁 Pulmonary Metrics":   _tab_pulmonary()
    elif tab == "⚡ ECG Coupling":        _tab_ecg()
    elif tab == "⌚ Smartwatch":          _tab_smartwatch()
    elif tab == "✍️ Manual Assessment":   _tab_manual()
    elif tab == "📊 Twin History":        _tab_history()
    else:                                 _tab_cardio()

    # ── Bottom bar ────────────────────────────────────────────────────────────
    phase = (
        "CRITICAL" if any(a[0] == "critical" for a in s.alerts) else
        "WARNING"  if any(a[0] == "warning"  for a in s.alerts) else
        "STABLE"
    )
    pcol = "#dc2626" if phase == "CRITICAL" else "#d97706" if phase == "WARNING" else "#0d9488"
    watch_live = (
        f' <span style="color:#7c3aed">· ⌚ {s.watch_hr:.0f}</span>'
        if s.watch_connected else ""
    )
    st.markdown(f"""
    <div class="bot-bar">
        <div class="bot-item">
            <div class="bot-dot" style="background:#dc2626"></div>
            HR <span class="bot-val">{s.hr:.0f} bpm{watch_live}</span>
        </div>
        <div class="bot-item">
            <div class="bot-dot" style="background:#0d9488"></div>
            SpO₂ <span class="bot-val">{s.spo2:.1f}%</span>
        </div>
        <div class="bot-item">
            <div class="bot-dot" style="background:#2563eb"></div>
            RR <span class="bot-val">{s.rr:.1f} br/min</span>
        </div>
        <div class="bot-item">
            <div class="bot-dot" style="background:#d97706"></div>
            O₂D <span class="bot-val">{s.o2d:.3f}</span>
        </div>
        <div class="bot-item">
            <div class="bot-dot" style="background:#7c3aed"></div>
            PSI <span class="bot-val">{s.psi:.3f}</span>
        </div>
        <div style="flex:1"></div>
        <div style="color:{pcol};font-weight:700;font-size:0.74rem;letter-spacing:1.5px;
                    font-family:'DM Mono',monospace;">
            AeroCordis: {phase} · Next update 2s
        </div>
    </div>""", unsafe_allow_html=True)


# ── TAB: CARDIO ───────────────────────────────────────────────────────────────
def _tab_cardio() -> None:
    s = st.session_state
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)

    left, centre, right = st.columns([1.4, 2.2, 1.4])

    with left:
        st.markdown(
            '<div style="background:#ffffff;border:1px solid #e2e6eb;border-radius:12px;'
            'padding:12px 16px;margin-bottom:8px;text-align:center;">'
            '<div class="ac-label">HEART RATE</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            gauge_circular(s.hr, 200, "#dc2626", "BPM"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        if s.watch_connected:
            st.markdown(
                f'<div style="text-align:center;color:#7c3aed;font-size:0.72rem;'
                f'margin-top:-8px;margin-bottom:6px;">⌚ Watch: {s.watch_hr:.0f} bpm</div>',
                unsafe_allow_html=True,
            )

        ef_color = "#0d9488" if s.ef >= 50 else "#d97706"
        ef_label = "Normal EF" if s.ef >= 50 else "Reduced EF"
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e2e6eb;border-top:3px solid {ef_color};'
            f'border-radius:12px;padding:16px;text-align:center;margin-bottom:8px;">'
            f'<div class="ac-label">EJECTION FRACTION</div>'
            f'<div style="font-size:2.2rem;font-weight:600;color:{ef_color};'
            f'font-family:\'DM Mono\',monospace;">{s.ef:.0f}'
            f'<span class="ac-unit">%</span></div>'
            f'<div class="ac-sub ok">{ef_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e2e6eb;border-top:3px solid #2563eb;'
            f'border-radius:12px;padding:16px;text-align:center;">'
            f'<div class="ac-label">CARDIAC OUTPUT</div>'
            f'<div style="font-size:2.2rem;font-weight:600;color:#2563eb;'
            f'font-family:\'DM Mono\',monospace;">{s.co:.1f}'
            f'<span class="ac-unit">L/min</span></div>'
            f'<div class="ac-sub">HR {s.hr:.0f} × SV</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with centre:
        st.markdown("""
        <div style="display:flex;justify-content:center;align-items:center;height:300px;
                    background:linear-gradient(135deg,#f8fafc,#f1f5f9);
                    border-radius:14px;border:1px solid #e2e6eb;
                    position:relative;overflow:hidden;margin-bottom:10px">
            <svg viewBox="0 0 260 340" width="190" height="250" xmlns="http://www.w3.org/2000/svg">
              <path d="M115 30 L145 30 L148 65 L112 65 Z"
                    fill="rgba(13,148,136,0.06)" stroke="rgba(13,148,136,0.25)" stroke-width="1"/>
              <path d="M60 70 Q40 90 38 160 Q36 220 45 280 Q55 310 90 318
                       L130 322 L170 318 Q205 310 215 280 Q224 220 222 160
                       Q220 90 200 70 Q175 60 130 58 Q85 60 60 70Z"
                    fill="rgba(248,250,252,0.6)" stroke="rgba(13,148,136,0.22)" stroke-width="1.5"/>
              <path d="M105 125 Q95 110 112 108 Q122 107 128 118
                       Q134 107 144 108 Q161 110 151 125 L128 150 Z"
                    fill="rgba(220,38,38,0.55)" stroke="#dc2626" stroke-width="1.8"/>
              <path d="M105 125 Q95 110 112 108 Q122 107 128 118
                       Q134 107 144 108 Q161 110 151 125 L128 150 Z"
                    fill="none" stroke="rgba(220,38,38,0.18)" stroke-width="7"/>
              <path d="M128 118 Q128 95 132 82"
                    stroke="rgba(220,38,38,0.55)" stroke-width="2.5" fill="none"/>
              <path d="M155 100 Q192 108 196 155 Q198 185 175 198
                       Q160 204 150 190 Q140 175 142 155 Q144 128 155 100Z"
                    fill="rgba(37,99,235,0.07)" stroke="rgba(37,99,235,0.28)" stroke-width="1.5"/>
              <path d="M105 100 Q68 108 64 155 Q62 185 85 198
                       Q100 204 110 190 Q120 175 118 155 Q116 128 105 100Z"
                    fill="rgba(37,99,235,0.07)" stroke="rgba(37,99,235,0.28)" stroke-width="1.5"/>
            </svg>
            <div style="position:absolute;top:12px;left:14px;color:#0d9488;
                        font-size:0.6rem;letter-spacing:2px;font-weight:700;
                        font-family:'DM Mono',monospace;">TWIN ACTIVE</div>
            <div style="position:absolute;bottom:10px;right:14px;color:#94a3b8;
                        font-size:0.58rem;font-family:'DM Mono',monospace;">AeroCordis v2.1</div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        for col, lbl, val_str, color, sub in [
            (m1, "STABILITY",   f"{s.stability:.2f}", "#0d9488", "/1.0 index"),
            (m2, "O₂ DELIVERY", f"{s.o2d:.3f}",       "#16a34a" if s.o2d >= 0.95 else "#d97706", "adequate" if s.o2d >= 0.95 else "reduced"),
            (m3, "STRESS LOAD", f"{s.stress:.0f}",    "#d97706" if s.stress < 70 else "#dc2626", "low" if s.stress < 40 else "moderate" if s.stress < 70 else "HIGH"),
        ]:
            with col:
                st.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e2e6eb;border-top:3px solid {color};'
                    f'border-radius:12px;padding:14px;text-align:center;">'
                    f'<div class="ac-label">{lbl}</div>'
                    f'<div style="font-size:1.9rem;font-weight:600;color:{color};'
                    f'font-family:\'DM Mono\',monospace;">{val_str}</div>'
                    f'<div class="ac-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown(
            '<div style="background:#ffffff;border:1px solid #e2e6eb;border-radius:12px;'
            'padding:12px 16px;margin-bottom:8px;text-align:center;">'
            '<div class="ac-label">SpO₂</div></div>',
            unsafe_allow_html=True,
        )
        spo2_col = "#0d9488" if s.spo2 >= 94 else "#d97706" if s.spo2 >= 90 else "#dc2626"
        st.plotly_chart(
            gauge_donut(s.spo2, 100, spo2_col, "%"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        if s.watch_connected:
            st.markdown(
                f'<div style="text-align:center;color:#7c3aed;font-size:0.72rem;'
                f'margin-top:-8px;margin-bottom:6px;">⌚ Watch: {s.watch_spo2:.1f}%</div>',
                unsafe_allow_html=True,
            )

        rr_col   = "#2563eb" if 12 <= s.rr <= 20 else "#d97706"
        rr_label = "Normal" if 12 <= s.rr <= 20 else "Abnormal"
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e2e6eb;border-top:3px solid {rr_col};'
            f'border-radius:12px;padding:16px;text-align:center;margin-bottom:8px;">'
            f'<div class="ac-label">RESPIRATION RATE</div>'
            f'<div style="font-size:2.2rem;font-weight:600;color:{rr_col};'
            f'font-family:\'DM Mono\',monospace;">{s.rr:.0f}'
            f'<span class="ac-unit">RR</span></div>'
            f'<div class="ac-sub ok">{rr_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e2e6eb;border-top:3px solid #7c3aed;'
            f'border-radius:12px;padding:16px;text-align:center;">'
            f'<div class="ac-label">TIDAL VOLUME</div>'
            f'<div style="font-size:2.2rem;font-weight:600;color:#7c3aed;'
            f'font-family:\'DM Mono\',monospace;">{s.tv:.0f}'
            f'<span class="ac-unit">ml</span></div>'
            f'<div class="ac-sub">Spontaneous</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _sp(10)
    st.markdown('<div class="ac-section">VITALS TRENDS · Live (60s window)</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        fig = plot_trend(
            "Heart Rate (24h)", s.hr_hist, "#dc2626", [40, 160],
            [(140, "rgba(220,38,38,0.5)"), (100, "rgba(217,119,6,0.5)")],
            s.watch_hr_hist if s.watch_connected else None, "#7c3aed", "⌚ Watch HR",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with tc2:
        fig = plot_trend(
            "Blood Oxygen SpO₂", s.spo2_hist, "#0d9488", [82, 101],
            [(90, "rgba(220,38,38,0.5)"), (94, "rgba(217,119,6,0.5)")],
            s.watch_spo2_hist if s.watch_connected else None, "#7c3aed", "⌚ Watch SpO₂",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ── TAB: PULMONARY ────────────────────────────────────────────────────────────
def _tab_pulmonary() -> None:
    s = st.session_state
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="ac-section" style="margin-top:0">PULMONARY METRICS</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "RESP RATE",      f"{s.rr:.1f}",  " br/min", "#2563eb",  "Normal" if 12<=s.rr<=20 else "Abnormal"),
        (c2, "TIDAL VOLUME",   f"{s.tv:.0f}",  " ml",     "#7c3aed",  "Spontaneous"),
        (c3, "COUPLING INDEX", f"{s.psi:.3f}", "",         "#0d9488",  "Phase sync (PSI)"),
        (c4, "RSA AMPLITUDE",  f"{s.rsa:.1f}", " bpm",    "#d97706",  "Vagal tone proxy"),
    ]
    for col, lbl, val, unit, color, sub in cards:
        with col:
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #e2e6eb;border-top:3px solid {color};'
                f'border-radius:12px;padding:18px 20px;">'
                f'<div class="ac-label">{lbl}</div>'
                f'<div style="font-size:2.2rem;font-weight:600;color:{color};'
                f'font-family:\'DM Mono\',monospace;">{val}'
                f'<span class="ac-unit">{unit}</span></div>'
                f'<div class="ac-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    _sp(12)
    lc, rc = st.columns([1, 2])
    with lc:
        psi_s = "🟢 Strong sync" if s.psi > 0.6 else ("🟡 Moderate" if s.psi > 0.3 else "🔴 Weak/absent")
        st.markdown(
            '<div style="background:#ffffff;border:1px solid #e2e6eb;border-radius:12px;padding:16px;">'
            '<div class="ac-section" style="margin-top:0">HEART-LUNG COUPLING</div>'
            + coupling_bar("RSA",   s.rsa,   0, 15, "#d97706")
            + coupling_bar("PSI",   s.psi,   0, 1,  "#0d9488")
            + coupling_bar("XCORR", s.xcorr, -1, 1, "#2563eb")
            + coupling_bar("COH",   s.coh,   0, 1,  "#7c3aed")
            + f'<div style="font-size:0.78rem;color:#64748b;margin-top:8px;line-height:1.8;">'
            f'n:m ratio ≈ 4.2:1 · XCorr lag: +1.8s<br>{psi_s}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with rc:
        ax = _ax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(60)), y=s.psi_hist, mode="lines",
                                 name="PSI", line={"color": "#0d9488", "width": 1.8}))
        fig.add_trace(go.Scatter(x=list(range(60)), y=[v / 15 for v in s.rsa_hist], mode="lines",
                                 name="RSA/15", line={"color": "#d97706", "width": 1.6}))
        fig.add_trace(go.Scatter(x=list(range(60)), y=[v / 100 for v in s.stress_hist], mode="lines",
                                 name="Stress/100", line={"color": "#dc2626", "width": 1.4}))
        fig.update_layout(
            **_base_layout(
                height=260, showlegend=True,
                title={"text": "Coupling Metrics — 60s", "font": {"size": 11, "color": "#94a3b8"}},
                legend={"orientation": "h", "y": 1.1, "font": {"size": 9, "color": "#94a3b8"}, "bgcolor": "rgba(0,0,0,0)"},
                xaxis={**ax, "showticklabels": False},
                yaxis={**ax, "range": [-0.05, 1.15]},
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    _sp(12)
    st.markdown('<div class="ac-section">HRV · ECG MORPHOLOGY</div>', unsafe_allow_html=True)
    h1, h2, h3, h4, h5 = st.columns(5)
    hrv_cards = [
        (h1, "RMSSD",  f"{s.rmssd:.0f}", " ms",  "#0d9488" if s.rmssd >= 20 else "#d97706"),
        (h2, "SDNN",   f"{s.sdnn:.0f}",  " ms",  "#2563eb" if s.sdnn  >= 50 else "#d97706"),
        (h3, "LF/HF",  f"{s.lf_hf:.2f}", "",     "#7c3aed"),
        (h4, "QTc",    f"{s.qtc:.0f}",   " ms",  "#dc2626" if s.qtc > 500 else "#d97706" if s.qtc > 460 else "#0d9488"),
        (h5, "pNN50",  f"{s.pnn50:.1f}", "%",    "#16a34a"),
    ]
    for col, lbl, val, unit, color in hrv_cards:
        with col:
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #e2e6eb;border-top:3px solid {color};'
                f'border-radius:12px;padding:16px 18px;">'
                f'<div class="ac-label">{lbl}</div>'
                f'<div style="font-size:1.9rem;font-weight:600;color:{color};'
                f'font-family:\'DM Mono\',monospace;">{val}'
                f'<span class="ac-unit">{unit}</span></div></div>',
                unsafe_allow_html=True,
            )

    fig2 = plot_trend("RMSSD — HRV (60s)", s.rmssd_hist, "#7c3aed", [0, 200],
                      [(20, "rgba(217,119,6,0.5)"), (50, "rgba(13,148,136,0.4)")])
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ── TAB: ECG COUPLING ─────────────────────────────────────────────────────────
def _tab_ecg() -> None:
    s = st.session_state
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="ac-section" style="margin-top:0">⚡ ECG COUPLING ANALYSIS</div>', unsafe_allow_html=True)

    # Synthetic ECG waveform
    fs, n = 250, 1250
    t_arr = np.linspace(s.t, s.t + n / fs, n)
    ecg   = np.zeros(n)
    for i, ti in enumerate(t_arr):
        ph = (ti * s.hr / 60.0) % 1.0
        if 0.10 < ph < 0.22: ecg[i] = 0.18 * np.sin(np.pi * (ph - 0.10) / 0.12)
        if 0.25 < ph < 0.35:
            p = (ph - 0.25) / 0.10
            ecg[i] = np.exp(-((p - 0.35) ** 2) / 0.005)
            if p < 0.2:  ecg[i] -= 0.3 * p
            if p > 0.6:  ecg[i] -= 0.15 * (p - 0.6)
        if 0.45 < ph < 0.65: ecg[i] = 0.28 * np.sin(np.pi * (ph - 0.45) / 0.20)
    ecg += np.random.normal(0, 0.012, n)

    ax = _ax()
    fig_ecg = go.Figure(go.Scatter(
        x=t_arr[:500], y=ecg[:500], mode="lines",
        line={"color": "#dc2626", "width": 1.5},
    ))
    fig_ecg.update_layout(
        **_base_layout(
            height=170,
            title={"text": f"ECG Waveform (lead II) — Twin HR {s.hr:.0f} bpm", "font": {"size": 11, "color": "#94a3b8"}},
            xaxis={**ax, "title": "Time (s)"},
            yaxis={**ax, "range": [-0.6, 1.3]},
        )
    )
    st.plotly_chart(fig_ecg, use_container_width=True, config={"displayModeBar": False})

    # PSD
    freqs = np.fft.rfftfreq(n, 1 / fs)
    psd   = np.abs(np.fft.rfft(ecg)) ** 2
    mask  = freqs < 3.0
    fig_psd = go.Figure(go.Scatter(
        x=freqs[mask], y=psd[mask], mode="lines", fill="tozeroy",
        line={"color": "#0d9488", "width": 1.8},
        fillcolor="rgba(13,148,136,0.07)",
    ))
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    if lf_mask.any():
        fig_psd.add_trace(go.Scatter(x=freqs[lf_mask], y=psd[lf_mask], fill="tozeroy",
                                      mode="lines", line={"color": "#d97706", "width": 0},
                                      fillcolor="rgba(217,119,6,0.18)", name="LF"))
    if hf_mask.any():
        fig_psd.add_trace(go.Scatter(x=freqs[hf_mask], y=psd[hf_mask], fill="tozeroy",
                                      mode="lines", line={"color": "#2563eb", "width": 0},
                                      fillcolor="rgba(37,99,235,0.18)", name="HF"))
    fig_psd.update_layout(
        **_base_layout(
            height=190, showlegend=True,
            title={"text": "HRV Power Spectral Density", "font": {"size": 11, "color": "#94a3b8"}},
            legend={"orientation": "h", "y": 1.1, "font": {"size": 9, "color": "#94a3b8"}, "bgcolor": "rgba(0,0,0,0)"},
            xaxis={**ax, "title": "Frequency (Hz)", "range": [0, 0.5]},
            yaxis={**ax},
        )
    )
    st.plotly_chart(fig_psd, use_container_width=True, config={"displayModeBar": False})

    _sp(8)
    st.markdown('<div class="ac-section">CARDIORESPIRATORY COUPLING INDICES</div>', unsafe_allow_html=True)
    coup_left, coup_right = st.columns([1.2, 1])

    with coup_left:
        m1, m2, m3, m4, m5 = st.columns(5)
        idx_cards = [
            (m1, "PSI",       f"{s.psi:.3f}",  "", "#0d9488"),
            (m2, "XCORR",     f"{s.xcorr:.3f}","", "#2563eb"),
            (m3, "COHERENCE", f"{s.coh:.3f}",  "", "#7c3aed"),
            (m4, "LF/HF",     f"{s.lf_hf:.2f}","","#d97706"),
            (m5, "RSA",       f"{s.rsa:.1f}",  " bpm", "#16a34a"),
        ]
        for col, lbl, val, unit, color in idx_cards:
            with col:
                st.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e2e6eb;'
                    f'border-top:3px solid {color};border-radius:12px;padding:14px 12px;">'
                    f'<div class="ac-label">{lbl}</div>'
                    f'<div style="font-size:1.5rem;font-weight:600;color:{color};'
                    f'font-family:\'DM Mono\',monospace;">{val}'
                    f'<span class="ac-unit">{unit}</span></div></div>',
                    unsafe_allow_html=True,
                )

        _sp(8)
        psi_interp  = "🟢 Strong synchrony" if s.psi > 0.6 else ("🟡 Moderate coupling" if s.psi > 0.3 else "🔴 Weak / decoupled")
        lf_hf_interp = "🟢 Balanced" if 0.5 < s.lf_hf < 2.0 else ("🟡 Sympathetic dom." if s.lf_hf >= 2.0 else "🟡 Vagal dom.")
        st.markdown(
            '<div style="background:#ffffff;border:1px solid #e2e6eb;border-radius:12px;padding:14px 16px;">'
            '<div class="ac-section" style="margin-top:0">COUPLING STRENGTH</div>'
            + coupling_bar("PSI",   s.psi,   0, 1,  "#0d9488")
            + coupling_bar("XCORR", s.xcorr, -1, 1, "#2563eb")
            + coupling_bar("COH",   s.coh,   0, 1,  "#7c3aed")
            + coupling_bar("RSA",   s.rsa,   0, 15, "#d97706")
            + f'<div style="font-size:0.72rem;color:#64748b;margin-top:8px;line-height:1.8">'
            f'PSI: {psi_interp}<br>n:m lock ≈ 4.2:1 · XCorr lag +1.8s<br>LF/HF: {lf_hf_interp}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with coup_right:
        eta_hr_secs   = _compute_eta_seconds(s.hr_hist,    140.0, "above")
        eta_spo2_secs = _compute_eta_seconds(s.spo2_hist,   90.0, "below")
        eta_o2d_secs  = _compute_eta_seconds(s.o2d_hist,    0.6,  "below")
        eta_hrv_secs  = _compute_eta_seconds(s.rmssd_hist,  20.0, "below")
        n_trending = sum(1 for x in [eta_hr_secs, eta_spo2_secs, eta_o2d_secs, eta_hrv_secs] if x is not None)
        conf_color = "#dc2626" if n_trending >= 3 else "#d97706" if n_trending >= 1 else "#0d9488"
        conf_label = "HIGH RISK" if n_trending >= 3 else "MODERATE RISK" if n_trending >= 1 else "LOW RISK"

        st.markdown(
            '<div style="background:#ffffff;border:1px solid #e2e6eb;border-radius:12px;padding:14px 16px;">'
            '<div style="color:#dc2626;font-size:0.68rem;font-weight:700;letter-spacing:2px;'
            'margin-bottom:10px;font-family:\'DM Mono\',monospace;">⏱ PREDICTED TIME TO THRESHOLD</div>'
            + _eta_card("HR → Tachycardia (140 bpm)", "HR", s.hr, 140, " bpm", eta_hr_secs, "above")
            + _eta_card("SpO₂ → Hypoxia (90%)",       "SpO₂", s.spo2, 90, "%",     eta_spo2_secs, "below")
            + _eta_card("O₂ Delivery → Critical (0.6)","O₂D", s.o2d, 0.6, "",      eta_o2d_secs,  "below")
            + _eta_card("HRV RMSSD → Low (<20 ms)",    "RMSSD", s.rmssd, 20, " ms", eta_hrv_secs, "below")
            + f'<div style="margin-top:8px;padding:8px 10px;border-radius:7px;'
            f'background:#f8fafc;border:1px solid #e2e6eb;">'
            f'<span style="color:{conf_color};font-size:0.72rem;font-weight:700;'
            f'font-family:\'DM Mono\',monospace;">{conf_label}</span>'
            f'<span style="color:#94a3b8;font-size:0.68rem;margin-left:8px;">'
            f'{n_trending}/4 parameters trending to threshold</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── TAB: SMARTWATCH ───────────────────────────────────────────────────────────
def _tab_smartwatch() -> None:
    s = st.session_state
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="ac-section" style="margin-top:0">⌚ SMARTWATCH INTEGRATION</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown('<div class="watch-panel">', unsafe_allow_html=True)
        st.markdown('<div class="ac-section" style="margin-top:0">DEVICE</div>', unsafe_allow_html=True)

        if not s.watch_connected:
            st.markdown("""
            <div style="text-align:center;padding:20px 0;">
                <div style="font-size:3rem;margin-bottom:8px">⌚</div>
                <div style="color:#94a3b8;font-size:0.78rem;">No device connected</div>
            </div>""", unsafe_allow_html=True)
            model = st.selectbox("Select Device", WATCH_MODELS, key="watch_select")
            if st.button("🔗 Connect Device", use_container_width=True, type="primary"):
                st.session_state.watch_connected   = True
                st.session_state.watch_model       = model
                st.session_state.watch_battery     = random.randint(55, 100)
                st.session_state.watch_steps       = random.randint(2000, 8000)
                st.session_state.watch_calories    = random.randint(200, 600)
                st.session_state.watch_hr          = s.hr + np.random.normal(0, 2)
                st.session_state.watch_spo2        = s.spo2 + np.random.normal(0, 0.3)
                st.session_state.watch_hrv         = s.rmssd + np.random.normal(0, 3)
                st.session_state.watch_skin_temp   = 36.6
                st.session_state.watch_hr_hist     = [float(s.hr)] * 60
                st.session_state.watch_spo2_hist   = [float(s.spo2)] * 60
                st.rerun()
        else:
            batt_col = "#16a34a" if s.watch_battery > 20 else "#dc2626"
            sync_str = s.watch_last_sync.strftime("%H:%M:%S") if s.watch_last_sync else "—"
            st.markdown(f"""
            <div style="text-align:center;padding:10px 0 14px;">
                <div style="font-size:3rem;margin-bottom:6px">⌚</div>
                <div style="color:#334155;font-size:0.78rem;font-weight:700;letter-spacing:1.5px">
                    {s.watch_model.upper()}</div>
                <div style="color:#16a34a;font-size:0.72rem;margin-top:3px;">● CONNECTED</div>
                <div style="margin-top:8px;font-size:0.75rem;color:{batt_col}">🔋 {s.watch_battery}%</div>
                <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px">Last sync: {sync_str}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(coupling_bar("SIGNAL", 0.92, 0, 1, "#0d9488"), unsafe_allow_html=True)
            st.markdown(coupling_bar("BATT", s.watch_battery / 100, 0, 1, batt_col), unsafe_allow_html=True)
            _sp(8)

            if st.button("🔌 Disconnect", use_container_width=True):
                st.session_state.watch_connected   = False
                st.session_state.watch_hr          = 0.0
                st.session_state.watch_spo2        = 0.0
                st.session_state.watch_hr_hist     = [0.0] * 60
                st.session_state.watch_spo2_hist   = [0.0] * 60
                st.rerun()

            ecg_ok  = any(b in s.watch_model for b in ["Apple", "Samsung", "Withings"])
            spo2_ok = any(b in s.watch_model for b in ["Apple", "Fitbit", "Withings", "Garmin"])
            st.markdown(f"""
            <div style="margin-top:10px;font-size:0.75rem;color:#64748b;line-height:1.9">
                <b style="color:#334155">Sensors active:</b><br>
                ✅ PPG (HR + SpO₂)<br>
                {'✅ ECG (1-lead)' if ecg_ok else '⚫ ECG N/A'}<br>
                ✅ Accelerometer<br>
                ✅ Skin Temperature<br>
                {'✅ Blood Oxygen' if spo2_ok else '⚫ SpO₂ N/A'}
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        if not s.watch_connected:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                height:400px;border:1px dashed #cbd5e1;border-radius:14px;
                flex-direction:column;gap:12px;color:#94a3b8;">
                <div style="font-size:3rem">⌚</div>
                <div style="font-size:0.85rem">Connect a smartwatch to see live vitals</div>
                <div style="font-size:0.72rem;color:#cbd5e1">Supports BLE, ANT+, and WiFi sync</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="ac-section" style="margin-top:0">LIVE WATCH VITALS</div>', unsafe_allow_html=True)
            w1, w2, w3, w4 = st.columns(4)
            hr_c = "#0d9488" if 50 <= s.watch_hr <= 100 else "#d97706" if s.watch_hr < 130 else "#dc2626"
            watch_vitals = [
                (w1, "⌚ HEART RATE",   f"{s.watch_hr:.0f}",    " bpm", hr_c,    f"Twin: {s.hr:.0f}"),
                (w2, "⌚ SpO₂",         f"{s.watch_spo2:.1f}",  "%",    "#0d9488",f"Twin: {s.spo2:.1f}"),
                (w3, "⌚ HRV (RMSSD)",  f"{s.watch_hrv:.0f}",   " ms",  "#7c3aed","PPG-derived"),
                (w4, "⌚ SKIN TEMP",    f"{s.watch_skin_temp:.1f}"," °C", "#d97706","Wrist sensor"),
            ]
            for col, lbl, val, unit, color, sub in watch_vitals:
                with col:
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #e2e6eb;'
                        f'border-top:3px solid {color};border-radius:12px;padding:16px 14px;">'
                        f'<div class="ac-label">{lbl}</div>'
                        f'<div style="font-size:1.9rem;font-weight:600;color:{color};'
                        f'font-family:\'DM Mono\',monospace;">{val}'
                        f'<span class="ac-unit">{unit}</span></div>'
                        f'<div class="ac-sub">{sub}</div></div>',
                        unsafe_allow_html=True,
                    )

            _sp(10)
            w5, w6, w7, w8 = st.columns(4)
            act_vitals = [
                (w5, "STEPS TODAY",   f"{s.watch_steps:,}",     "",       "#2563eb", f"{s.watch_steps * 0.0008:.2f} km"),
                (w6, "CALORIES",      f"{s.watch_calories}",    " kcal",  "#16a34a", "Active burn"),
                (w7, "ACTIVITY",      s.watch_activity,         "",       "#7c3aed", "Auto-detected"),
            ]
            hr_diff = abs(s.watch_hr - s.hr)
            sync_q  = "Excellent" if hr_diff < 3 else "Good" if hr_diff < 8 else "Fair"
            sync_c  = "#16a34a" if hr_diff < 3 else "#d97706" if hr_diff < 8 else "#dc2626"
            act_vitals.append((w8, "SYNC QUALITY", sync_q, "", sync_c, f"ΔHR {hr_diff:.1f} bpm"))
            for col, lbl, val, unit, color, sub in act_vitals:
                with col:
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #e2e6eb;'
                        f'border-top:3px solid {color};border-radius:12px;padding:16px 14px;">'
                        f'<div class="ac-label">{lbl}</div>'
                        f'<div style="font-size:1.7rem;font-weight:600;color:{color};'
                        f'font-family:\'DM Mono\',monospace;">{val}'
                        f'<span class="ac-unit">{unit}</span></div>'
                        f'<div class="ac-sub">{sub}</div></div>',
                        unsafe_allow_html=True,
                    )

            _sp(10)
            ax = _ax()
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=list(range(60)), y=s.hr_hist, mode="lines",
                                          name="Twin HR", line={"color": "#dc2626", "width": 2.0}))
            fig_cmp.add_trace(go.Scatter(x=list(range(60)), y=s.watch_hr_hist, mode="lines",
                                          name="⌚ Watch HR", line={"color": "#7c3aed", "width": 1.6, "dash": "dot"}))
            fig_cmp.update_layout(
                **_base_layout(
                    height=210, showlegend=True,
                    title={"text": "Twin vs Watch — HR Comparison (60s)", "font": {"size": 11, "color": "#94a3b8"}},
                    legend={"orientation": "h", "y": 1.12, "font": {"size": 9, "color": "#94a3b8"}, "bgcolor": "rgba(0,0,0,0)"},
                    xaxis={**ax, "showticklabels": False},
                    yaxis={**ax, "range": [30, 180]},
                )
            )
            st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)


# ── TAB: MANUAL ASSESSMENT ────────────────────────────────────────────────────
def _tab_manual() -> None:
    s = st.session_state
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="ac-section" style="margin-top:0">✍️ MANUAL ASSESSMENT · Risk Stratification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#f8fafc;border:1px solid #e2e6eb;border-radius:10px;'
        'padding:12px 16px;margin-bottom:16px;font-size:0.82rem;color:#64748b;line-height:1.7;">'
        'Enter custom physiological values to run static predictive risk stratification. '
        'Values are pre-filled from the current twin state.</div>',
        unsafe_allow_html=True,
    )

    with st.form("manual_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            hr_in   = st.number_input("Heart Rate (bpm)",      value=int(s.hr),    min_value=20, max_value=250)
            spo2_in = st.number_input("SpO₂ (%)",              value=int(s.spo2),  min_value=50, max_value=100)
            ef_in   = st.number_input("Ejection Fraction (%)", value=int(s.ef),    min_value=10, max_value=80)
        with fc2:
            rr_in      = st.number_input("Resp Rate (br/min)", value=int(s.rr),    min_value=2,  max_value=60)
            lactate_in = st.number_input("Lactate (mmol/L)",   value=1.5,          min_value=0.5,max_value=20.0, step=0.1)
            tv_in      = st.number_input("Tidal Volume (ml)",  value=int(s.tv),    min_value=100,max_value=1200)
        with fc3:
            rmssd_in = st.number_input("RMSSD (ms)",           value=int(s.rmssd), min_value=1,  max_value=300)
            psi_in   = st.number_input("PSI (0–1)",            value=round(s.psi, 2), min_value=0.0, max_value=1.0, step=0.01)
            hgb_in   = st.number_input("Haemoglobin (g/dL)",   value=14.0,         min_value=4.0,max_value=22.0, step=0.5)
        notes_in  = st.text_area("Clinical Notes (optional)", placeholder="Clinician observations...", height=68)
        submitted = st.form_submit_button("▶ RUN RISK STRATIFICATION", use_container_width=True, type="primary")

    if submitted:
        hr_s   = max(0.0, (hr_in - 60) / 80)
        rr_s   = max(0.0, (rr_in - 12) / 18)
        spo2_s = max(0.0, (100 - spo2_in) / 10)
        hrv_s  = max(0.0, 1 - rmssd_in / 60)
        lac_s  = max(0.0, (lactate_in - 1) / 5)
        psi_s  = max(0.0, 1 - psi_in)
        ef_s   = max(0.0, (50 - ef_in) / 30) if ef_in < 50 else 0.0
        risk   = hr_s * 0.18 + rr_s * 0.14 + spo2_s * 0.22 + hrv_s * 0.15 + lac_s * 0.12 + psi_s * 0.12 + ef_s * 0.07
        risk_pct = min(100.0, risk * 100)
        o2d_c  = min(1.5, (spo2_in / 100) * (hr_in / 74) * (hgb_in / 14) * 0.9)
        stab_c = max(0.0, min(1.0, rmssd_in / 42 * 0.3 + psi_in * 0.4 + 0.3))
        phase  = "CRITICAL" if risk_pct > 60 else "ELEVATED" if risk_pct > 35 else "STABLE"

        r1, r2, r3, r4 = st.columns(4)
        rclr = "#dc2626" if risk_pct > 60 else "#d97706" if risk_pct > 35 else "#0d9488"
        result_cards = [
            (r1, "RISK SCORE",  f"{risk_pct:.1f}", "/100",  rclr,    phase),
            (r2, "O₂ DELIVERY", f"{o2d_c:.3f}",   "",      "#16a34a","Fick estimate"),
            (r3, "STABILITY",   f"{stab_c:.3f}",   "",      "#2563eb","Composite"),
            (r4, "COUPLING",    f"{psi_in:.3f}",   "",      "#7c3aed","PSI input"),
        ]
        for col, lbl, val, unit, color, sub in result_cards:
            with col:
                st.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e2e6eb;'
                    f'border-top:3px solid {color};border-radius:12px;padding:16px 18px;">'
                    f'<div class="ac-label">{lbl}</div>'
                    f'<div style="font-size:2rem;font-weight:600;color:{color};'
                    f'font-family:\'DM Mono\',monospace;">{val}'
                    f'<span class="ac-unit">{unit}</span></div>'
                    f'<div class="ac-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

        if risk_pct > 60:   st.error(f"🔴 CRITICAL (Score: {risk_pct:.1f}/100) — Immediate clinical review required.")
        elif risk_pct > 35: st.warning(f"🟡 ELEVATED (Score: {risk_pct:.1f}/100) — Close monitoring advised.")
        else:               st.success(f"✅ STABLE (Score: {risk_pct:.1f}/100) — Parameters within normal range.")

        if s.watch_connected:
            delta = abs(s.watch_hr - hr_in)
            st.info(
                f"⌚ Watch HR at assessment: {s.watch_hr:.0f} bpm "
                f"(Δ {delta:.1f} bpm vs manual — {'consistent' if delta < 10 else 'discrepant'})"
            )

        _sp(12)
        st.markdown('<div class="ac-section">RISK FACTOR BREAKDOWN</div>', unsafe_allow_html=True)
        factors = {
            "Heart Rate": hr_s, "Resp Rate": rr_s, "SpO₂": spo2_s,
            "HRV": hrv_s, "Lactate": lac_s, "Phase Sync": psi_s, "EF": ef_s,
        }
        ax = _ax()
        fig = go.Figure(go.Bar(
            x=list(factors.keys()),
            y=[v * 100 for v in factors.values()],
            marker_color=["#dc2626" if v > 0.5 else "#d97706" if v > 0.25 else "#0d9488" for v in factors.values()],
            text=[f"{v * 100:.1f}" for v in factors.values()],
            textposition="outside",
        ))
        fig.update_layout(
            **_base_layout(
                height=200,
                yaxis={**ax, "range": [0, 105], "title": "Contribution (%)"},
                xaxis=ax,
                margin={"l": 40, "r": 10, "t": 20, "b": 28},
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        s.manual_results.insert(0, {
            "ts": datetime.utcnow(), "hr": hr_in, "spo2": spo2_in,
            "rr": rr_in, "rmssd": rmssd_in, "psi": psi_in,
            "risk_pct": risk_pct, "phase": phase, "o2d": o2d_c,
            "notes": notes_in,
            "watch_hr": s.watch_hr if s.watch_connected else None,
        })

    if s.manual_results:
        _sp(12)
        st.markdown('<div class="ac-section">PREVIOUS ASSESSMENTS</div>', unsafe_allow_html=True)
        for r in s.manual_results[:5]:
            bc_color = "#dc2626" if r["phase"] == "CRITICAL" else "#d97706" if r["phase"] == "ELEVATED" else "#0d9488"
            wt = f" · ⌚ {r['watch_hr']:.0f} bpm" if r.get("watch_hr") else ""
            rc_l, rc_r = st.columns([4, 1])
            with rc_l:
                notes_line = f'<br><span style="color:#94a3b8;font-size:0.72rem;">📝 {r["notes"]}</span>' if r.get("notes") else ""
                st.markdown(
                    '<div style="background:#f8fafc;border:1px solid #e2e6eb;border-radius:10px;'
                    'padding:10px 14px;margin-bottom:6px;">'
                    f'<span style="color:#0d9488;font-size:0.7rem;font-weight:700;font-family:\'DM Mono\',monospace;">'
                    f'{r["ts"].strftime("%H:%M:%S · %d %b %Y")}</span><br>'
                    f'<span style="color:#334155;font-size:0.78rem;">'
                    f'HR {r["hr"]} · SpO₂ {r["spo2"]}% · RMSSD {r["rmssd"]} ms · PSI {r["psi"]:.2f}{wt}</span>'
                    f'{notes_line}</div>',
                    unsafe_allow_html=True,
                )
            with rc_r:
                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #e2e6eb;border-top:3px solid {bc_color};'
                    f'border-radius:10px;padding:10px 14px;text-align:center;margin-bottom:6px;">'
                    f'<span style="color:{bc_color};font-size:0.75rem;font-weight:700;">{r["phase"]}</span><br>'
                    f'<span style="color:{bc_color};font-size:1.1rem;font-weight:700;font-family:\'DM Mono\',monospace;">'
                    f'{r["risk_pct"]:.0f}</span>'
                    f'<span style="color:#94a3b8;font-size:0.7rem;">/100</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)


# ── TAB: TWIN HISTORY ─────────────────────────────────────────────────────────
def _tab_history() -> None:
    s = st.session_state
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="ac-section" style="margin-top:0">📊 TWIN HISTORY · Session Log</div>', unsafe_allow_html=True)

    if not s.twin_history:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;
            height:300px;border:1px dashed #cbd5e1;border-radius:14px;
            flex-direction:column;gap:10px;color:#94a3b8;">
            <div style="font-size:2.5rem">📊</div>
            <div>No history yet — snapshots auto-save every 30 cycles</div>
            <div style="font-size:0.72rem;color:#cbd5e1">Use 💾 Save Snapshot in the sidebar to save manually</div>
        </div>""", unsafe_allow_html=True)
    else:
        h = s.twin_history
        all_hr   = [e["hr"]   for e in h]
        all_spo2 = [e["spo2"] for e in h]
        all_psi  = [e["psi"]  for e in h]
        all_o2d  = [e["o2d"]  for e in h]
        crit_count = sum(1 for e in h if e["phase"] == "CRITICAL")
        warn_count = sum(1 for e in h if e["phase"] == "WARNING")

        s1, s2, s3, s4, s5 = st.columns(5)
        summary_cards = [
            (s1, "SNAPSHOTS", str(len(h)),                 "",     "#0d9488","Session records"),
            (s2, "AVG HR",    f"{np.mean(all_hr):.1f}",   " bpm", "#dc2626",f"Range {min(all_hr):.0f}–{max(all_hr):.0f}"),
            (s3, "AVG SpO₂",  f"{np.mean(all_spo2):.1f}", "%",    "#0d9488",f"Min {min(all_spo2):.1f}%"),
            (s4, "CRITICAL",  str(crit_count),            " events","#dc2626" if crit_count else "#94a3b8",f"{warn_count} warnings"),
            (s5, "AVG PSI",   f"{np.mean(all_psi):.3f}",  "",     "#7c3aed","Coupling quality"),
        ]
        for col, lbl, val, unit, color, sub in summary_cards:
            with col:
                st.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e2e6eb;'
                    f'border-top:3px solid {color};border-radius:12px;padding:16px 18px;">'
                    f'<div class="ac-label">{lbl}</div>'
                    f'<div style="font-size:1.9rem;font-weight:600;color:{color};'
                    f'font-family:\'DM Mono\',monospace;">{val}'
                    f'<span class="ac-unit">{unit}</span></div>'
                    f'<div class="ac-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

        _sp(12)
        ax = _ax()
        idx      = list(range(len(h)))[::-1]
        hr_vals  = [e["hr"]   for e in reversed(h)]
        spo2_vals= [e["spo2"] for e in reversed(h)]

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=idx, y=hr_vals,   mode="lines+markers", name="HR",
                                    line={"color": "#dc2626", "width": 1.8}, marker={"size": 4}))
        fig_h.add_trace(go.Scatter(x=idx, y=spo2_vals, mode="lines+markers", name="SpO₂",
                                    line={"color": "#0d9488", "width": 1.8}, marker={"size": 4}))
        fig_h.update_layout(
            **_base_layout(
                height=210, showlegend=True,
                title={"text": "HR & SpO₂ — Session History", "font": {"size": 11, "color": "#94a3b8"}},
                legend={"orientation": "h", "y": 1.1, "font": {"size": 9, "color": "#94a3b8"}, "bgcolor": "rgba(0,0,0,0)"},
                xaxis={**ax, "title": "Snapshot index"},
                yaxis={**ax, "range": [50, 210]},
            )
        )
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

        psi_vals = [e["psi"] for e in reversed(h)]
        o2d_vals = [e["o2d"] for e in reversed(h)]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=idx, y=psi_vals, mode="lines+markers", name="PSI",
                                    line={"color": "#7c3aed", "width": 1.8}, marker={"size": 4},
                                    fill="tozeroy", fillcolor="rgba(124,58,237,0.05)"))
        fig_p.add_trace(go.Scatter(x=idx, y=o2d_vals, mode="lines+markers", name="O₂D",
                                    line={"color": "#16a34a", "width": 1.6}, marker={"size": 4}))
        fig_p.update_layout(
            **_base_layout(
                height=190, showlegend=True,
                title={"text": "PSI & O₂ Delivery — Session History", "font": {"size": 11, "color": "#94a3b8"}},
                legend={"orientation": "h", "y": 1.1, "font": {"size": 9, "color": "#94a3b8"}, "bgcolor": "rgba(0,0,0,0)"},
                xaxis={**ax, "title": "Snapshot index"},
                yaxis={**ax, "range": [0, 1.6]},
            )
        )
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="ac-section">SNAPSHOT LOG</div>', unsafe_allow_html=True)
        for entry in h[:50]:
            phase_color = "#dc2626" if entry["phase"] == "CRITICAL" else "#d97706" if entry["phase"] == "WARNING" else "#0d9488"
            sc_col = "#7c3aed" if entry["scenario"] != "Normal" else "#94a3b8"
            wh = f" · ⌚ {entry['watch_hr']:.0f}" if entry.get("watch_hr") else ""
            hl, hr_ = st.columns([4, 1])
            with hl:
                st.markdown(
                    '<div style="background:#f8fafc;border:1px solid #e2e6eb;border-radius:10px;'
                    'padding:10px 14px;margin-bottom:5px;">'
                    f'<span style="color:#0d9488;font-size:0.7rem;font-weight:700;font-family:\'DM Mono\',monospace;">'
                    f'{entry["ts"].strftime("%H:%M:%S · %d %b %Y")}</span>'
                    f'<span style="color:{sc_col};font-size:0.7rem;margin-left:10px;">'
                    f'Scenario: {entry["scenario"]} · Cycle #{entry["cycle"]}</span><br>'
                    f'<span style="color:#334155;font-size:0.77rem;">'
                    f'HR <b style="color:#dc2626">{entry["hr"]}</b> bpm · '
                    f'SpO₂ <b style="color:#0d9488">{entry["spo2"]}%</b> · '
                    f'RR {entry["rr"]} · O₂D {entry["o2d"]:.3f} · '
                    f'PSI {entry["psi"]:.3f} · RMSSD {entry["rmssd"]} ms{wh}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
            with hr_:
                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #e2e6eb;'
                    f'border-top:3px solid {phase_color};border-radius:10px;'
                    'padding:10px 14px;text-align:center;margin-bottom:5px;">'
                    f'<span style="color:{phase_color};font-size:0.78rem;font-weight:700;">{entry["phase"]}</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        if st.button("🗑️ Clear History"):
            st.session_state.twin_history = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ── TAB: LIGHT DASHBOARD (embedded HTML) ──────────────────────────────────────
def _tab_light_dashboard() -> None:
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="ac-section" style="margin-top:0">🌐 LIGHT DASHBOARD · Standalone View</div>',
        unsafe_allow_html=True,
    )

    # Search for the HTML file relative to this script, then cwd
    html_path: Optional[Path] = None
    candidates = [
        Path(__file__).parent / "aerocordis_light.html",
        Path(__file__).parent.parent / "aerocordis_light.html",
        Path.cwd() / "aerocordis_light.html",
    ]
    for candidate in candidates:
        if candidate.exists():
            html_path = candidate
            break

    if html_path is not None:
        html_content = html_path.read_text(encoding="utf-8")
        components.html(html_content, height=920, scrolling=True)
    else:
        st.warning(
            "**aerocordis_light.html not found.**\n\n"
            "Place `aerocordis_light.html` in the same directory as `dashboard.py` "
            f"(currently: `{Path(__file__).parent}`) and refresh this page."
        )
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:300px;border:1px dashed #cbd5e1;border-radius:14px;'
            'flex-direction:column;gap:10px;color:#94a3b8;">'
            '<div style="font-size:3rem">🌐</div>'
            '<div style="font-size:0.85rem">aerocordis_light.html not found in expected locations</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not st.session_state.logged_in:
        if st.session_state.get("show_register", False):
            view_register()
        else:
            view_login()
        return

    if st.session_state.get("just_logged_in", False):
        st.session_state.current_tab   = "🫀 Cardio Vitals"
        st.session_state._last_nav     = "🫀 Cardio Vitals"
        st.session_state._last_tool    = "✍️ Manual Assessment"
        st.session_state.just_logged_in = False

    view_dashboard()

    # ── Auto-sync on live tabs ────────────────────────────────────────────────
    live_tabs = {"🫀 Cardio Vitals", "🫁 Pulmonary Metrics", "⚡ ECG Coupling", "⌚ Smartwatch"}
    if st.session_state.live_feed and st.session_state.current_tab in live_tabs:
        time.sleep(1.8)
        _step_engine()
        st.rerun()


if __name__ == "__main__":
    main()