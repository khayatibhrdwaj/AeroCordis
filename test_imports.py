#!/usr/bin/env python3
"""
test_imports.py
───────────────
Verify all modules can be imported without errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("CARDIOPULMONARY TWIN — Import Verification")
print("=" * 70)

tests = [
    ("Config Loader", "from src.core.config import TwinConfig, load_config"),
    ("Twin State", "from src.core.twin_state import TwinState, AlertEvent"),
    ("ECG Processor", "from src.signals.ecg_processor import ECGProcessor"),
    ("Respiratory Processor", "from src.signals.resp_processor import RespProcessor"),
    ("SpO2 Processor", "from src.signals.spo2_processor import SpO2Processor"),
    ("RSA Analyzer", "from src.coupling.rsa import RSAAnalyzer"),
    ("Cross-Correlation", "from src.coupling.cross_correlation import HRRRCrossCorrelation"),
    ("Phase Sync", "from src.coupling.phase_sync import PhaseSynchronisation"),
    ("Hypoxia Response", "from src.coupling.hypoxia_response import HypoxiaResponseMonitor"),
    ("Alert Engine", "from src.alerts.alert_engine import AlertEngine"),
    ("Trend Predictor", "from src.prediction.trend_predictor import TrendPredictor"),
    ("Twin Engine", "from src.core.twin_engine import TwinEngine"),
    ("Math Helpers", "from src.utils.math_helpers import compute_o2_delivery, compute_stability, compute_stress"),
]

passed = 0
failed = 0

for name, import_stmt in tests:
    try:
        exec(import_stmt)
        print(f"✓ {name:.<40} OK")
        passed += 1
    except Exception as e:
        print(f"✗ {name:.<40} FAILED")
        print(f"  Error: {str(e)[:100]}")
        failed += 1

print("=" * 70)
print(f"Result: {passed}/{len(tests)} modules loaded successfully")
print("=" * 70)

if failed == 0:
    print("✓ All imports successful! Ready to run demos.")
    sys.exit(0)
else:
    print(f"✗ {failed} module(s) failed. Fix errors above.")
    sys.exit(1)
