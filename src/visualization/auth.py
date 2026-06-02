"""
src/visualization/auth.py
AeroCordis — Authentication & Patient Registry
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple

# ── Storage path ──────────────────────────────────────────────────────────────
# Patients are stored in a JSON file next to this module.
_REGISTRY_PATH = Path(__file__).parent / "patients.json"


def _load_registry() -> dict:
    if _REGISTRY_PATH.exists():
        try:
            with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_registry(registry: dict) -> None:
    with open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _hash_password(password: str) -> str:
    """Return a SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def register_patient(
    uid: str,
    full_name: str,
    age: int,
    gender: str,
    password: str,
) -> bool:
    """
    Register a new patient.  Returns True on success, False if the UID
    already exists (which should never happen given _gen_uid, but is a
    safe-guard).
    """
    registry = _load_registry()
    if uid in registry:
        return False
    registry[uid] = {
        "full_name": full_name,
        "age": age,
        "gender": gender,
        "password_hash": _hash_password(password),
    }
    _save_registry(registry)
    return True


def authenticate(uid: str, password: str) -> Tuple[bool, Optional[dict]]:
    """
    Validate a Patient ID / Access Key pair.

    Returns:
        (True,  patient_record)  on success.
        (False, None)            on failure.
    """
    registry = _load_registry()
    record = registry.get(uid.strip())
    if record is None:
        return False, None
    if record["password_hash"] == _hash_password(password):
        return True, record
    return False, None


def patient_exists(uid: str) -> bool:
    """Return True if the UID is already in the registry."""
    return uid.strip() in _load_registry()