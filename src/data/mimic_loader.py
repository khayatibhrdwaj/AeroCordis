"""
src/data/mimic_loader.py
─────────────────────────
MIMIC-IV Waveform Database loader via PhysioNet / WFDB.

MIMIC-IV Waveform DB path structure (mimic4wdb/0.1.0):
  PN_DIR root = mimic4wdb/0.1.0
  Subject dir = waves/p<3-digit-prefix>/p<subject_id>/
      e.g. subject 10014354 → waves/p100/p10014354/
  Record      = <stay_folder>/<record_name>
      e.g.                  → 81739927/81739927
  Numerics    = <stay_folder>/<record_name>n
      e.g.                  → 81739927/81739927n

  wfdb calls: wfdb.rdheader("81739927/81739927",
                              pn_dir="mimic4wdb/0.1.0/waves/p100/p10014354")

Usage
-----
    loader = MIMICLoader(cfg)
    loader.authenticate()
    for chunk in loader.stream_waveforms(subject_id, stay_id):
        engine.ingest(chunk.ecg, chunk.resp, chunk.spo2, chunk.timestamp)
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Generator, List, Optional
import numpy as np
import yaml
from loguru import logger

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
    logger.warning("wfdb not installed — MIMIC waveform loading unavailable. Run: pip install wfdb")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WaveformChunk:
    """One chunk of multi-channel waveform data."""
    ecg: np.ndarray
    resp: np.ndarray
    spo2: np.ndarray
    abp: Optional[np.ndarray] = None
    timestamp: Optional[datetime] = None
    subject_id: Optional[str] = None
    stay_id: Optional[str] = None
    chunk_index: int = 0
    fs_ecg: float = 125.0
    fs_resp: float = 62.5
    fs_spo2: float = 1.0


# ── Loader ────────────────────────────────────────────────────────────────────

class MIMICLoader:
    """
    Streams waveform and numerics data from MIMIC-IV Waveform DB (mimic4wdb/0.1.0).

    Credentials are loaded from config/credentials.yaml.
    Requires wfdb and PhysioNet credentialed access.
    """

    _CRED_PATH = Path(__file__).parent.parent.parent / "config" / "credentials.yaml"
    # Root pn_dir for all wfdb calls
    _PNDIR_ROOT = "mimic4wdb/0.1.0"

    def __init__(self, cfg):
        from ..core.config import TwinConfig
        self.cfg: TwinConfig = cfg
        self._chunk_s = cfg.mimic.chunk_size_s
        self._authenticated = False
        self._username: Optional[str] = None
        self._password: Optional[str] = None

    # ── Path helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _subject_pndir(subject_id: str) -> str:
        """
        Build the pn_dir argument for wfdb calls for a given subject.
        MIMIC-IV groups subjects: p<first_3_digits>/p<subject_id>
        e.g. subject_id='10014354' → 'mimic4wdb/0.1.0/waves/p100/p10014354'
        """
        prefix = subject_id[:3]
        return f"mimic4wdb/0.1.0/waves/p{prefix}/p{subject_id}"

    # ── Authentication ────────────────────────────────────────────────────────

    def authenticate(
        self, username: Optional[str] = None, password: Optional[str] = None
    ) -> bool:
        """Load PhysioNet credentials and configure wfdb authentication via .netrc."""
        import os
        from pathlib import Path
        import yaml
        
        if username and password:
            self._username, self._password = username, password
        else:
            self._username = os.environ.get("PHYSIONET_USER")
            self._password = os.environ.get("PHYSIONET_PASS")
            if not self._username and self._CRED_PATH.exists():
                with open(self._CRED_PATH) as f:
                    creds = yaml.safe_load(f) or {}
                pn = creds.get("physionet", {})
                self._username = pn.get("username")
                self._password = pn.get("password")

        if not self._username or not self._password:
            logger.error("PhysioNet credentials not found.")
            return False

        # Build the .netrc file so wfdb's underlying requests can authenticate
        try:
            netrc_name = "_netrc" if os.name == "nt" else ".netrc"
            netrc_path = Path.home() / netrc_name
            
            with open(netrc_path, "w") as f:
                f.write(f"machine physionet.org login {self._username} password {self._password}\n")
            
            # Protect the file (required on Mac/Linux)
            if os.name != "nt":
                os.chmod(netrc_path, 0o600)
                
            logger.info(f"Authenticated to PhysioNet as: {self._username} (via {netrc_name})")
            self._authenticated = True
            return True
        except Exception as e:
            logger.error(f"Authentication setup failed: {e}")
            return False

    # ── Record discovery ──────────────────────────────────────────────────────

    def list_records(self, subject_id: str) -> List[str]:
        """
        List waveform records for a subject.

        Returns record names relative to the subject pn_dir,
        e.g. ['81739927/81739927'].
        """
        if not self._check_ready():
            return []
        pndir = self._subject_pndir(subject_id)
        try:
            records: List[str] = list(wfdb.get_record_list(pndir))  # type: ignore[arg-type]
            logger.info(f"Subject {subject_id}: {len(records)} record(s) → {records}")
            return records
        except Exception as e:
            logger.warning(f"Record list failed for {subject_id} (pn_dir={pndir}): {e}")
            return []

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream_waveforms(self, subject_id: str, stay_id: str, max_chunks: int = 1200):
        """Streams waveform data across multiple MIMIC-IV segments seamlessly."""
        import wfdb
        import numpy as np
        from dataclasses import dataclass
        from loguru import logger

        @dataclass
        class WaveformChunk:
            ecg: np.ndarray
            resp: np.ndarray
            spo2: np.ndarray
            timestamp: float

        # Hardcode the standard PhysioNet path to avoid config type errors
        subject_group = str(subject_id)[:3]
        pn_dir = f"mimic4wdb/0.1.0/waves/p{subject_group}/p{subject_id}/{stay_id}"
        master_record = stay_id

        logger.info(f"Reading Master Header: {master_record} | pn_dir={pn_dir}")
        
        try:
            # 1. Read the Master Header
            master_header = wfdb.rdheader(record_name=master_record, pn_dir=pn_dir)
            # Use getattr to silence Pylance warnings on untyped wfdb objects
            segments = getattr(master_header, 'seg_name', [])
        except Exception as e:
            logger.error(f"Failed to read master header: {e}")
            return

        chunks_yielded = 0
        chunk_size_s = 0.25 # 250ms chunks
        current_time = 0.0

        # 2. Loop through every single file in the ICU stay sequentially
        for seg_name in segments:
            if chunks_yielded >= max_chunks:
                break
                
            if seg_name == '~':
                continue # Skip dead air/gap segments
                
            # 🚨 THE FIX: Just use the segment name directly
            seg_record_name = seg_name
            logger.info(f"Opening segment: {seg_record_name}")
            
            try:
                rec = wfdb.rdrecord(record_name=seg_record_name, pn_dir=pn_dir)
            except Exception as e:
                logger.warning(f"Failed to read {seg_name}. Skipping...")
                continue

            # Safely extract attributes to keep Pylance happy
            sig_len = getattr(rec, 'sig_len', 0)
            p_signal = getattr(rec, 'p_signal', None)

            # 3. Skip empty calibration files (like 0017, 0018, 0019)
            if sig_len == 0 or p_signal is None:
                logger.warning(f"Segment {seg_name} is empty. Skipping forward...")
                continue

            # Identify channels dynamically
            sig_names = [n.lower() for n in getattr(rec, 'sig_name', [])]
            ecg_idx = next((i for i, n in enumerate(sig_names) if 'ii' in n or 'ecg' in n), 0)
            resp_idx = next((i for i, n in enumerate(sig_names) if 'resp' in n), 1)
            spo2_idx = next((i for i, n in enumerate(sig_names) if 'pleth' in n or 'spo2' in n), 2)

            fs = getattr(rec, 'fs', 125)
            samples_per_chunk = int(fs * chunk_size_s)

            # 4. Slice the segment and stream
            for start_idx in range(0, sig_len - samples_per_chunk, samples_per_chunk):
                if chunks_yielded >= max_chunks:
                    break
                    
                end_idx = start_idx + samples_per_chunk
                
                ecg_data = np.nan_to_num(p_signal[start_idx:end_idx, ecg_idx])
                resp_data = np.nan_to_num(p_signal[start_idx:end_idx, resp_idx])
                spo2_data = np.nan_to_num(p_signal[start_idx:end_idx, spo2_idx])
                
                yield WaveformChunk(ecg_data, resp_data, spo2_data, current_time)
                
                chunks_yielded += 1
                current_time += chunk_size_s

    # ── Numerics (1 Hz values) ────────────────────────────────────────────────

    def load_numerics(
        self,
        subject_id: str,
        record_name: Optional[str] = None,
    ) -> dict:
        """Load 1 Hz HR/RR/SpO₂ numerics from the 'n'-suffixed record."""
        if not self._check_ready():
            return {}
        pndir = self._subject_pndir(subject_id)
        records = self.list_records(subject_id) if not record_name else [record_name]
        if not records:
            return {}

        # Prefer the record that ends in 'n'
        num_recs = [r for r in records if r.rstrip("/").endswith("n")]
        if not num_recs:
            # Attempt: append 'n' to the first waveform record basename
            base = records[0]
            num_recs = [base + "n"]

        try:
            record = wfdb.rdrecord(local_record_path, sampfrom=0, sampto=10000)  # type: ignore[arg-type]
            sig_names_raw = getattr(record, "sig_name", []) or []
            sig_names = [str(s).lower().strip() for s in sig_names_raw]
            col_map = {
                "heart rate": "heart_rate", "hr": "heart_rate",
                "respiratory rate": "respiratory_rate", "resp rate": "respiratory_rate",
                "spo2": "spo2", "spco2": "spo2",
            }
            result = {}
            p_sig: Optional[np.ndarray] = getattr(record, "p_signal", None)
            if p_sig is not None:
                p_sig = np.asarray(p_sig, dtype=float)
                for raw, clean in col_map.items():
                    if raw in sig_names and clean not in result:
                        idx = sig_names.index(raw)
                        result[clean] = p_sig[:, idx]
            logger.info(f"Numerics loaded: {list(result.keys())}")
            return result
        except Exception as e:
            logger.warning(f"Numerics load failed: {e}")
            return {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_ready(self) -> bool:
        if not HAS_WFDB:
            logger.error("wfdb not installed")
            return False
        if not self._authenticated:
            logger.error("Not authenticated — call authenticate() first")
            return False
        return True

    @staticmethod
    def _map_channels(sig_names: List[str]) -> dict:
        """Map physiological role → column index in the wfdb record."""
        desired = [
            ("ii",    "ecg"),
            ("ecg",   "ecg"),
            ("resp",  "resp"),
            ("pleth", "spo2"),
            ("spo2",  "spo2"),
            ("abp",   "abp"),
            ("art",   "abp"),
        ]
        result: dict = {}
        for col_idx, name in enumerate(sig_names):
            for key, role in desired:
                if key in name and role not in result:
                    result[role] = col_idx
                    break
        return result

    @staticmethod
    def _build_chunk(
        signals: np.ndarray,
        ch_map: dict,          # {role: column_index_in_signals_array}
        fs: float,
        timestamp: Optional[datetime],
        subject_id: str,
        stay_id: Optional[str],
        chunk_idx: int,
    ) -> WaveformChunk:
        n = signals.shape[0] if signals.ndim == 2 else len(signals)

        def extract(role: str) -> np.ndarray:
            if role not in ch_map:
                return np.zeros(n)
            col = ch_map[role]
            if signals.ndim == 2 and col < signals.shape[1]:
                arr = signals[:, col].astype(float)
            elif signals.ndim == 1:
                arr = signals.astype(float)
            else:
                return np.zeros(n)
            return np.where(np.isnan(arr), 0.0, arr)

        return WaveformChunk(
            ecg=extract("ecg"),
            resp=extract("resp"),
            spo2=extract("spo2"),
            abp=extract("abp") if "abp" in ch_map else None,
            timestamp=timestamp,
            subject_id=subject_id,
            stay_id=stay_id,
            chunk_index=chunk_idx,
            fs_ecg=float(fs),
            fs_resp=float(fs) / 2.0,
            fs_spo2=1.0,
        )

    @staticmethod
    def _sample_to_datetime(header, sample: int, fs: float) -> Optional[datetime]:
        try:
            if hasattr(header, "base_datetime") and header.base_datetime:
                return header.base_datetime + timedelta(seconds=sample / fs)
        except Exception:
            pass
        return None