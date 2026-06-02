#!/usr/bin/env python3
"""
download_waveforms.py
─────────────────────
Download MIMIC-IV waveform records via wfdb.

MIMIC-IV Waveform DB path structure:
  PN_DIR  = mimic4wdb/0.1.0
  Subject = waves/p<3-digit-prefix>/p<subject_id>/
  Record  = <stay>/<record>
  Example: subject 10014354, stay 81739927
    pn_dir   = "mimic4wdb/0.1.0/waves/p100/p10014354"
    rec_name = "81739927/81739927"

Usage:
    python download_waveforms.py --subject 10014354
    python download_waveforms.py --subject 10014354 --stay 81739927
    python download_waveforms.py --list-only

Requirements:
    pip install wfdb
    Set PHYSIONET_USER and PHYSIONET_PASS env vars,
    or fill config/credentials.yaml
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

try:
    import wfdb
except ImportError:
    print("ERROR: wfdb not installed. Run:  pip install wfdb")
    sys.exit(1)

# ── PhysioNet auth ────────────────────────────────────────────────────────────

def authenticate() -> bool:
    """Authenticate to PhysioNet using env vars or credentials.yaml."""
    username = os.environ.get("PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASS")

    if not username:
        # Try credentials.yaml relative to this script
        cred_path = Path(__file__).parent / "config" / "credentials.yaml"
        if cred_path.exists():
            import yaml
            creds = yaml.safe_load(cred_path.read_text()) or {}
            pn = creds.get("physionet", {})
            username = pn.get("username")
            password = pn.get("password")

    if not username or not password:
        print("ERROR: PhysioNet credentials not found.")
        print("  Set PHYSIONET_USER and PHYSIONET_PASS environment variables,")
        print("  or fill config/credentials.yaml")
        return False

    if hasattr(wfdb, "set_db_auth"):
        wfdb.set_db_auth(username, password)  # type: ignore[attr-defined]
    elif hasattr(wfdb, "io") and hasattr(wfdb.io, "set_db_auth"):  # type: ignore[attr-defined]
        wfdb.io.set_db_auth(username, password)  # type: ignore[attr-defined]
    else:
        os.environ["PHYSIONET_USER"] = username
        os.environ["PHYSIONET_PASS"] = password
    print(f"Authenticated as: {username}")
    return True


# ── Path helpers ──────────────────────────────────────────────────────────────

def subject_pndir(subject_id: str) -> str:
    """Return the pn_dir for a given subject_id."""
    prefix = subject_id[:3]
    return f"mimic4wdb/0.1.0/waves/p{prefix}/p{subject_id}"


# ── Main logic ────────────────────────────────────────────────────────────────

def list_records(subject_id: str) -> list[str]:
    """List all waveform records for a subject."""
    pndir = subject_pndir(subject_id)
    print(f"Listing records at: {pndir}")
    try:
        records: list[str] = list(wfdb.get_record_list(pndir))  # type: ignore[arg-type,call-arg]
        print(f"Found {len(records)} record(s): {records}")
        return records
    except Exception as e:
        print(f"ERROR listing records: {e}")
        return []


def download_record(subject_id: str, record_name: str, out_dir: str = "mimic4wdb") -> bool:
    """Download one waveform record to local disk."""
    pndir = subject_pndir(subject_id)
    print(f"Downloading: {record_name}")
    try:
        full_record_path = f"waves/p{subject_id[:3]}/p{subject_id}/{record_name}"
               
        wfdb.dl_database(
            db_dir="mimic4wdb",
            dl_dir=out_dir,
            records=[full_record_path],  # type: ignore[arg-type]
        )
        print(f"  ✓ Saved to {out_dir}/{record_name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MIMIC-IV waveforms via wfdb")
    parser.add_argument("--subject", required=True, help="MIMIC-IV subject_id (e.g. 10014354)")
    parser.add_argument("--stay",    default=None,  help="Specific stay/record to download")
    parser.add_argument("--out-dir", default="mimic4wdb", help="Local output directory")
    parser.add_argument("--list-only", action="store_true", help="Only list records, don't download")
    args = parser.parse_args()

    if not authenticate():
        sys.exit(1)

    records = list_records(args.subject)
    if not records:
        sys.exit(1)

    if args.list_only:
        print("\nAvailable records:")
        for r in records:
            print(f"  {r}")
        return

    # Filter to requested stay
    to_download = records
    if args.stay:
        to_download = [r for r in records if args.stay in r]
        if not to_download:
            print(f"No records match stay '{args.stay}'. Available: {records}")
            sys.exit(1)

    downloaded, skipped = 0, 0
    for rec in to_download:
        if download_record(args.subject, rec, args.out_dir):
            downloaded += 1
        else:
            skipped += 1

    print(f"\nDownload complete. Downloaded: {downloaded} | Skipped: {skipped}")


if __name__ == "__main__":
    main()