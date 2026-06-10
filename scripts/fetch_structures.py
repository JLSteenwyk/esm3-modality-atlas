"""Fetch AlphaFold-DB structures for the scaled accession set.

Resolves each UniProt accession through the AFDB API (so we always get the
current model version rather than guessing a vN URL), downloads the PDB to
data/scaled/structures/<ACC>.pdb, and logs anything AFDB can't serve.

Resumable: skips accessions whose PDB already exists. Run with an integer arg
to cap the number processed (for a quick test), e.g. `python fetch_structures.py 5`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "scaled"   # overridden by --dir in main()
ACC_FILE = DATA_DIR / "accessions.txt"
OUT_DIR = DATA_DIR / "structures"
LOG_DIR = DATA_DIR / "logs"

API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
TIMEOUT = 30
PAUSE = 0.15          # polite delay between accessions
RETRIES = 3


def fetch_one(acc: str, session: requests.Session) -> tuple[str, str]:
    """Return (status, detail). status in {ok, exists, no_entry, error}."""
    out = OUT_DIR / f"{acc}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return "exists", ""
    for attempt in range(RETRIES):
        try:
            r = session.get(API.format(acc=acc), timeout=TIMEOUT)
            if r.status_code == 404:
                return "no_entry", "AFDB 404"
            r.raise_for_status()
            entries = r.json()
            if not entries:
                return "no_entry", "empty API response"
            pdb_url = entries[0].get("pdbUrl")
            if not pdb_url:
                return "no_entry", "no pdbUrl"
            pr = session.get(pdb_url, timeout=TIMEOUT)
            pr.raise_for_status()
            out.write_text(pr.text)
            ver = pdb_url.rsplit("model_", 1)[-1].replace(".pdb", "")
            return "ok", ver
        except Exception as e:  # noqa: BLE001
            if attempt == RETRIES - 1:
                return "error", f"{type(e).__name__}: {e}"
            time.sleep(1.0 * (attempt + 1))
    return "error", "unreachable"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/scaled",
                    help="dataset dir holding accessions.txt; writes structures/ + logs/")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    global ACC_FILE, OUT_DIR, LOG_DIR
    d = ROOT / args.dir
    ACC_FILE, OUT_DIR, LOG_DIR = d / "accessions.txt", d / "structures", d / "logs"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    accs = ACC_FILE.read_text().split()
    limit = args.limit if args.limit else len(accs)
    accs = accs[:limit]
    print(f"fetching {len(accs)} accessions -> {OUT_DIR.relative_to(ROOT)}")

    session = requests.Session()
    counts: dict[str, int] = {}
    missing: list[dict] = []
    versions: dict[str, int] = {}
    t0 = time.time()
    for i, acc in enumerate(accs, 1):
        status, detail = fetch_one(acc, session)
        counts[status] = counts.get(status, 0) + 1
        if status == "ok":
            versions[detail] = versions.get(detail, 0) + 1
            time.sleep(PAUSE)
        elif status in ("no_entry", "error"):
            missing.append({"accession": acc, "status": status, "detail": detail})
        if i % 50 == 0 or i == len(accs):
            print(f"  {i}/{len(accs)}  {dict(sorted(counts.items()))}  "
                  f"({time.time() - t0:.0f}s)")

    (LOG_DIR / "fetch_missing.json").write_text(json.dumps(missing, indent=2))
    summary = {"requested": len(accs), "counts": counts, "versions": versions,
               "n_missing": len(missing), "elapsed_sec": time.time() - t0}
    (LOG_DIR / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone: {counts}")
    print(f"versions: {versions}")
    if missing:
        print(f"{len(missing)} missing -> {(LOG_DIR / 'fetch_missing.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
