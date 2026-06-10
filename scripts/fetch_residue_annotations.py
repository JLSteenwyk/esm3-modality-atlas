"""Fetch InterPro residue-site annotations for the harvested proteins.

Queries the InterPro API endpoint
  /api/protein/uniprot/{acc}/?residues
which returns, per member-database entry, conserved-site `locations` each with a
`description` (e.g. "BINDING: ATP", "ACT_SITE: Proton acceptor") and `fragments`
giving the annotated `residues` identity, `start`, `end`. We flatten these into
the parallel lists ESM3's ResidueAnnotationsTokenizer expects.

Writes data/scaled/annotations/residue_sites.json:
  { "<ACC>": {"descriptions": [...], "starts": [...], "ends": [...], "residues": [...]}, ... }
Only proteins that actually have site annotations are included.

Resumable; run with an integer arg to cap the number processed (testing).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MP_DIR = ROOT / "activations" / "scaled" / "per_protein"
OUT = ROOT / "data" / "scaled" / "annotations" / "residue_sites.json"
LOG = ROOT / "data" / "scaled" / "logs" / "residue_sites_summary.json"

API = "https://www.ebi.ac.uk/interpro/api/protein/uniprot/{acc}/?residues"
TIMEOUT = 25
PAUSE = 0.1
RETRIES = 3


def fetch_one(acc: str, session: requests.Session):
    """Return flattened site lists, or None if no site annotations / error."""
    for attempt in range(RETRIES):
        try:
            r = session.get(API.format(acc=acc), timeout=TIMEOUT)
            if r.status_code == 204 or r.status_code == 404:
                return {}        # no annotations (definitive)
            r.raise_for_status()
            data = r.json()
            descs, starts, ends, residues = [], [], [], []
            for entry in data.values():
                for loc in entry.get("locations", []):
                    desc = loc.get("description") or ""
                    for frag in loc.get("fragments", []):
                        res = frag.get("residues")
                        s, e = frag.get("start"), frag.get("end")
                        if res is None or s is None or e is None:
                            continue
                        descs.append(desc)
                        starts.append(int(s))
                        ends.append(int(e))
                        residues.append(res)
            if not descs:
                return {}
            return {"descriptions": descs, "starts": starts,
                    "ends": ends, "residues": residues}
        except Exception:  # noqa: BLE001
            if attempt == RETRIES - 1:
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    accs = sorted(p.stem for p in MP_DIR.glob("*.npz"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(accs)
    accs = accs[:limit]

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    done_set = set(out)  # already-fetched with sites
    print(f"fetching residue sites for {len(accs)} proteins")

    session = requests.Session()
    n_with, n_without, n_err = len(out), 0, 0
    t0 = time.time()
    for i, acc in enumerate(accs, 1):
        if acc in done_set:
            continue
        res = fetch_one(acc, session)
        if res is None:
            n_err += 1
        elif res == {}:
            n_without += 1
        else:
            out[acc] = res
            n_with += 1
        time.sleep(PAUSE)
        if i % 100 == 0 or i == len(accs):
            print(f"  {i}/{len(accs)}  with_sites={n_with} none={n_without} "
                  f"err={n_err}  ({time.time() - t0:.0f}s)")

    OUT.write_text(json.dumps(out))
    summary = {"n_requested": len(accs), "n_with_sites": len(out),
               "n_without": n_without, "n_errors": n_err,
               "elapsed_sec": time.time() - t0}
    LOG.write_text(json.dumps(summary, indent=2))
    print(f"\ndone: {len(out)} proteins with site annotations -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
