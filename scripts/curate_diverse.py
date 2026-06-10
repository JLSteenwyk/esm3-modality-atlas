"""Curate a taxonomically diverse multi-organism protein set.

The scaled atlas used ~900 hand-picked HUMAN proteins. To test whether modality
fusion is universal rather than human-specific, this samples reviewed (Swiss-Prot)
proteins across organisms spanning all three superkingdoms (eukaryota, bacteria,
archaea), each with an InterPro annotation (for the function modality) and length
50-800 (every one has an AlphaFold-DB structure).

Writes data/diverse/:
  metadata.json   {acc: {sequence, length, interpro, go_terms, organism,
                         category, superkingdom}}   (pipeline-compatible)
  accessions.txt  one accession per line
  curation_summary.json

Reuse the rest of the pipeline pointed at data/diverse (fetch_structures --dir,
annotate_structures --dir, harvest_scaled --config config/harvest_diverse.json).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "diverse"

# (taxonomy id, short label, superkingdom)
ORGANISMS = [
    (9606,   "human",       "eukaryota"),
    (10090,  "mouse",       "eukaryota"),
    (559292, "yeast",       "eukaryota"),
    (7227,   "fly",         "eukaryota"),
    (6239,   "worm",        "eukaryota"),
    (3702,   "arabidopsis", "eukaryota"),
    (7955,   "zebrafish",   "eukaryota"),
    (83333,  "ecoli",       "bacteria"),
    (83332,  "mtuberculosis", "bacteria"),
    (224308, "bsubtilis",   "bacteria"),
    (243232, "mjannaschii", "archaea"),
    (273057, "sulfolobus",  "archaea"),
]

PER_ORG = 500          # target sample per organism
POOL_PAGES = 6         # pages (x500) to pull before sampling, for diversity
SEARCH = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,length,xref_interpro,go_id,sequence"


def fetch_pool(taxid: int, session: requests.Session) -> list[dict]:
    """Pull up to POOL_PAGES*500 reviewed, InterPro-annotated proteins."""
    params = {
        "query": f"(organism_id:{taxid}) AND (reviewed:true) AND "
                 f"(length:[50 TO 800]) AND (database:interpro)",
        "fields": FIELDS, "format": "tsv", "size": 500,
    }
    rows: list[dict] = []
    url, page = SEARCH, 0
    while page < POOL_PAGES:
        r = session.get(url, params=params if page == 0 else None, timeout=40)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if len(lines) <= 1:
            break
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            acc, length, ipr, go, seq = parts[:5]
            ipr_ids = [x for x in ipr.split(";") if x.strip()]
            if not ipr_ids or not seq:
                continue
            rows.append({
                "accession": acc, "length": int(length), "sequence": seq,
                "interpro": ipr_ids,
                "go_terms": [{"id": g.strip()} for g in go.split(";") if g.strip()],
            })
        link = r.headers.get("Link", "")
        if 'rel="next"' in link:
            url = link.split(";")[0].strip("<>")
            page += 1
        else:
            break
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    session = requests.Session()
    metadata: dict[str, dict] = {}
    summary: dict[str, dict] = {}

    for taxid, label, kingdom in ORGANISMS:
        try:
            pool = fetch_pool(taxid, session)
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: FETCH ERROR {e}")
            summary[label] = {"error": str(e)}
            continue
        rng.shuffle(pool)
        sample = pool[:PER_ORG]
        for row in sample:
            metadata[row["accession"]] = {
                **row, "organism": label, "category": label,
                "superkingdom": kingdom,
            }
        summary[label] = {"taxid": taxid, "superkingdom": kingdom,
                          "pool": len(pool), "sampled": len(sample)}
        print(f"  {label:<14} ({kingdom:<10}) pool={len(pool):>4} sampled={len(sample)}")
        time.sleep(0.3)

    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata))
    (OUT_DIR / "accessions.txt").write_text("\n".join(sorted(metadata)) + "\n")
    (OUT_DIR / "curation_summary.json").write_text(json.dumps(
        {"n_total": len(metadata), "per_organism": summary,
         "n_organisms": len(ORGANISMS)}, indent=2))
    print(f"\ncurated {len(metadata)} proteins across {len(ORGANISMS)} organisms "
          f"-> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
