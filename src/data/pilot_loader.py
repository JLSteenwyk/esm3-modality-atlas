"""Read the curated multimodal pilot dataset.

The pilot lives in ``data/pilot/``. 938 proteins are catalogued in
``annotations/metadata.json``; 199 have PDB
files staged in ``structures/``. We restrict to that 199-protein cohort so
every protein supports every modality condition uniformly.

DSSP entries are per-residue dicts with keys ``{resnum, aa, ss8, ss3, asa, rsa}``;
we collapse each protein's DSSP rows into one SS8 string and one SASA list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class PilotProtein:
    """One protein worth of multimodal inputs the harvest needs."""

    accession: str
    sequence: str
    length: int
    category: str
    pdb_path: Path
    ss8: Optional[str]
    sasa: Optional[list[float]]
    # Whole-protein-level annotation IDs (for future function-token wiring)
    interpro_ids: list[str]
    pfam_ids: list[str]
    go_ids: list[str]


def load_pilot_index(pilot_dir: Path) -> dict:
    """Return the raw metadata dict (accession → entry)."""
    meta_path = pilot_dir / "annotations" / "metadata.json"
    return json.loads(meta_path.read_text())


def _collapse_dssp(rows: list[dict], expected_len: int) -> tuple[Optional[str], Optional[list[float]]]:
    """Turn a list of per-residue DSSP dicts into (ss8_string, sasa_list).

    Returns (None, None) for both fields if the DSSP table doesn't align with
    the expected sequence length — better to drop a noisy modality than
    silently misalign it.
    """
    if len(rows) != expected_len:
        return None, None
    ss8 = "".join(r["ss8"] for r in rows)
    sasa = [float(r["asa"]) for r in rows]
    return ss8, sasa


def iter_pilot_proteins(
    pilot_dir: Path,
    *,
    require_structure: bool = True,
    min_length: int = 30,
    max_length: int = 800,
    accessions: Optional[set[str]] = None,
) -> Iterator[PilotProtein]:
    """Yield PilotProtein records that pass length/structure filters.

    Args:
        pilot_dir: path to the pilot dataset root (with sequences/, structures/,
            annotations/).
        require_structure: only yield proteins with PDB on disk and DSSP rows
            that align with the sequence (so structure + ss8 + sasa conditions
            are all runnable).
        min_length, max_length: residue bounds.
        accessions: optional restriction to a specific set of accessions.
    """
    metadata = load_pilot_index(pilot_dir)
    dssp_all = json.loads((pilot_dir / "annotations" / "dssp_annotations.json").read_text())
    struct_dir = pilot_dir / "structures"

    for acc, entry in metadata.items():
        if accessions is not None and acc not in accessions:
            continue
        seq = entry["sequence"]
        L = len(seq)
        if not (min_length <= L <= max_length):
            continue

        pdb_path = struct_dir / f"{acc}.pdb"
        ss8 = sasa = None
        if acc in dssp_all:
            ss8, sasa = _collapse_dssp(dssp_all[acc], L)

        if require_structure and (not pdb_path.exists() or ss8 is None):
            continue

        go_ids = [g["id"] for g in entry.get("go_terms", [])]

        yield PilotProtein(
            accession=acc,
            sequence=seq,
            length=L,
            category=entry.get("category", "unknown"),
            pdb_path=pdb_path,
            ss8=ss8,
            sasa=sasa,
            interpro_ids=entry.get("interpro", []) or [],
            pfam_ids=entry.get("pfam", []) or [],
            go_ids=go_ids,
        )


def iter_scaled_proteins(
    scaled_dir: Path,
    metadata_path: Path,
    *,
    min_length: int = 30,
    max_length: int = 800,
    accessions: Optional[set[str]] = None,
) -> Iterator[PilotProtein]:
    """Yield PilotProtein records for the scaled dataset (data/scaled/).

    Unlike the pilot, SS8/SASA come from ``annotations/structure_annotations.json``
    ({acc: {ss8, sasa, length}}, produced by annotate_structures.py from the
    fetched AFDB structures), and every yielded protein additionally carries
    InterPro IDs for the function modality.
    """
    metadata = json.loads(Path(metadata_path).read_text())
    struct_anno = json.loads(
        (scaled_dir / "annotations" / "structure_annotations.json").read_text()
    )
    struct_dir = scaled_dir / "structures"

    for acc, entry in metadata.items():
        if accessions is not None and acc not in accessions:
            continue
        seq = entry["sequence"]
        L = len(seq)
        if not (min_length <= L <= max_length):
            continue

        pdb_path = struct_dir / f"{acc}.pdb"
        anno = struct_anno.get(acc)
        if not pdb_path.exists() or anno is None:
            continue
        ss8, sasa = anno["ss8"], anno["sasa"]
        # SS8/SASA were computed from the structure; only keep proteins whose
        # structure length matches the canonical sequence so every residue-level
        # modality stays aligned.
        if not (len(ss8) == len(sasa) == L):
            continue

        yield PilotProtein(
            accession=acc,
            sequence=seq,
            length=L,
            category=entry.get("category", "unknown"),
            pdb_path=pdb_path,
            ss8=ss8,
            sasa=sasa,
            interpro_ids=entry.get("interpro", []) or [],
            pfam_ids=entry.get("pfam", []) or [],
            go_ids=[g["id"] for g in entry.get("go_terms", [])],
        )
