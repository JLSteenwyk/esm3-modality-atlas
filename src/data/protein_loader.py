"""Read the curated multimodal protein datasets (scaled, diverse, experimental).

Each protein is yielded as a ``ProteinRecord`` carrying the inputs the harvest
needs (sequence, structure PDB path, SS8, SASA, and whole-protein annotation IDs).
SS8/SASA come from ``annotations/structure_annotations.json`` ({acc: {ss8, sasa,
length}}, produced by annotate_structures.py from the fetched structures), and the
per-protein metadata (category, GO, InterPro) comes from the curation catalog.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class ProteinRecord:
    """One protein worth of multimodal inputs the harvest needs."""

    accession: str
    sequence: str
    length: int
    category: str
    pdb_path: Path
    ss8: Optional[str]
    sasa: Optional[list[float]]
    # Whole-protein-level annotation IDs (for the function modality)
    interpro_ids: list[str]
    pfam_ids: list[str]
    go_ids: list[str]


def iter_scaled_proteins(
    scaled_dir: Path,
    metadata_path: Path,
    *,
    min_length: int = 30,
    max_length: int = 800,
    accessions: Optional[set[str]] = None,
) -> Iterator[ProteinRecord]:
    """Yield ProteinRecord entries for a fetched dataset (scaled/diverse/experimental).

    SS8/SASA come from ``annotations/structure_annotations.json`` ({acc: {ss8, sasa,
    length}}, produced by annotate_structures.py from the fetched structures), and
    every yielded protein additionally carries InterPro IDs for the function modality.
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

        yield ProteinRecord(
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
