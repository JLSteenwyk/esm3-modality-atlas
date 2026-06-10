"""Random-init control figure: trained vs untrained ESM3 fusion across depth.

Overlays the fusion metrics of the trained model against a random-init model of
the same architecture (structure encoder preserved, trunk + modality embeddings
reinitialised). A flat random-init curve with a fusing trained curve shows that
modality fusion is LEARNED, not an artifact of the architecture summing modality
embeddings.

Reads results/scaled/metrics.json and results/scaled_randinit/metrics.json.
Output: figures/scaled/metrics/randinit_control.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz import CONDITION_COLOR, INK  # noqa: E402

TRAINED = ROOT / "results" / "scaled" / "metrics.json"
RANDINIT = ROOT / "results" / "scaled_randinit" / "metrics.json"
OUT = ROOT / "figures" / "scaled" / "metrics" / "randinit_control.png"


def main() -> None:
    t = json.loads(TRAINED.read_text())
    r = json.loads(RANDINIT.read_text())
    nt = t["meta"]["n_proteins"]
    nr = r["meta"]["n_proteins"]
    x = np.array(t["series"]["layers"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.7), dpi=130, facecolor="white")
    rand_c = "#94a3b8"
    for ax, key, ylab in [
        (axL, "silhouette", "condition separation (silhouette)"),
        (axR, "integration_index", "integration index (mean pairwise CKA)"),
    ]:
        ax.plot(x, t["series"][key], "-o", color=INK, lw=2.4, markersize=4,
                label=f"trained ESM3 (n={nt})")
        ax.plot(x, r["series"][key], "--s", color=rand_c, lw=2.2, markersize=4,
                label=f"random-init control (n={nr})")
        ax.set_xlabel("Layer", color=INK)
        ax.set_ylabel(ylab, color=INK)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")

    axL.annotate("untrained: flat —\nno fusion", xy=(24, 0.62), xytext=(30, 0.50),
                 fontsize=9, color=rand_c, style="italic",
                 arrowprops=dict(arrowstyle="->", color=rand_c, lw=1.2))
    axL.annotate("trained: separation\npeaks then collapses", xy=(38, 0.20),
                 xytext=(8, 0.10), fontsize=9, color=INK, style="italic",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

    fig.suptitle("Modality fusion is learned, not architectural\n"
                 "trained ESM3 vs a random-init model of the same architecture",
                 fontsize=14, fontweight="bold", color=INK, y=1.04)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
