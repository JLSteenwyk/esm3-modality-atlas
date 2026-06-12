"""Generate supplementary.tex from the supplementary captions in figure_captions.md.

Parses every "**Figure SN. ...**" block under the supplementary section and emits a
LaTeX document that embeds figures/figureSN.png with the caption typeset below it, one
per page. Keeps the PDF in sync with the captions (single source of truth) and matches
the manuscript preamble. Compile with tectonic (see build.sh).

Output: paper/supplementary.tex
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTIONS = HERE / "figure_captions.md"
OUT = HERE / "supplementary.tex"

PREAMBLE = r"""\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\parindent}{0pt}
\pagestyle{plain}

\title{Supplementary Information\\[4pt]
\large When does ESM3 fuse its modalities? A geometry-first atlas of the
multimodal residual stream}
\author{Jacob L. Steenwyk}
\date{}

\begin{document}
\maketitle
\thispagestyle{plain}
\vspace{1em}
This document contains supplementary Figures S1 to S10. Each figure supports, but does
not lead, a result in the main text.
\clearpage
"""

# LaTeX-special characters that appear in plain prose captions.
_ESC = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "$": r"\$"}


def esc(s: str) -> str:
    return "".join(_ESC.get(c, c) for c in s)


def main() -> None:
    text = CAPTIONS.read_text()
    text = text[text.index("# Supplementary figure captions"):]
    # each block starts with **Figure S<N>. and runs until the next one
    blocks = re.split(r"\n(?=\*\*Figure S\d+\.)", text)
    blocks = [b.strip() for b in blocks if b.strip().startswith("**Figure S")]

    body = []
    for b in blocks:
        b = " ".join(b.split())                       # collapse wrapped lines
        m = re.match(r"\*\*(Figure S(\d+)\.[^*]*?)\*\*\s*(.*)$", b)
        lead, num, rest = m.group(1), m.group(2), m.group(3)
        body.append(
            "\\begin{center}\n"
            f"\\includegraphics[width=\\linewidth,height=0.74\\textheight,"
            f"keepaspectratio]{{figureS{num}.png}}\n"
            "\\end{center}\n"
            "\\vspace{0.6em}\n"
            "{\\small\\textbf{" + esc(lead) + "} " + esc(rest) + "}\n"
            "\\clearpage")
    OUT.write_text(PREAMBLE + "\n" + "\n".join(body) + "\n\\end{document}\n")
    print(f"wrote {OUT.relative_to(HERE.parent)} ({len(blocks)} figures)")


if __name__ == "__main__":
    main()
