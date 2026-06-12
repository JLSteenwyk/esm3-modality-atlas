#!/usr/bin/env bash
# Compile the manuscript and the supplementary PDF. Requires a LaTeX engine
# (tectonic or pdflatex). The supplementary .tex is regenerated from the captions
# in figure_captions.md first, so it stays in sync.
set -e
cd "$(dirname "$0")"

python3 build_supplementary.py

compile() {  # $1 = .tex stem
  if command -v tectonic >/dev/null; then
    tectonic "$1.tex"
  elif command -v pdflatex >/dev/null; then
    pdflatex -interaction=nonstopmode "$1.tex" && pdflatex -interaction=nonstopmode "$1.tex"
  else
    echo "No LaTeX engine found. Install tectonic (conda install -c conda-forge tectonic)"
    echo "or upload $1.tex + figures/ to Overleaf."; exit 1
  fi
}

compile manuscript
compile supplementary
