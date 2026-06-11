#!/usr/bin/env bash
# Compile the manuscript. Requires a LaTeX engine (pdflatex or tectonic).
cd "$(dirname "$0")"
if command -v tectonic >/dev/null; then
  tectonic manuscript.tex
elif command -v pdflatex >/dev/null; then
  pdflatex -interaction=nonstopmode manuscript.tex && pdflatex -interaction=nonstopmode manuscript.tex
else
  echo "No LaTeX engine found. Install tectonic (conda install -c conda-forge tectonic)"
  echo "or upload manuscript.tex + figures/ to Overleaf."; exit 1
fi
