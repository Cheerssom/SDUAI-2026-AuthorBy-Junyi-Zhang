#!/usr/bin/env bash
# Linux build: sync sources into _build_linux (no Windows aux/pdf), compile main_33 + supplementary.
set -e
cd "$(dirname "$0")"
MAIN=${1:-main_33}
mkdir -p _build_linux
rsync -a --exclude '_build_linux' --exclude 'cvpr2027_revisions' --exclude '*.aux' --exclude '*.log' --exclude '*.out' \
  --exclude '*.bbl' --exclude '*.blg' --exclude '*.synctex*' --exclude 'main_*.pdf' --exclude 'supplementary.pdf' ./ _build_linux/
cd _build_linux
run(){ pdflatex -interaction=nonstopmode "$1.tex" > "$1.stdout" 2>&1 || true; }
run $MAIN; bibtex $MAIN > /dev/null 2>&1 || true; run $MAIN; run $MAIN
cp $MAIN.aux main.aux
run supplementary; bibtex supplementary > /dev/null 2>&1 || true; run supplementary; run supplementary
grep -h "Output written" $MAIN.log supplementary.log
echo "undefined refs: $(grep -c 'undefined' $MAIN.log) main / $(grep -c 'undefined' supplementary.log) supp"
grep -h "^!" $MAIN.log supplementary.log | head -5 || true
