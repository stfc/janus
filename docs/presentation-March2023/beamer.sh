#!/usr/bin/env bash 

set -x 

in=slides
for i in tex pdf; do
pandoc -s  -f markdown --toc -t beamer ${in}.md  \
  --pdf-engine=lualatex \
  --template=templates/daresbury.beamer \
  --highlight-style=kate  \
  -o ${in}.$i
done
pandoc -s  --mathjax -f markdown  -t revealjs ${in}.md  \
  --highlight-style=kate \
  -V theme=solarized \
  -o slides.html
