#!/bin/bash

mkdir -p task0-data/hall

for lng in ceb dje gaa gmh gml hil izh kjh kon lin lud mao mlg mwf sot tel tgk vro xno zpv zul; do
python src/data/hallucinate.py task0-data/out $lng --examples 10000
mv -v task0-data/out/$lng.hall task0-data/hall/$lng.hall
done