#!/bin/bash

# Example - run transformer on multitask training set, for all languages

model_copy=1
seed=0
for lang in aka ang ast aze azg bak ben bod cat ceb cly cpa cre crh ctp czn dak dan deu dje eng est evn fas fin frm frr fur gaa glg gmh gml gsw hil hin isl izh kan kaz kir kjh kon kpv krl lin liv lld lud lug mao mdf mhr mlg mlt mwf myv nld nno nob nya olo ood orm ote otm pei pus san sme sna sot swa swe syc tel tgk tgl tuk udm uig urd uzb vec vep vot vro xno xty zpv zul; do
    bash task0-trm-trn.sh $lang $seed $model_copy