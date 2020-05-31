#!/bin/bash

# Example - Pretrain pointer-generator transformer on hallucinated training set, for low-resource languages

model_copy=1
seed=0
for lang in ceb dje gaa gmh gml hil izh kjh kon lin lud mao mlg mwf sot tel tgk vro xno zpv zul; do
    bash task0-pg-pretrain_hall.sh $lang $seed $model_copy