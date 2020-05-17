#!/bin/bash

for model_copy in 2 3 4 5; do
for lang in Basque English Kannada Maltese Navajo; do
    seed=$(( 973*model_copy ))
    echo "Training model: Small $lang $seed $model_copy"
    sbatch run_baseline-transformer-task2-small.s $lang $seed $model_copy
done
done