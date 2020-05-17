#!/bin/bash

for model_copy in 2 3 4 5; do
for lang in Basque Bulgarian English Finnish German Kannada Maltese Navajo Persian Portuguese Russian Spanish Swedish Turkish; do
    seed=$(( 973*model_copy ))
    echo "Training model: $lang $seed $model_copy"
    sbatch run_baseline-transformer-task2.s $lang $seed $model_copy
done
done