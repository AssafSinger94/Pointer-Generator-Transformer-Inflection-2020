#!/bin/bash

for model_copy in 1 2 3 4 5; do
for lang in Basque Bulgarian English Finnish German Kannada Maltese Navajo Persian Portuguese Russian Spanish Swedish Turkish; do
    seed=$(( 973*model_copy ))
    echo "Training pointer-generator transformer: $lang $seed $model_copy"
    sbatch run-pg-task2.s $lang $seed $model_copy
done
done