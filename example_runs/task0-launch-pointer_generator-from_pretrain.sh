#!/bin/bash


while IFS= read -r line; do
    for model_copy in 6; do
    line_arr=( $line )
    lang=${line_arr[0]}
    echo "Training model: $lang $model_copy"
    sbatch run_baseline-pointer_generator-aug.s $lang $model_copy
done
done < SIG_2020-lang-patience-hall_pretrain.txt
