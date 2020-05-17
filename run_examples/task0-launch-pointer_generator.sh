#!/bin/bash


while IFS= read -r line; do
    for model_copy in 1; do
    line_arr=( $line )
    lang=${line_arr[0]}
    patience=${line_arr[1]}
    echo "Training model: $lang $patience $model_copy"
    sbatch run-pointer_generator-invsqr-aug.s $lang $patience $model_copy
done
done < SIG_2020-lang-patience.txt
