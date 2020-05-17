#!/bin/bash

for model_copy in 2 3 4 5; do
    seed=$(( 973*model_copy ))
    while IFS= read -r line; do
        line_arr=( $line )
        lang=${line_arr[0]}
        echo "Running transformer trn:Small $lang $seed $model_copy"
        sbatch run_baseline-transformer-trn-small.s $lang $seed $model_copy
done < SIG_2020-lang-patience-hall_pretrain.txt
done
