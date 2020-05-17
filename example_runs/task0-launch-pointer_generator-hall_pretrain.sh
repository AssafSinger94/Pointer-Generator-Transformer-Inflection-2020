#!/bin/bash


while IFS= read -r line; do
    line_arr=( $line )
    lang=${line_arr[0]}
    echo "Running model with hall pretraining: $lang"
    sbatch run_baseline-pointer_generator-pretrain_hall.s $lang
done < SIG_2020-lang-patience-hall_pretrain.txt
