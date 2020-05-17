#!/bin/bash


while IFS= read -r line; do
    line_arr=( $line )
    lang=${line_arr[0]}
    patience=${line_arr[1]}
    echo "Running model with hall pretraining: $lang $patience"
    sbatch run-pointer_generator-invsqr-hall_pretrain.s $lang $patience
done < SIG_2020-lang-patience-hall_pretrain.txt
