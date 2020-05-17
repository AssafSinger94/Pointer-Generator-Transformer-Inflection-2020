#!/bin/bash


while IFS= read -r line; do
    for model_copy in 1 2; do
    line_arr=( $line )
    lang=${line_arr[0]}
    echo "Training model: $lang $model_copy"
    bash task0-trm-aug-predict.sh $lang 0 $model_copy
done
done < SIG_2020-lang-patience-complete.txt
