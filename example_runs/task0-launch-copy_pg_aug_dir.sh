#!/bin/bash

while IFS= read -r line; do
    line_arr=( $line )
    lang=${line_arr[0]}
    echo "Copy pg model on aug: $lang"
    mkdir -p checkpoints/sigmorphon20-task0/transformer-pg/$lang/aug
    cp -r checkpoints/sigmorphon20-task0/transformer/$lang/aug/4 checkpoints/sigmorphon20-task0/transformer-pg/$lang/aug/1
done < SIG_2020-lang-patience.txt
