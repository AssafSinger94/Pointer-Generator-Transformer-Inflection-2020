#!/bin/bash

while IFS= read -r line; do
    line_arr=( $line )
    lang=${line_arr[0]}
    echo "Make log dir for: $lang"
    mkdir -p task0-logs/$lang
done < SIG_2020-lang-patience.txt