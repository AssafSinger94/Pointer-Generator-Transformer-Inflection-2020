#!/bin/bash

while IFS= read -r line; do
    line_arr=( $line )
    lang=${line_arr[0]}
    echo "Cover dev set: $lang"
    python cover.py --data-dir task0-data/out --lang $lang
done < SIG_2020-lang-patience.txt