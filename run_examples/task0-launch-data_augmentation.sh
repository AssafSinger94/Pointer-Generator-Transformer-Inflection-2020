#!/bin/bash

while IFS= read -r line; do
    line_arr=( $line )
    lang=${line_arr[0]}
    echo "Data augmentation for: $lang"
    python augment.py --data-dir task0-data/out --aug-dir task0-data/aug --lang $lang >> log-data-augmentation.out
done < SIG_2020-lang-patience.txt