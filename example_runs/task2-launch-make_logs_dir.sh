#!/bin/bash

for lang in Basque Bulgarian English Finnish German Kannada Maltese Navajo Persian Portuguese Russian Spanish Swedish Turkish; do
    echo "Make log dir for: $lang"
    mkdir -p task2-logs/$lang
done