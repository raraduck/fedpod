#!/bin/bash
echo "train process has zero-rounds configuration for central mode"
echo "firstly, non-zero configuration is to be tested"
./run_train.sh \
    -s 0 \
    -g 0 \
    -j test1 \
    -R 2 \
    -r 0 \
    -E 1 \
    -i 1