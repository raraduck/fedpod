#!/bin/bash
bash /fedpod/run_train.sh \
     -s 0 \
     -g 0 \
     -J trn1 \
     -R 2 \
     -r 0 \
     -E 1 \
     -i 1 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/cc359ppmi128/R00r00.pth && \

rm -rf /fedpod/states/R02r00/trn1 && \

rm -rf /fedpod/logs/R02r00_trn1.log && \

rmdir /fedpod/states/R02r00