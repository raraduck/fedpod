#!/bin/bash
bash /fedpod/run_train.sh \
     -s 0 \
     -g 0 \
     -J trn_2 \
     -R 2 \
     -r 0 \
     -E 10 \
     -e 0 \
     -i 2 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/cc359ppmi128/R00r00.pth && \

rm -rf /fedpod/states/trn_2/R02r00 && \

rm -rf /fedpod/logs/trn_2_R02r00.log && \

rmdir /fedpod/states/trn_2
