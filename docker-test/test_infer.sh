#!/bin/bash
bash /fedpod/run_infer.sh \
     -s 1 \
     -g 0 \
     -J infer_1 \
     -j infer \
     -R 2 \
     -r 0 \
     -E 1 \
     -i 1 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/cc359ppmi128/R00r00.pth

rm -rf /fedpod/states/infer_1/R02r00 && \

rm -rf /fedpod/logs/infer_1_R02r00.log && \

rmdir /fedpod/states/infer_1
