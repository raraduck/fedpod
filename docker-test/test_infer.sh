#!/bin/bash
bash /fedpod/run_infer1.sh \
     -s 1 \
     -g 0 \
     -J cen1 \
     -R 3 \
     -r 1 \
     -e 100 \
     -i 1 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/cc359ppmi128/R00r00.pth

rm -rf /fedpod/states/cen1/R02r00 && \

rm -rf /fedpod/logs/cen1_R02r00.log && \

rmdir /fedpod/states/cen1
