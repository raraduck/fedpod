#!/bin/bash
bash /fedpod/run_infer.sh \
     -s 0 \
     -g 0 \
     -j test3 \
     -R 2 \
     -r 0 \
     -E 1 \
     -i 3 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/cc359ppmi128/R00r00.pth && \
