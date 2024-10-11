#!/bin/bash
echo "initial aggregation requires initial_model"
echo "initial_model is just copied to under states/job-step folder"
source run_aggregation.sh \
    -R 2 \
    -r 0 \
    -a fedavg \
    -j test \
    -i 0 \
    -m /fedpod/cc359ppmi128/R00E000.pth \
&& \
rm -rf /fedpod/states/R02r00/test00
# echo "second aggregation requires to not specify model_path"
# echo "which is to collect them(local models) by round and job_prefix pattern"
# &&
# source run_aggregation.sh \
#     -R 2 \
#     -r 0 \
#     -a fedavg \
#     -j test \
#     -i 0
