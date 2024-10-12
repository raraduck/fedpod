#!/bin/bash
# echo "::::::::::initial aggregation requires initial_model" && \
# echo "::::::::::initial_model is just copied to under states/job-step folder" && \
bash /fedpod/run_train.sh \
     -s 0 \
     -g 0 \
     -j test1 \
     -R 2 \
     -r 0 \
     -E 1 \
     -i 1 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/cc359ppmi128/R00r00.pth && \

bash /fedpod/run_train.sh \
     -s 0 \
     -g 0 \
     -j test2 \
     -R 2 \
     -r 0 \
     -E 1 \
     -i 2 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/cc359ppmi128/R00r00.pth && \

mkdir -p /fedpod/states/R02r01/test0/models && \

# if [ -f /fedpod/states/R02r00/test1/models/R02r00_last.pth ]; then
#     cp /fedpod/states/R02r00/test1/models/R02r00_last.pth \
#     /fedpod/states/R02r01/test0/R02r01.pth
# else
#     cp /fedpod/states/R02r00/test1/models/R02r00.pth \
#     /fedpod/states/R02r01/test0/R02r01.pth
# fi && \
cp /fedpod/states/R02r00/test2/models/R02r00_last.pth \
    /fedpod/states/R02r01/test0/models/R02r01.pth && \

bash /fedpod/run_train.sh \
     -s 0 \
     -g 0 \
     -j test2 \
     -R 2 \
     -r 1 \
     -E 1 \
     -i 2 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/states/R02r01/test0/models/R02r01.pth && \

rm -rf /fedpod/states/R02r00/test1 \
    /fedpod/states/R02r00/test2 \
    /fedpod/states/R02r01/test0 \
    /fedpod/states/R02r01/test2 && \

rm -rf /fedpod/logs/R02r00_test1.log \
     /fedpod/logs/R02r00_test2.log \
     /fedpod/logs/R02r01_test2.log \

rmdir /fedpod/states/R02r00 && \

rmdir /fedpod/states/R02r01

# echo "::::::::::second aggregation requires to not specify model_path" && \

# echo "::::::::::which is to collect them(local models) by round and job_prefix pattern" && \

# mkdir -p \
#     /fedpod/states/R02r00/test1/models \
#     /fedpod/states/R02r00/test2/models && \

# cp /fedpod/states/R02r00/test0/R02r00.pth /fedpod/states/R02r00/test1/models/R02r00_last.pth && \

# cp /fedpod/states/R02r00/test0/R02r00.pth /fedpod/states/R02r00/test2/models/R02r00_last.pth && \

# bash run_aggregation.sh \
#     -R 2 \
#     -r 1 \
#     -a fedavg \
#     -j test \
#     -i 0 \
#     -m None && \
    
# rm -rf /fedpod/states/R02r00/test0 \
#     /fedpod/states/R02r00/test1 \
#     /fedpod/states/R02r00/test2 \
#     /fedpod/states/R02r01/test0 && \

# rmdir /fedpod/states/R02r00 && \

# rmdir /fedpod/states/R02r01