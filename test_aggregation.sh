#!/bin/bash
echo ">>>>> initial aggregation requires initial_model" && \
echo ">>>>> initial_model is just copied to under states/job-step folder" && \
bash run_aggregation.sh \
    -R 2 \
    -r 0 \
    -a fedavg \
    -j test \
    -i 0 \
    -m /fedpod/cc359ppmi128/R00E000.pth && \

echo ">>>>> second aggregation requires to not specify model_path" && \

echo ">>>>> which is to collect them(local models) by round and job_prefix pattern" && \

mkdir -p \
    /fedpod/states/R02r00/test1/models \
    /fedpod/states/R02r00/test2/models && \

cp /fedpod/states/R02r00/test0/R02r00.pth /fedpod/states/R02r00/test1/models/R02r00_last.pth && \

cp /fedpod/states/R02r00/test0/R02r00.pth /fedpod/states/R02r00/test2/models/R02r00_last.pth && \

bash run_aggregation.sh \
    -R 2 \
    -r 1 \
    -a fedavg \
    -j test \
    -i 0 \
    -m None && \
    
rm -rf /fedpod/states/R02r00/test0 \
    /fedpod/states/R02r00/test1 \
    /fedpod/states/R02r00/test2 \
    /fedpod/states/R02r01/test0 && \

rmdir /fedpod/states/R02r00 && \

rmdir /fedpod/states/R02r01