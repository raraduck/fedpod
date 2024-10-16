#!/bin/bash
# echo "::::::::::initial aggregation requires initial_model" && \
# echo "::::::::::initial_model is just copied to under states/job-step folder" && \
bash /fedpod/run_train.sh \
     -s 0 \
     -g 0 \
     -J trn_2 \
     -R 2 \
     -r 0 \
     -E 1 \
     -i 2 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/cc359ppmi128/R00r00.pth && \

mkdir -p /fedpod/states/trn_0/R02r01/models && \

cp /fedpod/states/trn_2/R02r00/models/R02r00_last.pth \
    /fedpod/states/trn_0/R02r01/models/R02r01.pth && \

bash /fedpod/run_train.sh \
     -s 0 \
     -g 0 \
     -J trn_2 \
     -R 2 \
     -r 1 \
     -E 1 \
     -i 2 \
     -c /fedpod/cc359ppmi128/CC359PPMI_v1-test.csv \
     -m /fedpod/states/trn_0/R02r01/models/R02r01.pth && \

rm -rf /fedpod/states/trn_2/R02r00 \
    /fedpod/states/trn_0/R02r01 \
    /fedpod/states/trn_2/R02r01 && \

rm -rf /fedpod/logs/trn_2_R02r00.log \
     /fedpod/logs/trn_2_R02r01.log && \

rmdir /fedpod/states/trn_2 && \

rmdir /fedpod/states/trn_0

# echo "::::::::::second aggregation requires to not specify model_path" && \

# echo "::::::::::which is to collect them(local models) by round and job_prefix pattern" && \

# mkdir -p \
#     /fedpod/states/R02r00/trn1/models \
#     /fedpod/states/R02r00/trn_2/models && \

# cp /fedpod/states/R02r00/trn_0/R02r00.pth /fedpod/states/R02r00/trn1/models/R02r00_last.pth && \

# cp /fedpod/states/R02r00/trn_0/R02r00.pth /fedpod/states/R02r00/trn_2/models/R02r00_last.pth && \

# bash run_aggregation.sh \
#     -R 2 \
#     -r 1 \
#     -a fedavg \
#     -j trn \
#     -i 0 \
#     -m None && \
    
# rm -rf /fedpod/states/R02r00/trn_0 \
#     /fedpod/states/R02r00/trn1 \
#     /fedpod/states/R02r00/trn_2 \
#     /fedpod/states/R02r01/trn_0 && \

# rmdir /fedpod/states/R02r00 && \

# rmdir /fedpod/states/R02r01