#!/bin/bash
echo "::::::::::initial aggregation requires initial_model" && \
echo "::::::::::initial_model is just copied to under states/job-step folder" && \
bash /fedpod/run_aggregation.sh \
    -R 2 \
    -r 0 \
    -a fedavg \
    -j agg \
    -i 0 \
    -m /fedpod/cc359ppmi128/R00r00.pth && \

echo "::::::::::second aggregation requires to not specify model_path" && \

echo "::::::::::which is to collect them(local models) by round and job_prefix pattern" && \

mkdir -p \
    /fedpod/states/agg_1/R02r00/models \
    /fedpod/states/agg_2/R02r00/models && \
# /fedpod/states/agg_1/R02r00/models \
# /fedpod/states/agg_2/R02r00/models && \

# cp  /fedpod/states/agg_0/R02r00/models/R02r00.pth \
cp  /fedpod/cc359ppmi128/Agg_5_R12r09.pth \
    /fedpod/states/agg_1/R02r00/models/R02r00_last.pth && \

# cp  /fedpod/states/agg_0/R02r00/models/R02r00.pth \
cp  /fedpod/cc359ppmi128/Agg_6_R12r09.pth \
    /fedpod/states/agg_2/R02r00/models/R02r00_last.pth && \

bash /fedpod/run_aggregation.sh \
    -R 2 \
    -r 1 \
    -a fedavg \
    -j agg \
    -i 0 \
    -m None && \

awk -F "\"*,\"*" '{print $1 $2 $3, ... $(NF-2), $(NF-1), $NF}}' /fedpod/logs/agg_0/agg_metrics.csv
# cat /fedpod/logs/agg_0/agg_metrics.csv
# cat /fedpod/logs/agg_0/agg_metrics.csv | column -t -s ","

rm -rf /fedpod/logs/agg_0 \
    /fedpod/logs/agg_R02r00.log \
    /fedpod/logs/agg_R02r01.log \
    /fedpod/states/agg_0/R02r00 \
    /fedpod/states/agg_1/R02r00 \
    /fedpod/states/agg_2/R02r00 \
    /fedpod/states/agg_0/R02r01 && \

rmdir /fedpod/states/agg_0 && \
rmdir /fedpod/states/agg_1 && \
rmdir /fedpod/states/agg_2