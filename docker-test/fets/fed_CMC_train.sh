#!/bin/bash
export DATAROOT=data240_fets1470
export DATASET=FETS1470
export INPUT_CHANNEL_NAMES="[t1,t1ce,t2,flair]"
export LABEL_GROUPS="[[1,2,4],[1,4],[4,4]]"
export LABEL_NAMES="[WT,TC,ET]"
export LABEL_INDEX="[2,1,4]"
export SPLIT_CSV="experiments/FETS1470_v3.csv"

Seed=10000
Rounds=5
Epochs=3
JobPrefix=fedtest;
export JOBPREFIX=$JobPrefix
export ROUNDS=$Rounds
export MODEL=None
for Round in $(seq 0 $Rounds);
do
    for Inst in {1..3};
    do
        JobName=$(printf "%s_%d" $JobPrefix $Inst);
        Seed=$(($Seed + 1))  # SEED 환경변수를 계산하여 설정
        FromEpoch=$(($Epochs*$Round))

        echo Round/Rounds:$Round/$Rounds Epochs:$Epochs FromEpoch: $FromEpoch Inst:$Inst Seed:$Seed JobPrefix:$JobPrefix JobName:$JobName

        
        export ROUND=$Round
        export EPOCHS=$Epochs
        export EPOCH=$FromEpoch
        export SEED=$Seed
        export JOBNAME=$JobName
        export INSTID=$Inst
        docker-compose -f compose-CMC-train.yaml up run_train_fets && \
        docker-compose -f compose-CMC-train.yaml down
    done;
    # export INSTID=0
    # export ALGO=fedavg
    # export MODEL=None
    # docker-compose -f compose-CMC-train.yaml up run_agg_fets && \
    # docker-compose -f compose-CMC-train.yaml down
done;