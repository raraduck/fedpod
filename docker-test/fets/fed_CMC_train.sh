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
for Round in 0;
do
    for Inst in 1;
    do
        JobName=$(printf "%s_%d" $JobPrefix $Inst);
        Seed=$(($Seed + 1))  # SEED 환경변수를 계산하여 설정
        FromEpoch=0

        echo Round/Rounds:$Round/$Rounds Epochs:$Epochs FromEpoch: $FromEpoch Inst:$Inst Seed:$Seed JobPrefix:$JobPrefix JobName:$JobName

        export ROUND=$Round
        export EPOCHS=$Epochs
        export EPOCH=$FromEpoch
        export SEED=$Seed
        export JOBNAME=$JobName
        export INSTID=$Inst
        export MODEL=None
        docker-compose -f compose-CMC-train.yaml up run_train_fets && \
        docker-compose -f compose-CMC-train.yaml down
    done;
    # export INSTID=0
    # export ALGO=fedavg
    # export MODEL=None
    # docker-compose -f compose-CMC-train.yaml up run_agg_fets && \
    # docker-compose -f compose-CMC-train.yaml down
done;

for Round in $(seq 1 $Rounds);
do
    for Inst in {1..3};
    do
        JobName=$(printf "%s_%d" $JobPrefix $Inst);
        Seed=$(($Seed + 1))  # SEED 환경변수를 계산하여 설정
        FromEpoch=$(($Epochs*($Round-1) + 1))

        echo Round/Rounds:$Round/$Rounds Epochs:$Epochs FromEpoch: $FromEpoch Inst:$Inst Seed:$Seed JobPrefix:$JobPrefix JobName:$JobName

        
        export ROUND=$Round
        export EPOCHS=$Epochs
        export EPOCH=$FromEpoch
        export SEED=$Seed
        export JOBNAME=$JobName
        export INSTID=$Inst
        # export MODEL="/fedpod/states/${JobName}_1/R${Rounds}r${Round}/models/R${Rounds}r${Round}_agg.pth"
        export MODEL="/fedpod/states/${JobName}_1/R${Rounds}r00/models/R${Rounds}r00_last.pth"
        docker-compose -f compose-CMC-train.yaml up run_train_fets && \
        docker-compose -f compose-CMC-train.yaml down
    done;
    # export INSTID=0
    # export ALGO=fedavg
    # export MODEL=None
    # docker-compose -f compose-CMC-train.yaml up run_agg_fets && \
    # docker-compose -f compose-CMC-train.yaml down
done;