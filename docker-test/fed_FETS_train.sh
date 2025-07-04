#!/bin/bash

# SIGINT 신호를 받았을 때 실행할 함수 정의
cleanup() {
    echo "SIGINT received, stopping all containers..."
    docker stop $(docker ps -q)  # 모든 실행 중인 Docker 컨테이너를 정지
    exit 1  # 스크립트 비정상 종료
}

get_data_percentage() {
    local round=$1
    local inst=$2
    local agg=$3

    if [ "$inst" -eq 1 ]; then
        echo 5
    elif [ "$inst" -eq 18 ]; then
        echo 10
    else
        echo 0
    fi
}

trap 'cleanup' SIGINT  # SIGINT 신호를 cleanup 함수로 처리

export INPUT_CHANNEL_NAMES="[t1,t1ce,t2,flair]"
export LABEL_GROUPS="[[1,2,4],[1,4],[4,4]]"
export LABEL_NAMES="[WT,TC,ET]"
export LABEL_INDEX="[2,1,4]"

JobPrefix=$1;
# JobPrefix=fedfets;
Seed=10000
Rounds=10
Epochs=3
Algo=$2

export DATAROOT=data240_fets1470
export DATASET=FETS1470
export SPLIT_CSV="experiments/FETS1470_v3.csv"
export JOBPREFIX=$JobPrefix
export ROUNDS=$Rounds
for Round in 0;
do
    for Inst in 1;
    do
        JobName=$(printf "%s_%d" $JobPrefix $Inst);
        Seed=$(($Seed + 1))  # SEED 환경변수를 계산하여 설정
        FromEpoch=0
        # DataPercentage=$(get_data_percentage $Round $Inst)
        echo Round/Rounds:$Round/$Rounds Epochs:$Epochs FromEpoch: $FromEpoch Inst:$Inst DataPercentage:$DataPercentage Seed:$Seed JobPrefix:$JobPrefix JobName:$JobName

        export ROUND=$Round
        export EPOCHS=0
        export EPOCH=$FromEpoch
        export SEED=$Seed
        export JOBNAME=$JobName
        export INSTID=$Inst
        export MODEL=None
        export DATA_PERCENTAGE=100
        docker-compose -f compose-CMC-train.yaml up run_train_fets && \
        docker-compose -f compose-CMC-train.yaml down
    done;
    export INSTID=0
    export ALGO=fedavg
    export MODEL=None
    docker-compose -f compose-CMC-train.yaml up run_agg_fets && \
    docker-compose -f compose-CMC-train.yaml down
done;

for Round in $(seq 1 $(($Rounds - 1)));
do
    for Inst in {1..23};
    do
        JobName=$(printf "%s_%d" $JobPrefix $Inst);
        Seed=$(($Seed + 1))  # SEED 환경변수를 계산하여 설정
        FromEpoch=$(($Epochs*($Round-1)))
        DataPercentage=$(get_data_percentage $Round $Inst)
        echo Round/Rounds:$Round/$Rounds Epochs:$Epochs FromEpoch: $FromEpoch Inst:$Inst DataPercentage:$DataPercentage Seed:$Seed JobPrefix:$JobPrefix JobName:$JobName
        
        export ROUND=$Round
        export EPOCHS=$Epochs
        export EPOCH=$FromEpoch
        export SEED=$Seed
        export JOBNAME=$JobName
        export INSTID=$Inst
        export MODEL="/fedpod/states/${JobPrefix}_0/$(printf 'R%02dr%02d' $Rounds $Round)/models/$(printf 'R%02dr%02d_agg.pth' $Rounds $Round)"
        # echo $MODEL
        export DATA_PERCENTAGE=$DataPercentage
        docker-compose -f compose-CMC-train.yaml up run_train_fets && \
        docker-compose -f compose-CMC-train.yaml down
    done;
    export INSTID=0
    export ALGO=$Algo
    export MODEL=None
    docker-compose -f compose-CMC-train.yaml up run_agg_fets && \
    docker-compose -f compose-CMC-train.yaml down
done;