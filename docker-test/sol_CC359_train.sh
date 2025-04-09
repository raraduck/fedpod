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
    local inst_selected=$4

    if [ "$agg" == "fedavg" ]; then
        if [ "$inst_selected" == "$inst" ]; then
            if [ "$round" -le 4 ]; then
                echo 60 # 30개
            elif [ "$round" -le 9 ]; then
                echo 30 # 15개
            elif [ "$round" -le 14 ]; then
                echo 20 # 10개
            else
                echo 10 # 5개
            fi
        else
            echo 0
        fi
    elif [ "$agg" == "fedpod" ]; then
        if [ "$inst_selected" == "$inst" ]; then
            if [ "$round" -le 4 ]; then
                echo 60 # 30개
            elif [ "$round" -le 9 ]; then
                echo 30 # 15개
            elif [ "$round" -le 14 ]; then
                echo 20 # 10개
            else
                echo 10 # 5개
            fi
        else
            echo 0
        fi
    else
        if [ "$inst" -eq 1 ]; then
            echo 4
        else
            echo 0
        fi
    fi
}

trap 'cleanup' SIGINT  # SIGINT 신호를 cleanup 함수로 처리


if [ "$2" = "STAGE1" ]; then
    export INPUT_CHANNEL_NAMES="[t1]"
    export LABEL_GROUPS="[[1,2,3,4,5,6],[7,8,9,10,11,12]]"
    export LABEL_NAMES="[LS,RS]"
    export LABEL_INDEX="[1,2]"
elif [ "$2" = "STAGE2" ]; then
    export INPUT_CHANNEL_NAMES="[t1,seg]"
    export LABEL_GROUPS="[[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12]]"
    export LABEL_NAMES="[LVS,LAC,LPC,LAP,LPP,LVP,RVS,RAC,RPC,RAP,RPP,RVP]"
    export LABEL_INDEX="[1,2,3,4,5,6,7,8,9,10,11,12]"
else
    echo "Invalid STAGE specified"
    exit 1
fi
JobPrefix=$1;
Seed=10000
Rounds=19
Epochs=3
Algo=$3

export DATAROOT=data256_cc359_fnirt_raw_seg # data256_cc359ppmicmc_newseg
export DATASET=CC359PPMI
export SPLIT_CSV="experiments/CC359PPMICMC_v5_$4.csv"
export JOBPREFIX=$JobPrefix
export ROUNDS=$Rounds
for Round in 0;
do
    for Inst in $4;
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
        docker-compose -f compose-CMC-train.yaml up run_train_cc359 && \
        docker-compose -f compose-CMC-train.yaml down
    done;
    export INSTID=0
    export ALGO=fedavg
    export MODEL=None
    docker-compose -f compose-CMC-train.yaml up run_agg_cc359 && \
    docker-compose -f compose-CMC-train.yaml down
done;

for Round in $(seq 1 $(($Rounds - 1)));
do
    for Inst in {1..6};
    do
        JobName=$(printf "%s_%d" $JobPrefix $Inst);
        Seed=$(($Seed + 1))  # SEED 환경변수를 계산하여 설정
        FromEpoch=$(($Epochs*($Round-1)))
        DataPercentage=$(get_data_percentage $Round $Inst $Algo $4)
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
        docker-compose -f compose-CMC-train.yaml up run_train_cc359 && \
        docker-compose -f compose-CMC-train.yaml down
    done;
    export INSTID=0
    export ALGO=$Algo
    export MODEL=None
    docker-compose -f compose-CMC-train.yaml up run_agg_cc359 && \
    docker-compose -f compose-CMC-train.yaml down
done;