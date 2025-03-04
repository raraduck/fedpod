#!/bin/bash
set -e  # 명령어 실패 시 스크립트 종료

export ROUNDS=20 ROUND=20 

if [ "$1" = "FETS1470" ]; then
    export DATAROOT=data240_fets1470
    export DATASET=FETS1470
    export INPUT_CHANNEL_NAMES="[t1,t1ce,t2,flair]"
    export LABEL_GROUPS="[[1,2,4],[1,4],[4,4]]"
    export LABEL_NAMES="[WT,TC,ET]"
    export LABEL_INDEX="[2,1,4]"
    export JOBNAME1=cen01fets INSTID1=0
    export SPLIT_CSV="experiments/FETS1470_v3.csv"
    export SEG_POSTFIX="_sub"
    export MODEL="/fedpod/states/cen01fets_0/R20r20/models/R20r20_agg.pth" 
    docker-compose -f compose-CMC-forward.yaml up run_forward_fets && \
    docker-compose -f compose-CMC-forward.yaml down
elif [ "$1" = "BRATS2023" ]; then
    export DATAROOT=data240_brats2023
    export DATASET=FETS1470
    export INPUT_CHANNEL_NAMES="[t1,t1ce,t2,flair]"
    export LABEL_GROUPS="[[1,2,4],[1,4],[4,4]]"
    export LABEL_NAMES="[WT,TC,ET]"
    export LABEL_INDEX="[2,1,3]"
    export JOBNAME1=cen01brats INSTID1=0
    export SPLIT_CSV="experiments/BRATS2023_v3.csv"
    export SEG_POSTFIX="_sub"
    export MODEL="/fedpod/states/cen01fets_0/R20r20/models/R20r20_agg.pth" 
    docker-compose -f compose-CMC-forward.yaml up run_forward_fets && \
    docker-compose -f compose-CMC-forward.yaml down
elif [ "$1" = "CC359PPMI1" ]; then
    export DATAROOT=data256_cc359ppmicmc_newseg
    export DATASET=CC359PPMI
    export INPUT_CHANNEL_NAMES="[t1]"  # Assuming different channels for CC359
    export LABEL_GROUPS="[[1,2,3,4,5,6],[7,8,9,10,11,12]]"         # Assuming different label groups for CC359
    export LABEL_NAMES="[LS,RS]"                # Assuming different labels for CC359
    export LABEL_INDEX="[1,2]"                  # Assuming different label indices for CC359
    export JOBNAME1=cen01cc359 INSTID1=0          # Update as needed for CC359
    export SPLIT_CSV="experiments/CC359PPMICMC_v3.csv"
    export SEG_POSTFIX="_seg"
    export MODEL="/fedpod/states/cen01cc359-192-192-E20_best/R01r00/models/R01r00_best.pth" 
    docker-compose -f compose-CMC-forward.yaml up run_forward_cc359 && \
    docker-compose -f compose-CMC-forward.yaml down
elif [ "$1" = "CC359PPMI2" ]; then
    export DATAROOT=data256_cc359ppmicmc_newseg
    export DATASET=CC359PPMI
    export INPUT_CHANNEL_NAMES="[t1,seg]"  # Assuming different channels for CC359
    export LABEL_GROUPS="[[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12]]"         # Assuming different label groups for CC359
    export LABEL_NAMES="[LVS,LAC,LPC,LAP,LPP,LVP,RVS,RAC,RPC,RAP,RPP,RVP]"                # Assuming different labels for CC359
    export LABEL_INDEX="[1,2,3,4,5,6,7,8,9,10,11,12]"                  # Assuming different label indices for CC359
    export JOBNAME1=cen02cc359 INSTID1=0          # Update as needed for CC359
    export SPLIT_CSV="experiments/CC359PPMICMC_v3.csv"
    export SEG_POSTFIX="_sub"
    export MODEL="/fedpod/states/cen01cc359-192-192-E20_best/R01r00/models/R01r00_best.pth" 
    docker-compose -f compose-CMC-forward.yaml up run_forward_cc359 && \
    docker-compose -f compose-CMC-forward.yaml down
else
    echo "Invalid dataset specified"
    exit 1
fi
