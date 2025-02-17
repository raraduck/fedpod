#!/bin/bash
set -e  # 명령어 실패 시 스크립트 종료

if [ "$1" = "FETS1470" ]; then
    export DATAROOT=data240_fets1470
    export DATASET=FETS1470
    export INPUT_CHANNEL_NAMES="[t1,t1ce,t2,flair]"
    export LABEL_GROUPS="[[1,2,4],[1,4],[4,4]]"
    export LABEL_NAMES="[WT,TC,ET]"
    export LABEL_INDEX="[2,1,4]"
    export JOBNAME1=cen01fets INSTID1=0
    export SPLIT_CSV="experiments/FETS1470_v0.csv"
    export SEG_POSTFIX="_sub"
elif [ "$1" = "CC359PPMI" ]; then
    export DATAROOT=data256_cc359ppmicmc_newseg
    export DATASET=CC359PPMI
    export INPUT_CHANNEL_NAMES="[t1]"  # Assuming different channels for CC359
    export LABEL_GROUPS="[[1,2,3,4,5,6],[7,8,9,10,11,12]]"         # Assuming different label groups for CC359
    export LABEL_NAMES="[LS,RS]"                # Assuming different labels for CC359
    export LABEL_INDEX="[1,2]"                  # Assuming different label indices for CC359
    export JOBNAME1=cen01cc359 INSTID1=0          # Update as needed for CC359
    export SPLIT_CSV="experiments/CC359PPMICMC_v0.csv"
    export SEG_POSTFIX="_sub"
else
    echo "Invalid dataset specified"
    exit 1
fi

export ROUNDS=1 ROUND=0 
export MODEL=None 

docker-compose -f compose-CMC-train.yaml up centre0-forward && \
docker-compose -f compose-CMC-train.yaml down

# export JOBNAME2=centre02 INSTID2=1
# docker-compose -f compose-CMC-train.yaml up centre0-train2 && \
# docker-compose -f compose-CMC-train.yaml down




# export JOBNAME1=solo11 INSTID1=1
# docker compose -f compose-AICA-train.yaml up solo1-train1 && \
# docker compose -f compose-AICA-train.yaml down

# export JOBNAME2=solo12 INSTID2=1
# docker compose -f compose-AICA-train.yaml up solo1-train2 && \
# docker compose -f compose-AICA-train.yaml down




# export JOBNAME1=solo21 INSTID1=2
# docker compose -f compose-AICA-train.yaml up solo2-train1 && \
# docker compose -f compose-AICA-train.yaml down

# export JOBNAME2=solo22 INSTID2=2
# docker compose -f compose-AICA-train.yaml up solo2-train2 && \
# docker compose -f compose-AICA-train.yaml down




# export JOBNAME1=solo31 INSTID1=3
# docker compose -f compose-AICA-train.yaml up solo3-train1 && \
# docker compose -f compose-AICA-train.yaml down

# export JOBNAME2=solo32 INSTID2=3
# docker compose -f compose-AICA-train.yaml up solo3-train2 && \
# docker compose -f compose-AICA-train.yaml down




# export JOBNAME1=solo41 INSTID1=4
# docker compose -f compose-AICA-train.yaml up solo4-train1 && \
# docker compose -f compose-AICA-train.yaml down

# export JOBNAME2=solo42 INSTID2=4
# docker compose -f compose-AICA-train.yaml up solo4-train2 && \
# docker compose -f compose-AICA-train.yaml down




# export JOBNAME1=solo51 INSTID1=5
# docker compose -f compose-AICA-train.yaml up solo5-train1 && \
# docker compose -f compose-AICA-train.yaml down

# export JOBNAME2=solo52 INSTID2=5
# docker compose -f compose-AICA-train.yaml up solo5-train2 && \
# docker compose -f compose-AICA-train.yaml down




# export JOBNAME1=solo61 INSTID1=6
# docker compose -f compose-AICA-train.yaml up solo6-train1 && \
# docker compose -f compose-AICA-train.yaml down

# export JOBNAME2=solo62 INSTID2=6
# docker compose -f compose-AICA-train.yaml up solo6-train2 && \
# docker compose -f compose-AICA-train.yaml down
