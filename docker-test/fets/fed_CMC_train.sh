#!/bin/bash
export DATAROOT=data240_fets1470
export DATASET=FETS1470
export INPUT_CHANNEL_NAMES="[t1,t1ce,t2,flair]"
export LABEL_GROUPS="[[1,2,4],[1,4],[4,4]]"
export LABEL_NAMES="[WT,TC,ET]"
export LABEL_INDEX="[2,1,4]"
export EPOCHS=3

ROUNDS=2
export ROUNDS=$ROUNDS 
for ROUND in $(seq 1 $ROUNDS);
do
    # SUBJ=$(printf "VMAT%06d" $i);
    # echo $SUBJ;ROUND=$ROUND
    export MODEL=None 
    export JOBNAME1=fed01fets INSTID1=1
    export SPLIT_CSV="experiments/FETS1470_v0.csv"
    docker-compose -f compose-CMC-train.yaml up run_train_fets && \
    docker-compose -f compose-CMC-train.yaml down
done;