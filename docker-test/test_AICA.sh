#!/bin/bash
set -e  # 명령어 실패 시 스크립트 종료

export ROUNDS=3 ROUND=0 
export MODEL=None 
export JOBNAME1=testfed_2 INSTID1=2 
export JOBNAME2=testfed_3 INSTID2=3 
docker compose -f compose-AICA.yaml up fedpod-train1 && \
docker compose -f compose-AICA.yaml down

export ROUNDS=3 ROUND=0 
export JOBPREFIX=testfed INSTID=0 
docker compose -f compose-AICA.yaml up fedpod-aggregation && \
docker compose -f compose-AICA.yaml down

export ROUNDS=3 ROUND=1 
export MODEL=/fedpod/states/testfed_0/R03r01/models/R03r01_agg.pth 
export JOBNAME1=testfed_2 INSTID1=2 
export JOBNAME2=testfed_3 INSTID2=3 
docker compose -f compose-AICA.yaml up fedpod-train1 fedpod-train2 && \
docker compose -f compose-AICA.yaml down

export ROUNDS=3 ROUND=1 
export JOBPREFIX=testfed INSTID=0 
docker compose -f compose-AICA.yaml up fedpod-aggregation && \
docker compose -f compose-AICA.yaml down

export ROUNDS=1 ROUND=0 
export MODEL=/fedpod/states/testfed_0/R03r02/models/R03r02_agg.pth 
export JOBNAME=testfed_test INSTID=0 
docker compose -f compose-AICA.yaml up fedpod-infer && \
docker compose -f compose-AICA.yaml down
