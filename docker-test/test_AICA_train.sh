#!/bin/bash
set -e  # 명령어 실패 시 스크립트 종료

export ROUNDS=1 ROUND=0 
export MODEL=None 

# export JOBNAME01=centre01 INSTID01=6 
# docker compose -f compose-AICA-train1.yaml up centre0-train1 && \
# docker compose -f compose-AICA-train1.yaml down

# export JOBNAME02=centre02 INSTID02=6 
# docker compose -f compose-AICA-train1.yaml up centre0-train2 && \
# docker compose -f compose-AICA-train1.yaml down

export JOBNAME11=solo11 INSTID11=1
docker compose -f compose-AICA-train.yaml up solo1-train1 && \
docker compose -f compose-AICA-train.yaml down

export JOBNAME12=solo12 INSTID12=1
docker compose -f compose-AICA-train.yaml up solo1-train2 && \
docker compose -f compose-AICA-train.yaml down