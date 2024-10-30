#!/bin/bash
set -e  # 명령어 실패 시 스크립트 종료

ROUNDS=3 ROUND=0 \
MODEL=None \
JOBNAME1=testfed_2 INSTID1=2 \
JOBNAME2=testfed_3 INSTID2=3 \
docker compose -f compose-v0.4.16.0.yaml up fedpod-test1 && \
docker compose -f compose-v0.4.16.0.yaml down

ROUNDS=3 ROUND=0 \
JOBPREFIX=testfed INSTID=0 \
docker compose -f compose-v0.4.16.0.yaml up fedpod-aggregation && \
docker compose -f compose-v0.4.16.0.yaml down

ROUNDS=3 ROUND=1 \
MODEL=/fedpod/states/testfed_0/R03r01/models/R03r01_agg.pth \
JOBNAME1=testfed_2 INSTID1=2 \
JOBNAME2=testfed_3 INSTID2=3 \
docker compose -f compose-v0.4.16.0.yaml up fedpod-test1  fedpod-test2 && \
docker compose -f compose-v0.4.16.0.yaml down

ROUNDS=3 ROUND=1 \
JOBPREFIX=testfed INSTID=0 \
docker compose -f compose-v0.4.16.0.yaml up fedpod-aggregation && \
docker compose -f compose-v0.4.16.0.yaml down

ROUNDS=3 ROUND=1 \
MODEL=/fedpod/states/testfed_0/R03r02/models/R03r02_agg.pth \
JOBNAME=testfed_test INSTID=0 \
docker compose -f compose-infer1.yaml up fedpod-infer && \
docker compose -f compose-infer1.yaml down
