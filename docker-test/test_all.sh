#!/bin/bash

ROUNDS=3 ROUND=0 \
MODEL=None \
JOBNAME1=testfed_2 INSTID1=2 \
docker compose -f compose-train1.yaml up fedpod-test1
docker compose -f compose-train1.yaml down

ROUNDS=3 ROUND=0 \
JOBPREFIX=testfed INSTID=0 \
docker compose -f compose-aggregation.yaml up
docker compose -f compose-aggregation.yaml down

ROUNDS=3 ROUND=1 \
MODEL=/fedpod/states/testfed_0/R03r01/models/R03r01_agg.pth \
JOBNAME1=testfed_2 INSTID1=2 \
JOBNAME2=testfed_3 INSTID2=3 \
docker compose -f compose-train1.yaml up
docker compose -f compose-train1.yaml down

ROUNDS=3 ROUND=1 \
JOBPREFIX=testfed INSTID=0 \
docker compose -f compose-aggregation.yaml up
docker compose -f compose-aggregation.yaml down
