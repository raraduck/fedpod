#!/bin/bash

# round 값을 환경 변수로 전달하여 docker-compose 실행
ROUNDS=3 ROUND=0 JOBNAME=testfed_2 INSTID=2 \
docker-compose -f compose-train1.yaml up
docker-compose -f compose-train1.yaml down

# ROUNDS=3 ROUND=0 JOBNAME=testfed_3 INSTID=3 \
# docker-compose -f compose-train1.yaml up
# docker-compose -f compose-train1.yaml down

ROUNDS=3 ROUND=0 JOBPREFIX=testfed INSTID=0 \
docker-compose -f compose-aggregation.yaml up
docker-compose -f compose-aggregation.yaml down


# 시작 round와 종료 round 설정
# START_ROUND=1
# END_ROUND=3  # 원하는 round 종료값

# for ((round=$START_ROUND; round<$END_ROUND; round++))
# do
#     echo "Starting round $round..."

#     # round 값을 환경 변수로 전달하여 docker-compose 실행
#     ROUND=$round docker-compose -f compose-train1.yaml up

#     # 각 round가 완료되면 컨테이너 정리
#     docker-compose -f compose-train1.yaml down
# done