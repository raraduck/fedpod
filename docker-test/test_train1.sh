#!/bin/bash

# 시작 round와 종료 round 설정
START_ROUND=0
END_ROUND=3  # 원하는 round 종료값

for ((round=$START_ROUND; round<$END_ROUND; round++))
do
    echo "Starting round $round..."

    # round 값을 환경 변수로 전달하여 docker-compose 실행
    ROUND=$round docker-compose -f compose-train1.yaml up

    # 각 round가 완료되면 컨테이너 정리
    docker-compose -f compose-train1.yaml down
done