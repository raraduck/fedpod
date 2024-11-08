#!/bin/bash

# CSV 파일 경로 설정
OUTPUT_FILE="/home2/dwnusa/node00/fedpod/logs/test-v1.csv"

# 현재 시간을 초 단위로 가져옴
SECONDS_NOW=$(date +%s)

# cordon 상태가 아닌 노드들의 이름과 CPU 사용량을 추출
NODES=$(/usr/local/bin/kubectl get nodes --no-headers | awk '!/SchedulingDisabled/ {print $1}')
NODE_USAGE=""
for NODE in $NODES; do
  CPU=$(/usr/local/bin/kubectl top node $NODE | awk 'NR>1 {print $3}' | sed 's/%//g')
  NODE_USAGE+="${CPU},"
done
NODE_USAGE=$(echo $NODE_USAGE | sed 's/,$//')  # 마지막 콤마 제거

# 파일이 존재하지 않으면 헤더를 추가
if [ ! -f "$OUTPUT_FILE" ]; then
  echo "sec, $(echo $NODES | tr ' ' ',')" > $OUTPUT_FILE
fi

# 시간과 CPU 사용량을 파일에 추가
echo "${SECONDS_NOW}, ${NODE_USAGE}" >> $OUTPUT_FILE
