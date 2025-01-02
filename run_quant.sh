#!/bin/bash
src_base=""
trg_base=""

# 명령줄 옵션 처리
while getopts s:t: option
do
    case "${option}"
    in
        s) src_base=${OPTARG};;
        t) trg_base=${OPTARG};;
    esac
done
echo "src_base: $src_base, trg_base: $trg_base"

# 필수 옵션 검사
if [ -z "$src_base" ] || [ -z "$trg_base" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -s <src_base> -t <trg_base>"
    exit 1
fi

python3 scripts/utils/run_quant.py \
    --src_base $src_base \
    --trg_base $trg_base
