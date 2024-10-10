#!/bin/bash
rounds=""
round=""
method=""
job_name=""
inst_id=""

# 명령줄 옵션 처리
while getopts R:r:m:j:i: option
do
    case "${option}"
    in
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
		m) method=${OPTARG};;
        j) job_name=${OPTARG};;
        i) inst_id=${OPTARG};;
    esac
done

# 필수 옵션 검사
if [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$method" ] || [ -z "$job_name" ] || [ -z "$inst_id" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -R <rounds> -r <round> -m <method> -j <job_name> -i <inst_id>"
    exit 1
fi

echo "rounds: $rounds, round: $round, job: $job_name, inst: $inst_id, method: $method"

python3 scripts/run_aggregation.py \
	--rounds $rounds \
	--round $round \
	--method $method \
	--job_id $job_name \
	--inst_id $inst_id
