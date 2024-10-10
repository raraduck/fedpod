#!/bin/bash
rounds=""
round=""
algorithm=""
job_name=""
inst_id=""
model_pth=""

# 명령줄 옵션 처리
while getopts R:r:a:j:i:m: option
do
    case "${option}"
    in
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
		a) algorithm=${OPTARG};;
        j) job_name=${OPTARG};;
        i) inst_id=${OPTARG};;
		m) model_pth=${OPTARG};;
    esac
done

# 필수 옵션 검사
if [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$algorithm" ] || [ -z "$job_name" ] || [ -z "$inst_id" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -R <rounds> -r <round> -a <algorithm> -j <job_name> -i <inst_id> -m <model_pth>"
    exit 1
fi

echo "rounds: $rounds, round: $round, algorithm: $algorithm, job: $job_name, inst: $inst_id, model_pth: $model_pth"

python3 scripts/run_aggregation.py \
	--rounds $rounds \
	--round $round \
	--algorithm $algorithm \
	--job_id $job_name \
	--inst_id $inst_id \
	--weight_path $model_pth
