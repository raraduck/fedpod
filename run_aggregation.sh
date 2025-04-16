#!/bin/bash
rounds=""
round=""
epochs=""
epoch=""
split_csv=""
algorithm=""
job_prefix=""
inst_id=""
model_pth="None"

# 명령줄 옵션 처리
while getopts R:r:E:e:c:a:j:i:M: option
do
    case "${option}"
    in
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
        E) epochs=${OPTARG};;
        e) epoch=${OPTARG};;
        a) algorithm=${OPTARG};;
        c) split_csv=${OPTARG};;
        j) job_prefix=${OPTARG};;
        i) inst_id=${OPTARG};;
        M) model_pth=${OPTARG};;
    esac
done

echo "rounds: $rounds, round: $round, epochs: $epochs, epoch: $epoch, split_csv: $split_csv, algorithm: $algorithm, job: $job_prefix, inst: $inst_id, model_pth: $model_pth"

# 필수 옵션 검사
if [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$epochs" ] || [ -z "$epoch" ] || [ -z "$split_csv" ] ||[ -z "$algorithm" ] || [ -z "$job_prefix" ] || [ -z "$inst_id" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -R <rounds> -r <round> -E <epochs> -e <epoch> -c <split_csv> -a <algorithm> -j <job_prefix> -i <inst_id> -M <model_pth>"
    exit 1
fi

#  echo "rounds: $rounds, round: $round, algorithm: $algorithm, job: $job_prefix, inst: $inst_id, model_pth: $model_pth"

python3 scripts/run_aggregation.py \
	--rounds $rounds \
	--round $round \
	--epochs $epochs \
	--epoch $epoch \
    --cases_split $split_csv \
	--algorithm $algorithm \
	--job_prefix $job_prefix \
	--inst_id $inst_id \
	--weight_path $model_pth
