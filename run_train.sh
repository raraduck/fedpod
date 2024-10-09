#!/bin/bash
use_gpu=""
job_name=""
rounds=""
round=""
epochs=""
inst_id=""

# 명령줄 옵션 처리
while getopts g:j:R:r:E:i: option
do
    case "${option}"
    in
        g) use_gpu=${OPTARG};;
        j) job_name=${OPTARG};;
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
        E) epochs=${OPTARG};;
        i) inst_id=${OPTARG};;
    esac
done

# 필수 옵션 검사
if [ -z "$use_gpu" ] || [ -z "$job_name" ] || [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$epochs" ] || [ -z "$inst_id" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -j <job_name> -r <rounds> -R <round> -e <epochs> -i <inst_id> -g <use_gpu>"
    exit 1
fi

echo "job: $job_name, rounds: $rounds, round: $round, epochs: $epochs, inst: $inst_id"

python3 scripts/run_train.py \
    --job_name $job_name \
	--rounds $rounds \
	--round $round \
	--epochs $epochs \
	\
	--resize 128 \
	--patch_size 128 \
	--dataset CC359PPMI \
	--data_root cc359ppmi128 \
	--inst_root inst_0* \
	\
	--cases_split cc359ppmi128/CC359PPMI_v1.csv \
	--inst_ids [$inst_id] \
	--batch_size 1 \
    --use_gpu $gpu \
	\
	--input_channel_names [t1] \
	--label_groups [[26,26],[58,58],[11,11],[50,50],[12,12],[51,51]] \
	--label_names [LA,RA,LC,RC,LP,RP] \
	--label_index [26,58,11,50,12,51] \
	--unet_arch unet \
	--channels_list [32,64,128,256] \
	--block res \
	--optim adamw \
	--ds_layer 1 \
	--kernel_size 3 \
	--norm instance
