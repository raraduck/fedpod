#!/bin/bash
save_infer=""
use_gpu=""
job_name=""
rounds=""
round=""
epochs=""
inst_id=""
model_pth="None"

# 명령줄 옵션 처리
while getopts s:g:j:R:r:E:i:m: option
do
    case "${option}"
    in
        s) save_infer=${OPTARG};;
        g) use_gpu=${OPTARG};;
        j) job_name=${OPTARG};;
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
        E) epochs=${OPTARG};;
        i) inst_id=${OPTARG};;
		m) model_pth=${OPTARG};;
    esac
done
echo "save_infer: $save_infer, gpu: $use_gpu, job: $job_name, round: $round, epochs: $epochs, inst: $inst_id, model_pth: $model_pth"

# 필수 옵션 검사
if [ -z "$save_infer" ] || [ -z "$use_gpu" ] || [ -z "$job_name" ] || [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$epochs" ] || [ -z "$inst_id" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -s <save_infer> -g <use_gpu> -j <job_name> -R <rounds> -r <round> -E <epochs> -i <inst_id> -m <model_pth>"
    exit 1
fi


python3 scripts/run_infer.py \
	--save_infer $save_infer \
    --use_gpu $use_gpu \
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
	\
	--weight_path $model_pth \
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
