#!/bin/bash
save_infer=""
use_gpu=""
job_name=""
rounds=""
round=""
epochs=""
inst_id=""
split_csv="cc359ppmi128/CC359PPMI_v1.csv"
model_pth=""

# 명령줄 옵션 처리
while getopts s:g:J:R:r:E:i:c:m: option
do
    case "${option}"
    in
        s) save_infer=${OPTARG};;
        g) use_gpu=${OPTARG};;
        J) job_name=${OPTARG};;
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
        E) epochs=${OPTARG};;
        i) inst_id=${OPTARG};;
		c) split_csv=${OPTARG};;
		m) model_pth=${OPTARG};;
    esac
done
echo "save_infer: $save_infer, gpu: $use_gpu, job: $job_name, rounds: $rounds, round: $round, epochs: $epochs, inst: $inst_id, split_csv: $split_csv, model_pth: $model_pth"

# 필수 옵션 검사
if [ -z "$save_infer" ] || [ -z "$use_gpu" ] || [ -z "$job_name" ] || [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$epochs" ] || [ -z "$inst_id" ] || [ -z "$split_csv" ] || [ -z "$model_pth" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -s <save_infer> -g <use_gpu> -J <job_name> -R <rounds> -r <round> -e <epochs> -i <inst_id> -c <split_csv> -m <model_pth>"
    exit 1
fi


python3 scripts/run_train.py \
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
	--inst_root inst_* \
	\
	--cases_split $split_csv \
	--inst_ids [$inst_id] \
	--batch_size 1 \
	\
	--weight_path $model_pth \
	--input_channel_names [t1] \
	--label_groups [[1,1],[2,3],[4,5,6],[7,7],[8,9],[10,11,12]] \
	--label_names [LA,LC,LP,RA,RC,RP] \
	--label_index [26,11,12,58,50,51] \
	--unet_arch unet \
	--channels_list [32,64,128,256] \
	--block res \
	--optim adamw \
	--ds_layer 1 \
	--kernel_size 3 \
	--norm instance \
	--scheduler step \
	--milestones [16] \
	--lr_gamma 0.1
