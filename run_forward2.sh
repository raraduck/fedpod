#!/bin/bash
save_infer=""
use_gpu=""
job_name=""
rounds=""
round=""
curr_epoch=""
inst_id=""
split_csv="cc359ppmi128/CC359PPMI_v1-test.csv"
model_pth=""
test_mode=""
data_root="cc359ppmi128"
inst_root="inst_*"

# 명령줄 옵션 처리
while getopts s:g:J:R:r:e:i:c:M:t:d:n: option
do
    case "${option}"
    in
        s) save_infer=${OPTARG};;
        g) use_gpu=${OPTARG};;
        J) job_name=${OPTARG};;
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
        e) curr_epoch=${OPTARG};;
        i) inst_id=${OPTARG};;
		c) split_csv=${OPTARG};;
		M) model_pth=${OPTARG};;
		t) test_mode=${OPTARG};;
		d) data_root=${OPTARG};;
		n) inst_root=${OPTARG};;
    esac
done
echo "save_infer: $save_infer, gpu: $use_gpu, job: $job_name, rounds: $round, curr_epoch: $curr_epoch, inst: $inst_id, split_csv: $split_csv, model_pth: $model_pth, test_mode: $test_mode, data_root: $data_root, inst_root: $inst_root"

# 필수 옵션 검사
if [ -z "$save_infer" ] || [ -z "$use_gpu" ] || [ -z "$job_name" ] ||  [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$curr_epoch" ] || [ -z "$inst_id" ] || [ -z "$split_csv" ] || [ -z "$inst_id" ] || [ -z "$test_mode" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -s <save_infer> -g <use_gpu> -J <job_name> -R <rounds> -r <round> -e <curr_epoch> -i <inst_id> -c <split_csv> -M <model_pth> -t <test_mode> -d <data_root> -n <inst_root>"
    exit 1
fi


python3 scripts/run_forward.py \
	--test_mode $test_mode \
	--save_infer $save_infer \
    --use_gpu $use_gpu \
    --job_name $job_name \
	--rounds $rounds \
	--round $round \
	--epoch $curr_epoch \
	\
	--resize 128 \
	--patch_size 128 \
	--dataset CC359PPMI \
	--data_root $data_root \
	--inst_root $inst_root \
	--img_name brain \
	--seg_name striatum_sub \
	\
	--cases_split $split_csv \
	--inst_ids [$inst_id] \
	--batch_size 1 \
	\
	--weight_path $model_pth \
	--input_channel_names [t1,seg] \
	--label_groups [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12]] \
	--label_names [LVS,LAC,LAP,LAP,LPP,LVP,RVS,RAC,RAP,RAP,RPP,RVP] \
	--label_index [1,2,3,4,5,6,7,8,9,10,11,12] \
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
