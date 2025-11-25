#!/bin/bash
save_infer=""
use_gpu=""
job_name=""
rounds=""
round=""
curr_epoch=""
inst_id=""
split_csv=""
model_pth=""
sel_list=""
data_root=""
data_set=""
inst_root="inst_*"
intput_channels=""
label_groups=""
label_names=""
label_index=""
seg_postfix=""

# 명령줄 옵션 처리
while getopts s:g:J:R:r:e:i:c:M:t:D:d:n:C:G:N:I:p: option
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
        t) sel_list=${OPTARG};;
        D) data_root=${OPTARG};;
        d) data_set=${OPTARG};;
        n) inst_root=${OPTARG};;
        C) input_channels=${OPTARG};;
        G) label_groups=${OPTARG};;
        N) label_names=${OPTARG};;
        I) label_index=${OPTARG};;
        p) seg_postfix=${OPTARG};;
    esac
done
echo "save_infer: $save_infer, gpu: $use_gpu, job: $job_name, rounds: $round, curr_epoch: $curr_epoch, inst: $inst_id, split_csv: $split_csv, model_pth: $model_pth, sel_list: $sel_list, data_root: $data_root, data_set: $data_set, inst_root: $inst_root, input_channels: $input_channels, label_groups: $label_groups, label_names: $label_names, label_index: $label_index, seg_postfix: $seg_postfix"

# 필수 옵션 검사
if [ -z "$save_infer" ] || [ -z "$use_gpu" ] || [ -z "$job_name" ] ||  [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$curr_epoch" ] || [ -z "$inst_id" ] || [ -z "$split_csv" ] || [ -z "$inst_id" ] || [ -z "$sel_list" ] || [ -z "$data_set" ] || [ -z "$input_channels" ] || [ -z "$label_groups" ] || [ -z "$label_names" ] || [ -z "$label_index" ] || [ -z "$seg_postfix" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -s <save_infer> -g <use_gpu> -J <job_name> -R <rounds> -r <round> -e <curr_epoch> -i <inst_id> -c <split_csv> -M <model_pth> -t <sel_list> -D <data_root> -d <data_set> -n <inst_root> -C <input_channels> -G <label_groups> -N <label_names> -I <label_index> -p <seg_postfix>"
    exit 1
fi

python3 scripts/run_forward.py \
    --sel_list $sel_list \
    --save_infer $save_infer \
    --use_gpu $use_gpu \
    --job_name $job_name \
    --rounds $rounds \
    --round $round \
    --epoch $curr_epoch \
    \
    --resize 128 \
    --patch_size 128 \
    --zoom 1 \
    --flip_lr 0 \
    --dataset $data_set \
    --data_root $data_root \
    --inst_root $inst_root \
    --seg_postfix $seg_postfix \
    \
    --cases_split $split_csv \
    --inst_ids [$inst_id] \
    --batch_size 4 \
    \
    --weight_path $model_pth \
    --input_channel_names $input_channels \
    --label_groups $label_groups \
    --label_names $label_names \
    --label_index $label_index \
    --unet_arch unet \
    --channels_list [32,64,128,256] \
    --block res \
    --optim adamw \
    --ds_layer 1 \
    --kernel_size 3 \
    --norm instance \
    --scheduler step \
    --milestones [18] \
    --lr_gamma 0.1
