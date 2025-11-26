#!/bin/bash
seed=""
save_infer=""
eval_freq=""
milestone=""
use_gpu=""
flip_lr="1"
job_name=""
rounds=""
round=""
epochs=""
epoch=""
inst_id=""
split_csv=""
model_pth=""
data_percentage=""
data_root=""
inst_root="inst_*"
data_set=""
intput_channels=""
# label_groups_trn=""
label_groups=""
label_names=""
label_index=""

# 명령줄 옵션 처리
while getopts S:s:f:m:g:L:J:R:r:E:e:i:c:M:p:D:d:C:G:N:I: option
do
    case "${option}"
    in
        S) seed=${OPTARG};;
        s) save_infer=${OPTARG};;
        f) eval_freq=${OPTARG};;
        m) milestone=${OPTARG};;
        g) use_gpu=${OPTARG};;
        L) flip_lr=${OPTARG};;
        J) job_name=${OPTARG};;
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
        E) epochs=${OPTARG};;
        e) epoch=${OPTARG};;
        i) inst_id=${OPTARG};;
        c) split_csv=${OPTARG};;
        M) model_pth=${OPTARG};;
        p) data_percentage=${OPTARG};;
        D) data_root=${OPTARG};;
        d) data_set=${OPTARG};;
        C) input_channels=${OPTARG};;
        # T) label_groups_trn=${OPTARG};;
        G) label_groups=${OPTARG};;
        N) label_names=${OPTARG};;
        I) label_index=${OPTARG};;
    esac
done
echo "seed: $seed, save_infer: $save_infer, eval_freq: $eval_freq, milestone: $milestone, gpu: $use_gpu, flip_lr: $flip_lr, job: $job_name, rounds: $rounds, round: $round, epochs: $epochs, epoch: $epoch, inst: $inst_id, split_csv: $split_csv, model_pth: $model_pth, data_percentage: $data_percentage, data_root: $data_root, data_root: $data_set, input_channels: $input_channels, label_groups: $label_groups, label_names: $label_names, label_index: $label_index"

# 필수 옵션 검사
if [ -z "$seed" ] || [ -z "$save_infer" ] || [ -z "$eval_freq" ] || [ -z "$milestone" ] ||[ -z "$use_gpu" ] ||[ -z "$flip_lr" ] || [ -z "$job_name" ] || [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$epochs" ] || [ -z "$epoch" ] || [ -z "$inst_id" ] || [ -z "$split_csv" ] || [ -z "$model_pth" ] || [ -z "$data_percentage" ] || [ -z "$data_root" ] || [ -z "$data_set" ] || [ -z "$input_channels" ] || [ -z "$label_groups" ] || [ -z "$label_names" ] || [ -z "$label_index" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -S <seed> -s <save_infer> -f <eval_freq> -m <milestone> -g <use_gpu> -L <flip_lr> -J <job_name> -R <rounds> -r <round> -E <epochs> -e <epoch> -i <inst_id> -c <split_csv> -M <model_pth> -p <data_percentage> -D <data_root> -d <data_set> -C <input_channels> -G <label_groups> -N <label_names> -I <label_index>"
    exit 1
fi


python3 scripts/run_train.py \
  --seed $seed \
  --save_infer $save_infer \
  --eval_freq $eval_freq \
  --use_gpu $use_gpu \
  --job_name $job_name \
  --rounds $rounds \
  --round $round \
  --epochs $epochs \
  --epoch $epoch \
  \
  --resize 128 \
  --patch_size 128 \
  --zoom 1 \
  --flip_lr $flip_lr \
  --dataset $data_set \
  --data_root $data_root \
  --inst_root $inst_root \
  --data_percentage $data_percentage \
  \
  --cases_split $split_csv \
  --inst_ids [$inst_id] \
  --batch_size 1 \
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
  --lr 1e-3 \
  --ds_layer 1 \
  --kernel_size 3 \
  --norm instance \
  --scheduler step \
  --milestones $milestone \
  --lr_gamma 0.5
