#!/bin/bash
seed=""
save_infer=""
eval_freq=""
milestone=""
use_gpu=""
job_name=""
rounds=""
round=""
epochs=""
epoch=""
inst_id=""
split_csv="cc359ppmi128/CC359PPMI_v1.csv"
model_pth=""
data_percentage=""

# 명령줄 옵션 처리
while getopts S:s:f:m:g:J:R:r:E:e:i:c:M:p: option
do
    case "${option}"
    in
        S) seed=${OPTARG};;
        s) save_infer=${OPTARG};;
        f) eval_freq=${OPTARG};;
        m) milestone=${OPTARG};;
        g) use_gpu=${OPTARG};;
        J) job_name=${OPTARG};;
        R) rounds=${OPTARG};;
        r) round=${OPTARG};;
        E) epochs=${OPTARG};;
        e) epoch=${OPTARG};;
        i) inst_id=${OPTARG};;
        c) split_csv=${OPTARG};;
        M) model_pth=${OPTARG};;
        p) data_percentage=${OPTARG};;
    esac
done
echo "seed: $seed, save_infer: $save_infer, eval_freq: $eval_freq, milestone: $milestone, gpu: $use_gpu, job: $job_name, rounds: $rounds, round: $round, epochs: $epochs, epoch: $epoch, inst: $inst_id, split_csv: $split_csv, model_pth: $model_pth, data_percentage: $data_percentage"

# 필수 옵션 검사
if [ -z "$seed" ] || [ -z "$save_infer" ] || [ -z "$eval_freq" ] || [ -z "$milestone" ] || [ -z "$use_gpu" ] || [ -z "$job_name" ] || [ -z "$rounds" ] || [ -z "$round" ] || [ -z "$epochs" ] || [ -z "$epoch" ] || [ -z "$inst_id" ] || [ -z "$split_csv" ] || [ -z "$model_pth" ] || [ -z "$data_percentage" ]; then
    echo "Error: All parameters are required."
    echo "Usage: $0 -S <seed> -s <save_infer> -f <eval_freq> -m <milestone> -g <use_gpu> -J <job_name> -R <rounds> -r <round> -E <epochs> -e <epoch> -i <inst_id> -c <split_csv> -M <model_pth> -p <data_percentage>"
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
  --zoom 0 \
  --flip_lr 0 \
  --dataset CC359PPMI \
  --data_root cc359ppmi128 \
  --inst_root inst_* \
  --data_percentage $data_percentage \
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
  --milestones [$milestone] \
  --lr_gamma 0.1
