#!/bin/bash
python3 scripts/run_train.py \
     --save_infer 0 \
     --eval_freq 10 \
     --use_gpu 0 \
     --job_name testfed \
     --rounds 3 \
     --round 0 \
     --epochs 1 \
     --epoch 0 \
     \
     --resize 128 \
     --patch_size 128 \
     --dataset CC359PPMI \
     --data_root cc359ppmi128 \
     --inst_root inst_* \
     \
     --cases_split cc359ppmi128/CC359PPMI128_fed-test.csv \
     --inst_ids [1] \
     --batch_size 1 \
     \
     --weight_path None \
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
     --milestones [15] \
     --lr_gamma 0.1
