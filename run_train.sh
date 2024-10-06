python scripts/run_train.py \
    --rounds 5 \
    --round 0 \
    --epochs 10 \
    \
    --zoom \
    --resize 128 \
    --patch_size 128 \
    --dataset CC359PPMI \
    --data_root cc359ppmi \
    --inst_root inst_0* \
    \
    --cases_split cc359ppmi/CC359PPMI.csv \
    --inst_ids [-1] \
    --batch_size 1 \
    --use_gpu \
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