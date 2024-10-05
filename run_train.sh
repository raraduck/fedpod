python scripts/Train.py \
    --rounds 5 \
    --round 0 \
    --epochs 3 \
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
	--input_channel_names [t1,striatum] \

	--label_groups [[261,111,121],[112,112],[113,113],[122,122],[123,123],[124,124],[581,501,511],[502,502],[503,503],[512,512],[513,513],[514,514]] \
	--label_names [LVS,LAC,LAP,LAP,LPP,LVP,RVS,RAC,RAP,RAP,RPP,RVP] \
	--label_index [1,2,3,4,5,6,7,8,9,10,11,12] \
	--unet_arch unet \
	--channels_list [32,64,128,256] \
	--block res \
	--ds_layer 1 \
	--kernel_size 3 \
	--norm instance