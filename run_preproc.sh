python scripts/Preproc.py \
    --zoom \
    --resize 128 \
    --patch_size 128 \
    --dataset CC359PPMI \
    --data_root cc359ppmi \
    --inst_root inst_0* \
    \
    --cases_split cc359ppmi/CC359PPMI.csv \
    --inst_ids [0] \
    \
	--input_channel_names [t1] \
	--label_groups [[26,26],[58,58],[11,11],[50,50],[12,12],[51,51]] \
	--label_names [L_Accu,R_Accu,L_Caud,R_Caud,L_Puta,R_Puta] \
	--label_index [26,58,11,50,12,51] \