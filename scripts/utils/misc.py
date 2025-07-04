import os
import sys
import random
import logging
import os
import natsort
from os.path import join
from collections import OrderedDict
# from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
MASKS=['seg','ref','sub','label']
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    
def load_subjects_list(percentile: int, rounds: int, round: int, split_path: str, inst_ids: list, TrainOrVal: list, mode='train'):
    ## split csv 파일을 최초에는 split_path (= self.cli_args.cases_split) 경로에서 불러와야함
    ## 이후에는, states/${job-name}/f"{split_path}" 경로에서 불러와서 갱신까지 하도록 해야함
    ## 갱신해야하는 정보는 현재 Round 에서 추출하는 데이터를 기록하는 것
    df = pd.read_csv(split_path)
    Partition_ID = f"Partition_ID"
    rounds_list = [el for el in df.columns.to_list() if 'R' in el]
    assert rounds > 0, f"Rounds must be bigger than 1 otherwise raise exception"
    assert rounds_list.__len__() >= rounds, f"{split_path} has not enough columns of {rounds_list} to run {rounds} rounds."
    assert round <= rounds, f"Round must be smaller than rounds."
    assert f"R{round}" in rounds_list, f"{split_path} does not have R{round} column."

    if mode == 'train': # set(mode) == set(['train', 'val']):
        TrainOrVal_in_partition = df[df['TrainOrVal'].isin(TrainOrVal)][Partition_ID].dropna()
        assert TrainOrVal_in_partition.__len__() > 0, f"Not found train or val from current round {Partition_ID}, please check csv file {split_path}"
        unique_inst_ids = [int(el) for el in set(TrainOrVal_in_partition)]

        # assert len(inst_ids) == 1, f"[TRAIN] inst_ids parameters are not allowed to be multiply selected."
        # assert inst_ids[0] > 0, f"inst_ids 0 is not optional (legacy was for all selection)"
        unique_inst_ids = [el for el in unique_inst_ids if el in inst_ids]
        # unique_inst_ids = [el for el in unique_inst_ids if el not in [0]]

        Partition_round = f"R{round}"
        # trainset 의 경우에는 손실값이 있으면 손실값 기준으로 정렬하도록 하기 (R0, R1, R2, R3 ...)
        filtered_df = df[df[Partition_ID].isin(unique_inst_ids)] # [['Partition_ID','Subject_ID','TrainOrVal',f"{Partition_round}"]]
        if 'experiments' in split_path:
            train_list = list(filtered_df[filtered_df['TrainOrVal'].isin(['train'])]['Subject_ID'])
            total_len = len(train_list)
            percentile_indice = max(1,int(total_len * (percentile / 100)))
            random.shuffle(train_list)
            # train_list.sort()
            percentile_train_list = train_list[:percentile_indice]
        else:
            # Partition_round = f"R10"
            train_list = list(filtered_df[filtered_df['TrainOrVal'].isin(['train'])].sort_values(by=Partition_round, ascending=False)['Subject_ID'])
            total_len = len(train_list)
            percentile_indice = max(1,int(total_len * (percentile / 100)))
            # percentile_train_list = train_list[:percentile_indice]
            percentile_train_list = train_list[:percentile_indice]
            random.shuffle(percentile_train_list)

            # randomly change 20% of percentile_train_list with that of the rest of the train_list
            num_to_replace = int(np.floor(0.2 * len(percentile_train_list)))

            candidates_outside = train_list[percentile_indice:]
            # 후보군에서 교체할 요소를 랜덤하게 선택 (num_to_replace개)
            if len(candidates_outside) >= num_to_replace and num_to_replace > 0:
                replacement_candidates = list(np.random.choice(candidates_outside, num_to_replace, replace=False))
            else:
                replacement_candidates = list(candidates_outside)  # 후보가 부족하면 가능한 만큼 사용

            # percentile_train_list 내에서 랜덤하게 교체할 인덱스 선택 (num_to_replace개)
            if num_to_replace > 0:
                indices_to_replace = np.random.choice(range(len(percentile_train_list)), size=num_to_replace, replace=False)
                for idx, new_val in zip(indices_to_replace, replacement_candidates):
                    percentile_train_list[idx] = new_val

            # 최종 리스트 섞기 (선택사항)
            random.shuffle(percentile_train_list)

        # for debugging
        # DSC_list = list(filtered_df[filtered_df['TrainOrVal'].isin(['train'])].sort_values(by=Partition_round, ascending=True)[Partition_round])
        # print(f"from {train_list} to {percentile_train_list} in percentile ({percentile_indice}) out of total_len ({total_len})")
        # print(f"TRAIN_LIST: {train_list[:percentile_indice]}")
        # for debugging
        # print(f"DSC_LIST: {DSC_list[:percentile_indice]}")

        val_list = list(filtered_df[filtered_df['TrainOrVal'].isin(['val'])]['Subject_ID'])
        

        # loss_list = filtered_df[Partition_round]
        

        # assert train_list.__len__() > 0, 'train list empty'
        assert val_list.__len__() > 0, 'val list empty'
        train_val_dict = {
            'inst_ids': unique_inst_ids,
            'train': percentile_train_list, # natsort (default)
            'val': natsort.natsorted(val_list), # natsort (default)
        }
        return train_val_dict
    elif mode in ['val', 'test']: # in mode:
        # assert inst_ids == [0], 'test must have 0 inst_id'
        TrainOrVal_in_partition = df[df['TrainOrVal'].isin(TrainOrVal)][Partition_ID].dropna()
        assert TrainOrVal_in_partition.__len__() > 0, f"Not found train or val from current round {Partition_ID}, please check csv file {split_path}"
        unique_inst_ids = [int(el) for el in set(TrainOrVal_in_partition)]
        # assert len(inst_ids) == 1, f"[VAL or TEST] inst_ids parameters are not allowed to be multiply selected."
        # assert inst_ids[0] > 0, f"inst_ids 0 is not optional (legacy was for all selection)"
        unique_inst_ids = [el for el in unique_inst_ids if el in inst_ids]
        # unique_inst_ids = [el for el in unique_inst_ids if el not in [0]]
        # assert unique_inst_ids == [0], 'test must have 0 Partition_ID'
        filtered_df = df[df[Partition_ID].isin(unique_inst_ids)]
        infer_list = list(filtered_df[filtered_df['TrainOrVal'].isin(TrainOrVal)]['Subject_ID'])
        infer_dict = {
            'inst_ids': unique_inst_ids,
            'infer': natsort.natsorted(infer_list), # natsort (default)
        }
        return infer_dict
    elif mode in ['quant']: # in mode:
        # assert inst_ids == [0], 'test must have 0 inst_id'
        TrainOrVal_in_partition = df[df['TrainOrVal'].isin(TrainOrVal)][Partition_ID].dropna()
        assert TrainOrVal_in_partition.__len__() > 0, f"Not found train or val from current round {Partition_ID}, please check csv file {split_path}"
        unique_inst_ids = [int(el) for el in set(TrainOrVal_in_partition)]
        # assert len(inst_ids) == 1, f"[VAL or TEST] inst_ids parameters are not allowed to be multiply selected."
        # assert inst_ids[0] > 0, f"inst_ids 0 is not optional (legacy was for all selection)"
        unique_inst_ids = [el for el in unique_inst_ids if el in inst_ids]
        # unique_inst_ids = [el for el in unique_inst_ids if el not in [0]]
        # assert unique_inst_ids == [0], 'test must have 0 Partition_ID'
        filtered_df = df[df[Partition_ID].isin(unique_inst_ids)]
        infer_list = list(filtered_df[filtered_df['TrainOrVal'].isin(TrainOrVal)]['Subject_ID'])
        infer_dict = {
            'inst_ids': unique_inst_ids,
            'infer': natsort.natsorted(infer_list), # natsort (default)
        }
        return infer_dict
    else:
        raise NotImplementedError(f"[MODE:{mode}] is not implemented on load_inst_cases()")


def save_img_nifti(image: Tensor, prefix: str, postfix: str, affine_src: str, modality: list, save_epoch_path: str, plist: list):
    """
    Output val img in every iteration to save VRAM
    """
    # Convert image tensor to numpy
    image_numpy = image.cpu().numpy()
    B, _, H, W, D = image_numpy.shape

    # make save folder

    for b in range(B):
        save_epoch_seg_path = join(save_epoch_path, plist[b])
        os.makedirs(save_epoch_seg_path, exist_ok=True)
        # random modality is ok
        # original_img_path = join(data_root, src_path, names[b], affine_src)
        # affine = nib_affine(original_img_path)

        for ch_idx, el_modality in enumerate(modality):
            # Convert image tensor to numpy and scale to uint8
            
            image_modality = image_numpy[b][ch_idx].astype(np.float32)
            if el_modality not in MASKS:
                scale = 255
                # Normalize the image data to 0-255 or 0-100
                if el_modality in ['t1', 't1ce', 't2', 'flair']:
                    image_modality -= image_modality.min()  # Shift data to 0
                    image_modality /= image_modality.max()  # Normalize to 1
                else:
                    image_modality = np.clip(image_modality, 0, 1)
                image_modality = (image_modality * scale).astype(np.uint8)  # Scale to 0-255 and convert to uint8

            nib.save(
                nib.Nifti1Image(image_modality, affine_src[b]),
                join(save_epoch_seg_path, f'{prefix}{plist[b]}_{el_modality}{postfix}.nii.gz')
            )


def save_seg_nifti(seg_map: Tensor, prefix: str, postfix: str, affine_src: str, label_map: list, save_epoch_path: str, plist: list):
    """
    Output val seg map in every iteration to save VRAM
    """
    # Convert image tensor to numpy
    seg_map_numpy = seg_map.cpu().numpy()
    B, _, H, W, D = seg_map_numpy.shape

    # make save folder

    for b in range(B):
        save_epoch_seg_path = join(save_epoch_path, plist[b])
        os.makedirs(save_epoch_seg_path, exist_ok=True)
        
        output = seg_map_numpy[b]
        seg_img = np.zeros((H, W, D), dtype=np.uint16)

        for idx, lbl in enumerate(label_map):
            seg_img[np.where(output[idx, ...] == 1)] = lbl

        # random modality is ok
        # original_img_path = join(data_root, src_path, names[b], affine_src)
        # affine = nib_affine(original_img_path)

        nib.save(
            nib.Nifti1Image(seg_img, affine_src[b]),
            join(save_epoch_seg_path, f'{prefix}{plist[b]}{postfix}.nii.gz')
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def setup_logger(log_dir, log_file):

    root_logger = logging.getLogger()
    # logger.propagate = False
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    logfmt_str = "%(asctime)s %(levelname)-2s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)-12s %(message)s"
    formatter = logging.Formatter(logfmt_str, datefmt="%Y%m%d-%H%M%S")

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.DEBUG)
    root_logger.addHandler(streamHandler)

    # 파일 핸들러 설정 (선택적)
    # if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(filename=os.path.join(log_dir, log_file), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(logfmt_str)
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    root_logger.setLevel(logging.DEBUG)  # 전체 로거의 기본 레벨 설정
    return root_logger  # 로거 인스턴스를 반환합니다.


def initialization_logger(args, jobname, filename):
    # set random seed
    # seed_everything(args.seed)

    # make exp dir
    # writer = SummaryWriter(os.path.join('runs', args.exp_name))

    # init logger & save args
    basedir = os.path.join('logs', jobname)
    os.makedirs(basedir, exist_ok=True)
    logger = setup_logger(log_dir=basedir, log_file=filename)
    logger.info(f"\n{'-' * 20} New Experiment {'-' * 20}\n")
    # logger.info(' '.join(sys.argv))
    logger.info(args)

    return logger


class CaseSegMetricsMeter(object):
    """Stores segmentation metric (dice & hd95) for every case"""

    def __init__(self, label_names, metrics_list=['DSCL','DICE','HD95','PVDC']):
        self.cols = []
        for m in metrics_list:
            self.cols += [
                *[f'{m}_{el}' for el in label_names],
            ]
        # self.cols = [
        #     *[f'DSCL_{el}' for el in label_names],
        #     *[f'DICE_{el}' for el in label_names],
        #     *[f'HD95_{el}' for el in label_names],
        #     *[f'PVDC_{el}' for el in label_names],
        # ]
        self.reset()

    def reset(self):
        self.cases = pd.DataFrame(columns=self.cols)

    def update(self, dice, hd95, pdvc, names, bsz, dscloss=None, suv1=None, suv2=None):
        ch_size = dice.shape[1]
        if dscloss is not None:
            for i in range(bsz):
                self.cases.loc[names[i]] = [
                    *[dscloss[i, idx] for idx in range(ch_size)],
                    *[dice[i, idx] for idx in range(ch_size)],
                    *[hd95[i, idx] for idx in range(ch_size)],
                    *[pdvc[i, idx] for idx in range(ch_size)],
                ]
        elif suv1 is not None and suv2 is not None:
            for i in range(bsz):
                self.cases.loc[names[i]] = [
                    *[dice[i, idx] for idx in range(ch_size)],
                    *[hd95[i, idx] for idx in range(ch_size)],
                    *[pdvc[i, idx] for idx in range(ch_size)],
                    *[suv1[i, idx] for idx in range(ch_size)],
                    *[suv2[i, idx] for idx in range(ch_size)],
                ]
        else:
            for i in range(bsz):
                self.cases.loc[names[i]] = [
                    *[dice[i, idx] for idx in range(ch_size)],
                    *[hd95[i, idx] for idx in range(ch_size)],
                    *[pdvc[i, idx] for idx in range(ch_size)],
                ]

    def mean(self):
        return self.cases.mean(0).to_dict()

    def output(self, save_epoch_path):
        # all cases csv
        self.cases.to_csv(join(save_epoch_path, "case_metrics.csv"))
        # summary txt
        self.cases.mean(0).to_csv(join(save_epoch_path, "case_metrics_summary.txt"), sep='\t')
