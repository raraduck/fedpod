import os
import sys
import random
import logging
import os
from os.path import join
from collections import OrderedDict
# from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    
def load_subjects_list(rounds: int, round: int, split_path: str, inst_ids: list, TrainOrVal: list, mode='train'):
    df = pd.read_csv(split_path)
    Partition_round = f"R{round}"
    rounds_list = [el for el in df.columns.to_list() if 'R' in el]
    assert rounds > 0, f"Rounds must be bigger than 1 otherwise raise exception"
    assert rounds_list.__len__() >= rounds, f"{split_path} has not enough columns of {rounds_list} to run {rounds} rounds."
    assert round <= rounds, f"Round must be smaller than rounds."
    assert f"R{round}" in rounds_list, f"{split_path} does not have R{round} column."

    if mode == 'train': # set(mode) == set(['train', 'val']):
        TrainOrVal_in_partition = df[df['TrainOrVal'].isin(TrainOrVal)][Partition_round].dropna()
        assert TrainOrVal_in_partition.__len__() > 0, f"Not found train or val from current round {Partition_round}, please check csv file {split_path}"
        unique_inst_ids = [int(el) for el in set(TrainOrVal_in_partition)]

        assert len(inst_ids) == 1, f"[TRAIN] inst_ids parameters are not allowed to be multiply selected."
        # assert inst_ids[0] > 0, f"inst_ids 0 is not optional (legacy was for all selection)"
        unique_inst_ids = [el for el in unique_inst_ids if el in inst_ids]

        filtered_df = df[df[Partition_round].isin(unique_inst_ids)]
        train_list = list(filtered_df[filtered_df['TrainOrVal'].isin(['train'])]['Subject_ID'])
        val_list = list(filtered_df[filtered_df['TrainOrVal'].isin(['val'])]['Subject_ID'])
        # assert train_list.__len__() > 0, 'train list empty'
        assert val_list.__len__() > 0, 'val list empty'
        train_val_dict = {
            'inst_ids': unique_inst_ids,
            'train': train_list,
            'val': val_list,
        }
        return train_val_dict
    elif mode in ['val', 'test']: # in mode:
        # assert inst_ids == [0], 'test must have 0 inst_id'
        TrainOrVal_in_partition = df[df['TrainOrVal'].isin(TrainOrVal)][Partition_round].dropna()
        assert TrainOrVal_in_partition.__len__() > 0, f"Not found train or val from current round {Partition_round}, please check csv file {split_path}"
        unique_inst_ids = [int(el) for el in set(TrainOrVal_in_partition)]
        assert len(inst_ids) == 1, f"[VAL or TEST] inst_ids parameters are not allowed to be multiply selected."
        # assert inst_ids[0] > 0, f"inst_ids 0 is not optional (legacy was for all selection)"
        unique_inst_ids = [el for el in unique_inst_ids if el in inst_ids]
        # assert unique_inst_ids == [0], 'test must have 0 Partition_ID'
        filtered_df = df[df[Partition_round].isin(unique_inst_ids)]
        infer_list = list(filtered_df[filtered_df['TrainOrVal'].isin(TrainOrVal)]['Subject_ID'])
        infer_dict = {
            'inst_ids': unique_inst_ids,
            'infer': infer_list,
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
            if el_modality not in ['seg','ref']:
                scale = 255
                # Normalize the image data to 0-255 or 0-100
                image_modality -= image_modality.min()  # Shift data to 0
                image_modality /= image_modality.max()  # Normalize to 1
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

    def __init__(self, label_names):
        self.cols = [
            *[f'DSCL_{el}' for el in label_names],
            *[f'DICE_{el}' for el in label_names],
            *[f'HD95_{el}' for el in label_names],
            *[f'PVDC_{el}' for el in label_names],
        ]
        self.reset()

    def reset(self):
        self.cases = pd.DataFrame(columns=self.cols)

    def update(self, dscloss, dice, hd95, pdvc, names, bsz):
        ch_size = dice.shape[1]
        for i in range(bsz):
            self.cases.loc[names[i]] = [
                *[dscloss[i, idx] for idx in range(ch_size)],
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
