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

def load_subjects_list(split_path: str, inst_ids: list, TrainOrVal: list, mode='train'):
    df = pd.read_csv(split_path)
    if mode == 'train': # set(mode) == set(['train', 'val']):
        unique_inst_ids = [int(el) for el in set(df[df['TrainOrVal'].isin(TrainOrVal)]['Partition_ID'])]
        unique_inst_ids = unique_inst_ids if inst_ids == [-1] else [el for el in unique_inst_ids if el in inst_ids]
        filtered_df = df[df['Partition_ID'].isin(unique_inst_ids)]
        train_list = list(filtered_df[filtered_df['TrainOrVal'].isin(['train'])]['Subject_ID'])
        val_list = list(filtered_df[filtered_df['TrainOrVal'].isin(['val'])]['Subject_ID'])
        assert train_list.__len__() > 0, 'train list empty'
        assert val_list.__len__() > 0, 'val list empty'
        train_val_dict = {
            'inst_ids': unique_inst_ids,
            'train': train_list,
            'val': val_list,
        }
        return train_val_dict
    elif mode == 'test': # in mode:
        # assert inst_ids == [0], 'test must have 0 inst_id'
        unique_inst_ids = [int(el) for el in set(df[df['TrainOrVal'].isin(TrainOrVal)]['Partition_ID'])]
        unique_inst_ids = unique_inst_ids if inst_ids == [-1] else [el for el in unique_inst_ids if el in inst_ids]
        # assert unique_inst_ids == [0], 'test must have 0 Partition_ID'
        filtered_df = df[df['Partition_ID'].isin(unique_inst_ids)]
        test_list = list(filtered_df[filtered_df['TrainOrVal'].isin(TrainOrVal)]['Subject_ID'])
        test_dict = {
            'inst_ids': unique_inst_ids,
            'test': test_list,
        }
        return test_dict
    else:
        raise NotImplementedError(f"[MODE:{mode}] is not implemented on load_inst_cases()")


def save_img_nifti(image: Tensor, scale:int, names: list, mode: str, postfix:str, affine_src: str, modality: list, save_epoch_path: str):
    """
    Output val img in every iteration to save VRAM
    """
    # Convert image tensor to numpy
    image_numpy = image.cpu().numpy()
    B, _, H, W, D = image_numpy.shape

    # make save folder
    save_epoch_seg_path = join(save_epoch_path, f"{mode}_{postfix}")
    os.makedirs(save_epoch_seg_path, exist_ok=True)

    for b in range(B):
        # random modality is ok
        # original_img_path = join(data_root, src_path, names[b], affine_src)
        # affine = nib_affine(original_img_path)

        for ch_idx, el_modality in enumerate(modality):
            # Convert image tensor to numpy and scale to uint8
            image_modality = image_numpy[b][ch_idx].astype(np.float32)

            # Normalize the image data to 0-255 or 0-100
            image_modality -= image_modality.min()  # Shift data to 0
            image_modality /= image_modality.max()  # Normalize to 1
            image_modality = (image_modality * scale).astype(np.uint8)  # Scale to 0-255 and convert to uint8

            nib.save(
                nib.Nifti1Image(image_modality, affine_src[b]),
                join(save_epoch_seg_path, names[b] + f'_{postfix}_{el_modality}.nii.gz')
            )


def save_seg_nifti(seg_map: Tensor, names: list, mode: str, postfix:str, affine_src: str, label_map: list, save_epoch_path: str):
    """
    Output val seg map in every iteration to save VRAM
    """
    # Convert image tensor to numpy
    seg_map_numpy = seg_map.cpu().numpy()
    B, _, H, W, D = seg_map_numpy.shape

    # make save folder
    save_epoch_seg_path = join(save_epoch_path, f"{mode}_{postfix}")
    os.makedirs(save_epoch_seg_path, exist_ok=True)

    for b in range(B):
        output = seg_map_numpy[b]
        seg_img = np.zeros((H, W, D), dtype=np.uint16)

        for idx, lbl in enumerate(label_map):
            seg_img[np.where(output[idx, ...] == 1)] = lbl

        # random modality is ok
        # original_img_path = join(data_root, src_path, names[b], affine_src)
        # affine = nib_affine(original_img_path)

        nib.save(
            nib.Nifti1Image(seg_img, affine_src[b]),
            join(save_epoch_seg_path, f'{postfix}_{names[b]}.nii.gz')
        )
