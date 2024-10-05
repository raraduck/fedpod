import copy
import os
import sys
import time
import random
import math
import glob
import shutil
import traceback
from collections import Counter
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import natsort
import monai.transforms as transforms
from monai.inferers import sliding_window_inference

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils.configs import parse_args
from utils.misc import *
from models import get_unet
from dsets import get_dataset, get_base_transform, get_aug_transform, custom_collate
from utils.optim import get_optimizer
from utils.loss import SoftDiceBCEWithLogitsLoss, robust_sigmoid
from torch.cuda.amp import GradScaler, autocast

# from PIL import Image
# import torchvision.transforms as tf
# ToPILImage 변환기 초기화
# to_pil = tf.ToPILImage()
class Pre3DApp:
    def __init__(self, sys_argv=None):
        # 기본 로거 설정 (기본값 사용)
        args = parse_args(sys_argv)
        self.cli_args = args
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
            
        self.cli_args.num_classes = self.cli_args.label_groups.__len__()
        if self.cli_args.label_index.__len__() != self.cli_args.num_classes:
            self.cli_args.label_index = list(range(1, 1+self.cli_args.num_classes))
        self.cli_args.input_channels = self.cli_args.input_channel_names.__len__()

    def initValDl(self, case_names:list, mode:str):
        base_transform  = get_base_transform(self.cli_args)
        aug_transform = [transforms.EnsureTyped(keys=["image", 'label'])]
        infer_transform = transforms.Compose(base_transform + aug_transform)
        infer_dataset = get_dataset(self.cli_args, case_names, infer_transform, 
                                    mode=mode, 
                                    label_names=self.cli_args.label_names,
                                    custom_min_len=1,
                                    custom_max_len=99999)
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.cli_args.num_workers,
            pin_memory=False)
        return infer_dataset, infer_loader

    def main(self):
        train_val_dict = load_subjects_list(
            self.cli_args.cases_split, self.cli_args.inst_ids, TrainOrVal=['test'], mode='test')

        test_cases = natsort.natsorted(train_val_dict['test'])
        test_dataset, test_loader = self.initValDl(test_cases, 'test')
        
        save_val_path = os.path.join("states", self.timestamp, f"test")
        os.makedirs(save_val_path, exist_ok=True)
        for i, (image, label, _, name, affine, label_names) in enumerate(test_loader):
            # label_name = [el[0] for el in label_names]
            image, label = image.float(), label.float()
            modality = self.cli_args.input_channel_names
            scale = 255
            save_img_nifti(image, scale, name, 'preproc', 'img',
                            affine, modality, save_val_path)
            label_map = self.cli_args.label_index
            save_seg_nifti(label, name, 'preproc', 'labels',
                            affine, label_map, save_val_path)

if __name__ == '__main__':
    args = sys.argv[1:]
    App_args = Pre3DApp(args)
    App_args.main()
