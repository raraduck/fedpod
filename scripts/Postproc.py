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
# class Post3DApp:
#     def __init__(self, sys_argv=None):
#         # 기본 로거 설정 (기본값 사용)
#         args = parse_args(sys_argv)
#         self.cli_args = args
#         self.timestamp = time.strftime("%Y%m%d_%H%M%S")
            
#         self.cli_args.num_classes = self.cli_args.label_groups.__len__()
#         if self.cli_args.label_index.__len__() != self.cli_args.num_classes:
#             self.cli_args.label_index = list(range(1, 1+self.cli_args.num_classes))
#         self.cli_args.input_channels = self.cli_args.input_channel_names.__len__()

#     def initValDl(self, case_names:list, mode:str):
#         base_transform  = get_base_transform(self.cli_args)
#         aug_transform = [transforms.EnsureTyped(keys=["image", 'label'])]
#         infer_transform = transforms.Compose(base_transform + aug_transform)
#         infer_dataset = get_dataset(self.cli_args, case_names, infer_transform, 
#                                     mode=mode, 
#                                     label_names=self.cli_args.label_names,
#                                     custom_min_len=1,
#                                     custom_max_len=99999)
#         infer_loader = DataLoader(
#             infer_dataset,
#             batch_size=1,
#             shuffle=False,
#             drop_last=False,
#             num_workers=self.cli_args.num_workers,
#             pin_memory=False)
#         return infer_dataset, infer_loader

#     def main(self):
#         train_val_dict = load_subjects_list(
#             self.cli_args.cases_split, self.cli_args.inst_ids, TrainOrVal=['test'], mode='test')

#         test_cases = natsort.natsorted(train_val_dict['test'])
#         test_dataset, test_loader = self.initValDl(test_cases, 'test')
        
#         save_val_path = os.path.join("states", self.timestamp, f"test")
#         os.makedirs(save_val_path, exist_ok=True)
#         for i, (image, label, _, name, affine, label_names) in enumerate(test_loader):
#             # label_name = [el[0] for el in label_names]
#             image, label = image.float(), label.float()
#             modality = self.cli_args.input_channel_names
#             scale = 255
#             save_img_nifti(image, scale, name, 'preproc', 'img',
#                             affine, modality, save_val_path)
#             label_map = self.cli_args.label_index
#             save_seg_nifti(label, name, 'preproc', 'labels',
#                             affine, label_map, save_val_path)

import json
if __name__ == '__main__':
    job_list = ['cen1R12_0', 'trf1R12_0', 'fed1R12_0', 'sol1R12_0', 'sol2R12_0', 'sol3R12_0', 'sol4R12_0', 'sol5R12_0', 'sol6R12_0']
    base_dir = os.path.join('/backup', 'fedpod', 'temp', 'v1', 'logs', )
    # job_list = ['cen2R12_0', 'trf2R12_0', 'fed2R12_0', 'solo1R12_0', 'solo2R12_0', 'solo3R12_0', 'solo4R12_0', 'solo5R12_0', 'solo6R12_0']
    # base_dir = os.path.join('/backup', 'fedpod', 'temp', 'v2', 'logs', )
    # json_pattern = os.path.join(base_dir, job_list[0], '*.json')
    # json_path = glob.glob(json_pattern)
    # json_path_sorted = natsort.natsorted(json_path)

    mean_by_job = {}
    for job in job_list:
        json_pattern = os.path.join(base_dir, job, '*.json')
        json_path = glob.glob(json_pattern)
        json_path_sorted = natsort.natsorted(json_path)
        json_dict = {}
        if os.path.exists(json_path_sorted[0]):
            with open(json_path_sorted[0], 'r', encoding='utf-8') as file:
                json_dict = json.load(file)

        prev_mean_list = []
        for rnd, v in json_dict.items():
            # print(rnd, v.keys())
            prev_dice_at_rnd = []
            prev_mean = 0
            for inst, prev_post in v.items():
                # print(inst, prev_post.keys())
                for when, metrics in prev_post.items():
                    # print(rnd, inst, when, list(metrics.keys())[0])
                    if when == 'prev':
                        # print(f"{int(rnd):2d}, {inst}, {when}, DICE_AVG: {metrics['DICE_AVG']:5.3f}, DSCL_AVG: {metrics['DSCL_AVG']:5.3f}")
                        prev_dice_at_rnd.append(metrics['DICE_AVG']) 
            # print(prev_dice_at_rnd, sum(prev_dice_at_rnd)/len(prev_dice_at_rnd))
            prev_mean = sum(prev_dice_at_rnd)/len(prev_dice_at_rnd)
            # print(f"{job} mean: {prev_mean:5.3f}")
            prev_mean_list.append(f"{prev_mean:.3f}")
        mean_by_job[job] = prev_mean_list
        print(f"{job}: {prev_mean_list}")

    # print(json_path_sorted)
    # for el in json_path_sorted:
        # print(el)

    # job_dir = os.path.join(logs_dir, f"{args.job_prefix}_{args.inst_id}")
    # os.makedirs(job_dir, exist_ok=True)

    # last-metrics 파일 작성
    # metrics_file = os.path.join(job_dir, filename)

    # 기존 데이터 로드 또는 초기화
    # if os.path.exists(metrics_file):
    #     with open(metrics_file, 'r', encoding='utf-8') as file:
    #         json_metrics_dict = json.load(file)
    # else:
    #     json_metrics_dict = {}
    # args = sys.argv[1:]
    # App_args = Post3DApp(args)
    # App_args.main()
    
    # fed_round_to_json(args, logger, local_dict, f'{args.job_prefix}.json')

    # load json

    
    # for jobname, job_dict in local_dict.items():
    #     writer = SummaryWriter(os.path.join('runs', args.job_prefix, jobname))
    #     for prev_post, metric_dict in job_dict.items():
    #         for metric_name, value in metric_dict.items():
    #             writer.add_scalar(f"{prev_post}/{metric_name}", value, args.round)
    #     writer.flush()
    #     writer.close()
