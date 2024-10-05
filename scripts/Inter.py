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
from tqdm import tqdm

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
class Unet3DApp:
    def __init__(self, sys_argv=None):
        # 기본 로거 설정 (기본값 사용)
        args = parse_args(sys_argv)
        self.cli_args = args
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cpu")
        self.cli_args.multi_batch_size = self.cli_args.batch_size

        self.use_cuda = self.cli_args.use_gpu and torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda")
            self.selected_gpu_count = torch.cuda.device_count()
            self.cli_args.multi_batch_size *= self.selected_gpu_count
            
        self.cli_args.num_classes = self.cli_args.label_groups.__len__()
        if self.cli_args.label_index.__len__() != self.cli_args.num_classes:
            self.cli_args.label_index = list(range(1, 1+self.cli_args.num_classes))
        self.cli_args.input_channels = self.cli_args.input_channel_names.__len__()

    def setup_gpu(self, model):
        current_device = next(model.parameters()).device
        if current_device == self.device:
            # self.logger.info(f"Model is already on the correct device: {self.device}.")
            print(f"Model is already on the correct device: {self.device}.")
        else:
            # self.logger.info(f"Model is currently on {current_device}, moving to {self.device}.")
            print(f"Model is currently on {current_device}, moving to {self.device}.")
            model = model.to(self.device)
            if self.use_cuda and self.selected_gpu_count > 1:
                model = nn.DataParallel(model)
                # self.logger.info(f"DataParallel applied with {self.selected_gpu_count} GPUs.")
                print(f"DataParallel applied with {self.selected_gpu_count} GPUs.")
        return model

    def initModel(self, weight_path=None):
        model = get_unet(self.cli_args)
        # load model
        if weight_path is None:
            state = {'model': model.state_dict(), 'epoch': 0, 'args': self.cli_args}
            # exp_folder = f"{self.cli_args.exp_name}"
            # exp_folder = time.strftime("%Y%m%d_%H%M%S")
            save_model_path = os.path.join("states", self.timestamp, "models")
            os.makedirs(save_model_path, exist_ok=True)
            init_model_path = os.path.join(save_model_path, f"R{0:02}E{0:02}.pth")
            torch.save(state, init_model_path)
        else:
            # self.logger.info(f"==> Loading pretrain model: {self.cli_args.weight_path}...")
            assert weight_path.endswith(".pth")
            model_state = torch.load(weight_path, map_location='cpu')['model']
            new_state_dict = {k.replace('module.', ''): v for k, v in model_state.items()}
            model.load_state_dict(new_state_dict)
            init_model_path = weight_path
        # self.logger.info("Model initialized on CPU")

        return model, init_model_path

    def initTrainDl(self, case_names:list, mode='training'):
        base_transform = get_base_transform(self.cli_args)
        aug_transform = get_aug_transform(self.cli_args)
        train_transforms = transforms.Compose(base_transform + aug_transform)
        train_dataset = get_dataset(self.cli_args, case_names, 
                                    train_transforms,
                                    mode=mode,
                                    label_names=self.cli_args.label_names,
                                    custom_min_len=self.cli_args.min_dlen,
                                    custom_max_len=self.cli_args.max_dlen)

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=custom_collate,
            shuffle=True,
            drop_last=False,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)
        return train_dataset, train_loader


    def initValDl(self, case_names:list, mode:str):
        base_transform  = get_base_transform(self.cli_args)
        aug_transform = [transforms.EnsureTyped(keys=["image", 'label'])]
        infer_transform = transforms.Compose(base_transform + aug_transform)
        infer_dataset = get_dataset(self.cli_args, case_names, infer_transform, 
                                    mode=mode, 
                                    label_names=self.cli_args.label_names,
                                    custom_min_len=1,
                                    custom_max_len=self.cli_args.max_dlen)
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=self.cli_args.multi_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)
        return infer_dataset, infer_loader

    def initializer(self, subjects_dict, mode='train'):
        if mode == 'train':
            train_cases = natsort.natsorted(subjects_dict['train'])
            train_dataset, train_loader = self.initTrainDl(train_cases)

            val_cases = natsort.natsorted(subjects_dict['val'])
            val_dataset, val_loader = self.initValDl(val_cases, 'val')

            model, _ = self.initModel(self.cli_args.weight_path)
            model = self.setup_gpu(model)
            optimizer = get_optimizer(self.cli_args, model)
            loss_fn = SoftDiceBCEWithLogitsLoss(channel_weights=None).to(self.device)
            scaler = GradScaler() if self.cli_args.amp else None
            # scheduler = get_scheduler(self.cli_args, optimizer)

            return {
                'train_dataset': train_dataset,
                'train_loader': train_loader,
                'val_dataset': val_dataset,
                'val_loader': val_loader,
                'model': model,
                'optimizer': optimizer,
                'loss_fn': loss_fn,
                'scaler': scaler,
            }
        elif mode == 'test':
            # train_cases = natsort.natsorted(subjects_dict['test'])
            # train_dataset, train_loader = self.initTrainDl(train_cases)

            test_cases = natsort.natsorted(subjects_dict['test'])
            test_dataset, test_loader = self.initValDl(test_cases, 'test')

            model, _ = self.initModel(self.cli_args.weight_path)
            model = self.setup_gpu(model)
            # optimizer = get_optimizer(self.cli_args, model)
            loss_fn = SoftDiceBCEWithLogitsLoss(channel_weights=None).to(self.device)
            # scaler = GradScaler() if self.cli_args.amp else None
            # scheduler = get_scheduler(self.cli_args, optimizer)

            return {
                # 'train_dataset': train_dataset,
                # 'train_loader': train_loader,
                'test_dataset': test_dataset,
                'test_loader': test_loader,
                'model': model,
                # 'optimizer': optimizer,
                'loss_fn': loss_fn,
                # 'scaler': scaler,
            }
        else:
            raise NotImplementedError(f"[MODE:{mode}] is not implemented on local_initializer()")
    
    def infer(self, round, model: nn.Module, 
                infer_loader, loss_fn, 
                mode: str, save_pred: bool = True):
        model.eval()

        save_val_path = os.path.join("states", self.timestamp, f"R{round:02}")
        os.makedirs(save_val_path, exist_ok=True)
        # folder_lv3 = f"{mode}_epoch_{epoch:03d}"
        # save_epoch_path = os.path.join("states", folder_dir1, folder_dir2, folder_dir3)
        # if not os.path.exists(save_epoch_path):
        #     os.makedirs(save_epoch_path, exist_ok=True)

        # seg_names = self.cli_args.label_names
        with torch.no_grad():
            for i, (image, label, _, name, affine, label_names) in enumerate(tqdm(infer_loader, desc=f"Infer Progress")):
                # if i % 10 != 0:
                #     continue
                label_name = [el[0] for el in label_names]
                if self.cli_args.use_gpu:
                    image, label = image.float().cuda(), label.float().cuda()
                else:
                    image, label = image.float(), label.float()

                # get seg map
                seg_map = sliding_window_inference(
                    inputs=image,
                    predictor=model,
                    roi_size=self.cli_args.patch_size,
                    sw_batch_size=self.cli_args.sw_batch_size,
                    overlap=self.cli_args.patch_overlap,
                    mode=self.cli_args.sliding_window_mode
                )
                # val loss
                # _, dsc_loss_by_channels = loss_fn(seg_map, label)

                if self.cli_args.unet_arch == 'unet':
                    seg_map = robust_sigmoid(seg_map)
                else:
                    msg = f"currently model is {self.cli_args.unet_arch}.\n If the model is not unet, it is necessary to check the value range of seg_map before applying any thresholding."
                    print(msg)
                    raise NotImplementedError(msg)

                # discrete
                seg_map_th = torch.where(seg_map > 0.5, True, False)

                # output seg map
                if save_pred: # and (round == 0):
                    if (i % 1 == 0) and mode in ['pre', 'test']:
                        modality = self.cli_args.input_channel_names
                        scale = 255
                        save_img_nifti(image, scale, name, mode[:4], 'img',
                                       affine, modality, save_val_path)
                        label_map = self.cli_args.label_index
                        save_seg_nifti(label, name, mode[:4], 'labels',
                                       affine, label_map, save_val_path)

                    if (i % 1 == 0):
                        # seg_labels = label_names
                        scale = 100
                        save_img_nifti(seg_map, scale, name, mode[:4], 'prob',
                                    affine, label_name, save_val_path)
                        label_map = self.cli_args.label_index
                        save_seg_nifti(seg_map_th, name, mode[:4], 'pred',
                                    affine, label_map, save_val_path)


    def main(self):
        test_dict = load_subjects_list(
            self.cli_args.cases_split, self.cli_args.inst_ids, TrainOrVal=['train','val','test'], mode='test')
        
        _, self.cli_args.weight_path = self.initModel(self.cli_args.weight_path)

        test_setup = self.initializer(test_dict, mode='test')

        # Pre-Validation
        self.infer(self.cli_args.round, test_setup['model'], 
            test_setup['test_loader'], test_setup['loss_fn'], 
            mode='test')

if __name__ == '__main__':
    args = sys.argv[1:]
    App_args = Unet3DApp(args)
    App_args.main()
