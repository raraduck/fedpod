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
from torch.cuda.amp import GradScaler, autocast

from utils.configs import parse_args
from utils.misc import *
import utils.metrics as metrics
from models import get_unet
from dsets import get_dataset, get_base_transform, get_aug_transform, custom_collate
from utils.optim import get_optimizer
from utils.loss import SoftDiceBCEWithLogitsLoss, robust_sigmoid
from utils.scheduler import get_scheduler
# from PIL import Image
# import torchvision.transforms as tf
# ToPILImage 변환timestamp기 초기화
# to_pil = tf.ToPILImage()
class Unet3DApp:
    def __init__(self, sys_argv=None):
        # 기본 로거 설정 (기본값 사용)
        args = parse_args(sys_argv)
        self.cli_args = args
        self.cli_args.weight_path = None if self.cli_args.weight_path == "None" else self.cli_args.weight_path
        self.cli_args.amp = True if self.cli_args.amp else False
        self.job_name = self.cli_args.job_name if self.cli_args.job_name else time.strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cpu")
        log_filename = f"{self.cli_args.job_name}_R{self.cli_args.rounds:02}r{self.cli_args.round:02}.log"
        self.logger = initialization_logger(self.cli_args, log_filename)
        self.logger.info(f"[{self.cli_args.job_name.upper()}][INIT] created logger at job_name-{self.job_name}...")
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

    def setup_gpu(self, model, mode):
        current_device = next(model.parameters()).device
        if current_device == self.device:
            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode}] Model is already on the correct device: {self.device}.")
            # print(f"Model is already on the correct device: {self.device}.")
        else:
            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode}] Model is currently on {current_device}, moving to {self.device}.")
            # print(f"Model is currently on {current_device}, moving to {self.device}.")
            model = model.to(self.device)
            if self.use_cuda and self.selected_gpu_count > 1:
                model = nn.DataParallel(model)
                self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode}] DataParallel applied with {self.selected_gpu_count} GPUs.")
                # print(f"DataParallel applied with {self.selected_gpu_count} GPUs.")
        return model

    def initModel(self, weight_path=None, mode='INIT'):
        model = get_unet(self.cli_args)
        # load model
        if weight_path is None:
            state = {'model': model.state_dict(), 'epoch': 0, 'args': self.cli_args}
            # exp_folder = f"{self.cli_args.exp_name}"
            # exp_folder = time.strftime("%Y%m%d_%H%M%S")
            save_model_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", "models")
            os.makedirs(save_model_path, exist_ok=True)
            init_model_path = os.path.join(save_model_path, f"R{0:02}r{0:02}.pth")
            torch.save(state, init_model_path)
            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode}] ==> Initialize random model at {init_model_path}...")
            self.logger.warning(f"[{self.cli_args.job_name.upper()}][{mode}] ==> Created random model at round {self.cli_args.round} not at round 0...")
        else:
            assert weight_path.endswith(".pth")
            model_state = torch.load(weight_path, map_location='cpu')['model']
            new_state_dict = {k.replace('module.', ''): v for k, v in model_state.items()}
            model.load_state_dict(new_state_dict)
            init_model_path = weight_path
            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode}] ==> Loading model from: {init_model_path}...")

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
        if mode in ['train', 'TRN']:
            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with train_loader and val_loader...")
            train_cases = natsort.natsorted(subjects_dict['train'])
            train_dataset, train_loader = self.initTrainDl(train_cases)
            val_cases = natsort.natsorted(subjects_dict['val'])
            val_dataset, val_loader = self.initValDl(val_cases, 'val')

            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with model...")
            model, _ = self.initModel(self.cli_args.weight_path, mode.upper())
            model = self.setup_gpu(model, mode=mode.upper())
            optimizer = get_optimizer(self.cli_args, model)
            loss_fn = SoftDiceBCEWithLogitsLoss(channel_weights=None).to(self.device)
            scaler = GradScaler() if self.cli_args.amp else None
            scheduler = get_scheduler(self.cli_args, optimizer)

            return {
                'train_dataset': train_dataset,
                'train_loader': train_loader,
                'val_dataset': val_dataset,
                'val_loader': val_loader,
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'loss_fn': loss_fn,
                'scaler': scaler,
            }
        elif mode in ['val', 'test']:
            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with infer_loader...")
            infer_cases = natsort.natsorted(subjects_dict['infer'])
            infer_dataset, infer_loader = self.initValDl(infer_cases, mode)

            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with model...")
            model, _ = self.initModel(self.cli_args.weight_path, mode=mode.upper())
            model = self.setup_gpu(model, mode=mode.upper())
            loss_fn = SoftDiceBCEWithLogitsLoss(channel_weights=None).to(self.device)

            return {
                'infer_dataset': infer_dataset,
                'infer_loader': infer_loader,
                'model': model,
                'loss_fn': loss_fn,
            }
        else:
            raise NotImplementedError(f"[{self.cli_args.job_name.upper()}][MODE:{mode}] is not implemented on initializer()")
    
    def train(self, round, epoch, model, train_loader, loss_fn, optimizer, scaler, mode='training'):
        model.train()
        data_time = AverageMeter('Data', ':6.3f')
        batch_time = AverageMeter('Time', ':6.3f')
        bce_meter = AverageMeter('BCE', ':.4f')
        dsc_meter = AverageMeter('Dice', ':.4f')
        loss_meter = AverageMeter('Loss', ':.4f')
        end = time.time()
        for i, (image, label, _, name, affine, label_names) in enumerate(train_loader): # inner_epoch should update local model
            data_time.update(time.time() - end)
            if self.cli_args.use_gpu:
                image, label = image.float().cuda(), label.float().cuda()
            else:
                image, label = image.float(), label.float()
            bsz = image.size(0)

            with autocast((self.cli_args.amp) and (scaler is not None)):
                if self.cli_args.unet_arch == 'unet':
                    preds = model(image)[0]
                else:
                    msg = f'{self.cli_args.unet_arch} Network output not yet guaranteed.'
                    self.logger.error(msg)
                    raise NotImplementedError(msg)

                bce_loss, dsc_loss_by_channels = loss_fn(preds, label)
                avg_dsc_loss = dsc_loss_by_channels.mean()
                loss = bce_loss + avg_dsc_loss

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            if self.cli_args.amp and scaler is not None:
                scaler.scale(loss).backward()
                if self.cli_args.clip_grad:
                    scaler.unscale_(optimizer)  # enable grad clipping
                    nn.utils.clip_grad_norm_(model.parameters(), 10)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.cli_args.clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
            # torch.cuda.synchronize()

            bce_meter.update(bce_loss.item(), bsz)
            dsc_meter.update(avg_dsc_loss.item(), bsz)
            loss_meter.update(loss.item(), bsz)
            batch_time.update(time.time() - end)
            # print(f"train: bloss-{bce_loss.item():.3f}, dloss-{avg_dsc_loss.item():.3f}, bdloss-{loss.item():.3f}")
            self.logger.info(" ".join([
                f"[{self.cli_args.job_name.upper()}][TRN]({((i+1)/len(train_loader)*100):3.0f}%)",
                f"R:{round:02}",
                f"E:{epoch:03}",
                f"D:{i:03}/{len(train_loader):03}",
                f"BCEL:{bce_loss.item():2.3f}",
                f"DSCL:{avg_dsc_loss.item():2.3f}",
                f"BCEL+DSCL:{loss.item():2.3f}",
                f"lr:{optimizer.state_dict()['param_groups'][0]['lr']:0.4f}",
                f"{str(batch_time)}",
            ]))
            end = time.time()
            # middle_slice = image.shape[4] // 2
            # slice_image = image[0, 0, :, :, middle_slice].cpu()  # GPU 사용 시 CPU로 변환
            # max_label = torch.max(label.type(torch.uint8), dim=1, keepdim=True)[0]
            # label_image = max_label[0, 0, :, :, middle_slice].cpu() * 255  # GPU 사용 시 CPU로 변환
            # # 텐서 값을 0-255로 스케일링하고 uint8로 변환
            # slice_image = slice_image - slice_image.min()  # 최소값을 0으로 만듦
            # slice_image = slice_image / slice_image.max() * 255.0  # 최대값을 255로 만듦
            # slice_image = slice_image.type(torch.uint8)  # uint8로 변환
            # # 텐서를 PIL 이미지로 변환
            # pil_image = to_pil(slice_image)
            # pil_label = to_pil(label_image)
            # # 이미지를 저장할 경로 설정 (폴더가 없을 경우 생성)
            # output_dir = 'test'
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # file_name1 = os.path.join(output_dir, f'image_{i}_{name[0]}.jpg')
            # pil_image.save(file_name1)
            # file_name2 = os.path.join(output_dir, f'label_{i}_{name[0]}.jpg')
            # pil_label.save(file_name2)
        return {
            'BCEL_AVG': bce_meter.avg,
            'DSCL_AVG': dsc_meter.avg,
            'TOTAL_AVG': loss_meter.avg,
            # 'lr': optimizer.state_dict()['param_groups'][0]['lr'],
        }

    def infer(self, curr_epoch, model: nn.Module, 
                infer_loader, loss_fn, 
                mode: str, save_infer: bool = True):
        model.eval()
        seg_names = self.cli_args.label_names
        batch_time = AverageMeter('Time', ':6.3f')
        case_metrics_meter = CaseSegMetricsMeter(seg_names)

        save_val_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", f"E{curr_epoch:03}")
        os.makedirs(save_val_path, exist_ok=True)
        # folder_lv3 = f"{mode}_epoch_{epoch:03d}"
        # save_epoch_path = os.path.join("states", folder_dir1, folder_dir2, folder_dir3)
        # if not os.path.exists(save_epoch_path):
        #     os.makedirs(save_epoch_path, exist_ok=True)

        end = time.time()
        with torch.no_grad():
            for i, (image, label, _, name, affine, label_names) in enumerate(infer_loader):
                # if i % 10 != 0:
                #     continue
                label_name = [el[0] for el in label_names]
                if self.cli_args.use_gpu:
                    image, label = image.float().cuda(), label.float().cuda()
                else:
                    image, label = image.float(), label.float()
                bsz = image.size(0)

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
                _, dsc_loss_by_channels = loss_fn(seg_map, label)
                # calc metric

                if self.cli_args.unet_arch == 'unet':
                    seg_map = robust_sigmoid(seg_map)
                else:
                    msg = f"currently model is {self.cli_args.unet_arch}.\n If the model is not unet, it is necessary to check the value range of seg_map before applying any thresholding."
                    self.logger.error(msg)
                    raise NotImplementedError(msg)

                # discrete
                seg_map_th = torch.where(seg_map > 0.5, True, False)

                dsc_loss_by_channels_np = dsc_loss_by_channels.cpu().numpy()
                dice = metrics.dice(seg_map_th, label)
                hd95 = metrics.hd95(seg_map_th, label)
                batch_time.update(time.time() - end)
                case_metrics_meter.update(dsc_loss_by_channels_np, dice, hd95, name, bsz)
                end = time.time()

                for bat_idx, (bat_dice, bat_hd95) in enumerate(zip(dice,hd95)):
                    bat_list=[]
                    bat_list+=[
                        f"[{self.cli_args.job_name.upper()}][{mode.upper()}]({((i+1)/len(infer_loader)*100):3.0f}%)",
                        f"{name[bat_idx]}",
                        f"[DSC:HD95]",
                    ]
                    for (lbl, dsc, hd) in zip(label_name, bat_dice, bat_hd95):
                        bat_list+=[f"{lbl}:{dsc:.2f}:{hd:2.1f}"]
                    self.logger.info(" ".join(bat_list))

                # output seg map
                if save_infer: # and (round == 0):
                    if (curr_epoch == 0) and (mode in ['pre', 'val', 'test']):
                        modality = self.cli_args.input_channel_names
                        scale = 255
                        save_img_nifti(image, scale, name, mode[:4], 'img',
                                       affine, modality, save_val_path)
                        label_map = self.cli_args.label_index
                        save_seg_nifti(label, name, mode[:4], 'labels',
                                       affine, label_map, save_val_path)

                    # if (i % 10 == 0):
                    # seg_labels = label_names
                    scale = 100
                    save_img_nifti(seg_map, scale, name, mode[:4], 'prob',
                                affine, label_name, save_val_path)
                    label_map = self.cli_args.label_index
                    save_seg_nifti(seg_map_th, name, mode[:4], 'pred',
                                affine, label_map, save_val_path)
        return {
            'DSCL_AVG': np.mean([v for k, v in case_metrics_meter.mean().items() if 'DSCL' in k]),
            'DICE_AVG': np.mean([v for k, v in case_metrics_meter.mean().items() if 'DICE' in k]),
            'HD95_AVG': np.mean([v for k, v in case_metrics_meter.mean().items() if 'HD95' in k]),
            **case_metrics_meter.mean(),
        }

    def run_train(self):
        time_in_total = time.time()
        train_val_dict = load_subjects_list(
            self.cli_args.rounds,
            self.cli_args.round, 
            self.cli_args.cases_split, 
            self.cli_args.inst_ids, 
            TrainOrVal=['train','val'], 
            partition_by_round=(self.cli_args.rounds > 0),
            mode='train'
        )
        
        _, self.cli_args.weight_path = self.initModel(self.cli_args.weight_path, mode='INIT')

        train_setup = self.initializer(train_val_dict, mode='TRN')

        from_epoch = self.cli_args.epochs * (self.cli_args.round)
        to_epoch = self.cli_args.epochs * (self.cli_args.round + 1)

        # TODO: val mode 체크 필요
        # if train_val_dict['train'].__len__() == 0:
        #     self.run_infer(infer_mode='val')
        #     return
        if train_val_dict['train'].__len__() == 0:
            infer_mode = 'val'
            infer_metrics = self.infer(
                from_epoch, 
                train_setup['model'], 
                train_setup['val_loader'], 
                train_setup['loss_fn'], 
                mode=infer_mode,
                save_infer=self.cli_args.save_infer
            )
            state = {
                'args': self.cli_args,
                f'{infer_mode}_metrics': infer_metrics,
                'time': time.time() - time_in_total,
            }
            save_model_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", "models")
            os.makedirs(save_model_path, exist_ok=True)
            torch.save(state, os.path.join(save_model_path, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}.pth"))
            return # Finish process because there is no training data

        # Pre-Validation
        pre_metrics = self.infer(
            from_epoch, 
            train_setup['model'], 
            train_setup['val_loader'], 
            train_setup['loss_fn'], 
            mode='pre',
            save_infer=self.cli_args.save_infer
        )

        for i in range(self.cli_args.round):
            train_setup['scheduler'].step()
        train_tb_dict = {}
        for epoch in range(from_epoch, to_epoch):
            train_tb_dict[epoch] = self.train(
                self.cli_args.round, epoch, 
                train_setup['model'], 
                train_setup['train_loader'], 
                train_setup['loss_fn'], 
                train_setup['optimizer'], 
                # train_setup['scheduler'],
                train_setup['scaler'], 
                mode='training'
            )
            # if train_setup['scheduler'] is not None:
            #     train_setup['scheduler'].step()

        # Post-Validation (every 10 epoch recordings for central learning)
            # if (epoch > 0) and ((epoch % 10 == 0) or epoch == (to_epoch - 1)): 
        post_metrics = self.infer(
            to_epoch-1, 
            train_setup['model'], 
            train_setup['val_loader'], 
            train_setup['loss_fn'], 
            mode='post',
            save_infer=self.cli_args.save_infer
        )

        # 학습, 평가 및 테스트 후 모델을 CPU로 이동
        if isinstance(train_setup['model'], nn.DataParallel):
            train_setup['model'] = train_setup['model'].module  # Extract original model from DataParallel wrapper
        train_setup['model'] = train_setup['model'].to('cpu')  # Move model back to CPU

        Pre_DSCL = pre_metrics['DSCL_AVG']
        Post_DSCL = post_metrics['DSCL_AVG']
        Post_DSCL = np.min([Pre_DSCL, Post_DSCL])
        state = {
            'model': train_setup['model'].state_dict(), 
            'args': self.cli_args,
            'train_tb_dict': train_tb_dict,
            'pre_metrics': pre_metrics,
            'post_metrics': post_metrics,
            'P': train_setup['train_loader'].dataset.__len__(),
            'I': (Pre_DSCL + Post_DSCL)/2,
            'D': (Pre_DSCL - Post_DSCL),
            'time': time.time() - time_in_total,
        }
        save_model_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", "models")
        os.makedirs(save_model_path, exist_ok=True)
        torch.save(state, os.path.join(save_model_path, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}_last.pth"))
        return

    def run_infer(self, infer_mode='test'):
        time_in_total = time.time()
        infer_dict = load_subjects_list(
            self.cli_args.rounds, 
            self.cli_args.round, 
            self.cli_args.cases_split, 
            self.cli_args.inst_ids, 
            TrainOrVal=[infer_mode], 
            partition_by_round=(self.cli_args.rounds > 0),
            mode=infer_mode
        )

        assert self.cli_args.weight_path is not None, f"run_infer must have weight_path for model to infer."
        # _, self.cli_args.weight_path = self.initModel(self.cli_args.weight_path)

        infer_setup = self.initializer(infer_dict, mode=infer_mode)

        val_epoch = self.cli_args.epochs * (self.cli_args.round)
        
        # from_epoch = self.cli_args.epochs * (self.cli_args.round)
        # to_epoch = self.cli_args.epochs * (self.cli_args.round + 1)
        # Validation
        infer_metrics = self.infer(
            val_epoch, 
            infer_setup['model'], 
            infer_setup['infer_loader'], 
            infer_setup['loss_fn'], 
            mode=infer_mode,
            save_infer=self.cli_args.save_infer
        )

        state = {
            'args': self.cli_args,
            f'{infer_mode}_metrics': infer_metrics,
            'time': time.time() - time_in_total,
        }
        save_model_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", "models")
        os.makedirs(save_model_path, exist_ok=True)
        torch.save(state, os.path.join(save_model_path, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}.pth"))
        return