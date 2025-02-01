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
from dsets import get_dataset, get_base_transform, get_forward_transform, get_aug_transform, custom_collate
from utils.optim import get_optimizer
from utils.loss import SoftDiceBCEWithLogitsLoss, robust_sigmoid
from utils.scheduler import get_scheduler
# from PIL import Image
# import torchvision.transforms as tf
# ToPILImage 변환timestamp기 초기화
# to_pil = tf.ToPILImage()
MASKS=['seg','ref','sub','label']
class Unet3DApp:
    def __init__(self, sys_argv=None):
        # 기본 로거 설정 (기본값 사용)
        args = parse_args(sys_argv)
        self.cli_args = args
        print(f"SEED: {self.cli_args.seed}")
        seed_everything(self.cli_args.seed)
        self.cli_args.weight_path = None if self.cli_args.weight_path == "None" else self.cli_args.weight_path
        self.cli_args.amp = True if self.cli_args.amp else False
        self.job_name = self.cli_args.job_name if self.cli_args.job_name else time.strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cpu")
        log_filename = f"{self.cli_args.job_name}_R{self.cli_args.rounds:02}r{self.cli_args.round:02}.log"
        job_prefix = self.job_name.split('_')[0]
        self.logger = initialization_logger(self.cli_args, job_prefix, log_filename)
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

    def initTrainDl(self, case_names:list, mode='training', index_filter=None):
        base_transform = get_base_transform(self.cli_args)
        aug_transform = get_aug_transform(self.cli_args)
        train_transforms = transforms.Compose(base_transform + aug_transform)
        train_dataset = get_dataset(self.cli_args, case_names, 
                                    train_transforms,
                                    mode=mode,
                                    label_names=self.cli_args.label_names,
                                    custom_min_len=self.cli_args.min_dlen,
                                    index_filter=index_filter)  # 필터 전달

        train_loader = DataLoader(
            train_dataset,
            # batch_size=self.cli_args.multi_batch_size,
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

    def initTestDl(self, case_names:list, mode:str):
        forward_transform  = get_forward_transform(self.cli_args)
        aug_transform = [transforms.EnsureTyped(keys=["image"])]
        forward_transform = transforms.Compose(forward_transform + aug_transform)
        forward_dataset = get_dataset(self.cli_args, case_names, forward_transform, 
                                    mode=mode, 
                                    label_names=self.cli_args.label_names,
                                    custom_min_len=1,
                                    custom_max_len=self.cli_args.max_dlen)
        test_loader = DataLoader(
            forward_dataset,
            batch_size=self.cli_args.multi_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)
        return forward_dataset, test_loader

    def initializer(self, subjects_dict, mode='train'):
        if mode in ['train', 'TRN']:
            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with train_loader and val_loader...")
            train_cases = natsort.natsorted(subjects_dict['train'])
            random.shuffle(train_cases)
            # train_dataset, train_loader = self.initTrainDl(train_cases)
            train_dataset, train_loader = self.initTrainDl(train_cases, index_filter=lambda x: x < self.cli_args.data_percentage)
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
        elif mode in ['val']:
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
        elif mode in ['test']:
            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with infer_loader...")
            test_cases = natsort.natsorted(subjects_dict['infer'])
            test_dataset, test_loader = self.initTestDl(test_cases, mode)

            self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with model...")
            model, _ = self.initModel(self.cli_args.weight_path, mode=mode.upper())
            model = self.setup_gpu(model, mode=mode.upper())
            # loss_fn = SoftDiceBCEWithLogitsLoss(channel_weights=None).to(self.device)

            return {
                'test_dataset': test_dataset,
                'test_loader': test_loader,
                'model': model,
                # 'loss_fn': loss_fn,
            }
        # elif mode in ['quant']:
        #     self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with infer_loader...")
        #     test_cases = natsort.natsorted(subjects_dict['infer'])
        #     test_dataset, test_loader = self.initTestDl(test_cases, mode)

        #     self.logger.info(f"[{self.cli_args.job_name.upper()}][{mode.upper()}] Processing with model...")
        #     # model, _ = self.initModel(self.cli_args.weight_path, mode=mode.upper())
        #     # model = self.setup_gpu(model, mode=mode.upper())
        #     # loss_fn = SoftDiceBCEWithLogitsLoss(channel_weights=None).to(self.device)

        #     return {
        #         'test_dataset': test_dataset,
        #         'test_loader': test_loader,
        #         # 'model': model,
        #         # 'loss_fn': loss_fn,
        #     }
        else:
            raise NotImplementedError(f"[{self.cli_args.job_name.upper()}][MODE:{mode}] is not implemented on initializer()")
    
    def train(self, round, rounds, from_epoch, epoch, to_epoch, model, train_loader, loss_fn, optimizer, scaler, mode='training'):
        model.train()
        data_time = AverageMeter('Data', ':6.3f')
        batch_time = AverageMeter('Time', ':6.3f')
        bce_meter = AverageMeter('BCE', ':.4f')
        dsc_meter = AverageMeter('Dice', ':.4f')
        loss_meter = AverageMeter('Loss', ':.4f')
        end = time.time()
        # for i, (image, label, _, name, affine, label_names) in enumerate(infer_loader):
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
            curr_data = len(train_loader) * (epoch - from_epoch) + (i+1)
            total_data = (to_epoch - from_epoch) * len(train_loader)
            self.logger.info(" ".join([
                f"[{self.cli_args.job_name.upper()}][TRN]({(curr_data/total_data*100):3.0f}%)",
                f"R:{round:02}/{rounds:02}",
                f"E:{epoch:03}/{to_epoch:03}",
                f"D:{i:03}/{len(train_loader):03}",
                f"BCEL:{bce_loss.item():2.3f}",
                f"DSCL:{avg_dsc_loss.item():2.3f}",
                f"BCEL+DSCL:{loss.item():2.3f}",
                f"lr:{optimizer.state_dict()['param_groups'][0]['lr']:0.4f}",
                f"N:{name}",
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
                pvdc = metrics.pvdc(seg_map_th, label)
                batch_time.update(time.time() - end)
                case_metrics_meter.update(dice, hd95, pvdc, name, bsz, dscloss=dsc_loss_by_channels_np)
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
                    # rounds == 1 인 경우는 대부분 없음 (infer 의 경우에 한함)
                    # if mode == 'test':
                    #     modality = self.cli_args.input_channel_names
                    #     scale = 255
                    #     # 여기서 최상위 폴더 하나로 잡고, 하위에 pid폴더를 생성하여 
                    #     # 그 pid폴더 내부에 각각 brain.nii.gz, striatum_orig.nii.gz
                    #     # 생성하면 됩니다.
                    #     label_map = self.cli_args.label_index
                    #     img_name = self.cli_args.img_name
                    #     seg_name = self.cli_args.seg_name
                    #     save_img_nifti(image, scale, [img_name]*len(name), affine, modality,    save_val_path, name)
                    #     save_seg_nifti(seg_map_th,   [seg_name]*len(name), affine, label_map,   save_val_path, name)
                    # else:
                    if (curr_epoch == 0):
                        modality = self.cli_args.input_channel_names
                        # scale = 255
                        label_map = self.cli_args.label_index
                        save_img_nifti(image, f"{mode[:4]}_", "_img", affine, modality,   save_val_path, name) #[f"{mode[:4]}_img"]*len(name))
                        save_seg_nifti(label, f"{mode[:4]}_", "_lbl", affine, label_map,  save_val_path, name) #[f"{mode[:4]}_lbl"]*len(name))

                        # save_img_nifti(image, scale, [img_name]*len(name), affine, modality,    save_val_path, name)
                        # save_seg_nifti(seg_map_th,   [seg_name]*len(name), affine, label_map,   save_val_path, name)

                    # scale = 100
                    label_map = self.cli_args.label_index
                    save_img_nifti(seg_map,    f"{mode[:4]}_", "_prb", affine, label_name, save_val_path, name) #[f"{mode[:4]}_prb"]*len(name))
                    save_seg_nifti(seg_map_th, f"{mode[:4]}_", "_prd", affine, label_map,  save_val_path, name) #[f"{mode[:4]}_prd"]*len(name))
                    # else:
                    #     raise f"mode must be test or pre"
                    # if (i % 10 == 0):
                    # seg_labels = label_names
                                
        # output case metric csv
        # save_epoch_path = os.path.join(save_val_path, 'case_metric.csv')
        case_metrics_meter.output(save_val_path)
        return {
            'DSCL_AVG': np.mean([v for k, v in case_metrics_meter.mean().items() if 'DSCL' in k]),
            'DICE_AVG': np.mean([v for k, v in case_metrics_meter.mean().items() if 'DICE' in k]),
            'HD95_AVG': np.mean([v for k, v in case_metrics_meter.mean().items() if 'HD95' in k]),
            'PVDC_AVG': np.mean([v for k, v in case_metrics_meter.mean().items() if 'PVDC' in k]),
            **case_metrics_meter.mean(),
        }

    def run_train(self):
        time_in_total = time.time()
        # select_all_inst = True if (self.cli_args.rounds == 0) else False
        # partition_ids = 'Partition_ID' if select_all_inst else f"R{self.cli_args.round}"
        train_val_dict = load_subjects_list(
            self.cli_args.rounds,
            self.cli_args.round, 
            self.cli_args.cases_split, 
            self.cli_args.inst_ids, 
            TrainOrVal=['train','val'], 
            mode='train'
        )
        
        _, self.cli_args.weight_path = self.initModel(self.cli_args.weight_path, mode='INIT')

        train_setup = self.initializer(train_val_dict, mode='TRN')

        from_epoch  = self.cli_args.epoch
        to_epoch    = self.cli_args.epoch + self.cli_args.epochs

        # Pre-Validation
        pre_metrics = self.infer(
            from_epoch, 
            train_setup['model'], 
            train_setup['val_loader'], 
            train_setup['loss_fn'], 
            mode='pre',
            save_infer=self.cli_args.save_infer
        )
        state = {
            'args': self.cli_args,
            'pre_metrics': pre_metrics,
            'time': time.time() - time_in_total,
        }
        save_model_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", "models")
        os.makedirs(save_model_path, exist_ok=True)
        torch.save(state, os.path.join(save_model_path, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}_prev.pth"))
        # print(f"****IMPORTANT: {train_setup['train_loader'].__len__()}")
        # if train_setup['train_loader'].__len__() == 0:
        if self.cli_args.data_percentage == 0:
            self.logger.info(f"[{self.cli_args.job_name.upper()}][TRN] exit before train as train data not exists at job_name-{self.job_name}...")
            return # Finish process because there is no training data
        
        train_tb_dict = {}
        MIN_DSCL_AVG = 10 # pre_metrics['DSCL_AVG']

        if train_setup['scheduler'] is not None:
            for i in range(from_epoch):
                train_setup['scheduler'].step()

        for i, epoch in enumerate(range(from_epoch, to_epoch)):
            train_tb_dict[epoch] = self.train(
                self.cli_args.round, self.cli_args.rounds, 
                from_epoch, epoch, to_epoch,
                train_setup['model'], 
                train_setup['train_loader'], 
                train_setup['loss_fn'], 
                train_setup['optimizer'], 
                train_setup['scaler'], 
                mode='training'
            )
            if train_setup['scheduler'] is not None:
                train_setup['scheduler'].step()
                
            if (i % self.cli_args.eval_freq == 0) and (self.cli_args.rounds < 5):
                infer_mode = 'val'
                val_metrics = self.infer(
                    epoch, 
                    train_setup['model'], 
                    train_setup['val_loader'], 
                    train_setup['loss_fn'], 
                    mode=infer_mode,
                    save_infer=self.cli_args.save_infer
                )
                
                MIN_DSCL_AVG, IS_UPDATED = (val_metrics['DSCL_AVG'], True) if val_metrics['DSCL_AVG'] < MIN_DSCL_AVG else (MIN_DSCL_AVG, False)
                if IS_UPDATED:
                    state = {
                        'model': train_setup['model'].state_dict(),
                        'args': self.cli_args,
                        f'{infer_mode}_metrics': val_metrics,
                        'time': time.time() - time_in_total,
                    }
                    save_model_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", "models")
                    os.makedirs(save_model_path, exist_ok=True)
                    torch.save(state, os.path.join(save_model_path, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}_best.pth"))

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

        Pre_LOSS = pre_metrics['PVDC_AVG'] # PVDC_AVG vs. DSCL_AVG
        Post_LOSS = post_metrics['PVDC_AVG'] # PVDC_AVG vs. DSCL_AVG
        Post_LOSS = Post_LOSS if Post_LOSS < Pre_LOSS else Pre_LOSS # np.min([Pre_LOSS, Post_LOSS])
        state = {
            'model': train_setup['model'].state_dict(), 
            'args': self.cli_args,
            'train_tb_dict': train_tb_dict,
            'pre_metrics': pre_metrics,
            'post_metrics': post_metrics,
            'P': train_setup['train_loader'].dataset.__len__(),
            'I': (Pre_LOSS + Post_LOSS)/2,
            'D': (Pre_LOSS - Post_LOSS),
            'time': time.time() - time_in_total,
        }
        save_model_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", "models")
        os.makedirs(save_model_path, exist_ok=True)
        torch.save(state, os.path.join(save_model_path, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}_last.pth"))
        return

    def forward(self, inst_root, model: nn.Module, test_loader, mode: str, save_infer: bool = True):
        model.eval()
        seg_names = self.cli_args.label_names

        save_val_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", inst_root)
        os.makedirs(save_val_path, exist_ok=True)

        with torch.no_grad():
            for i, (image, _, _, name, affine, label_names) in enumerate(test_loader):
                # label_name = [el[0] for el in label_names]
                if self.cli_args.use_gpu:
                    image = image.float().cuda()
                else:
                    image = image.float()

                seg_map = sliding_window_inference(
                    inputs=image,
                    predictor=model,
                    roi_size=self.cli_args.patch_size,
                    sw_batch_size=self.cli_args.sw_batch_size,
                    overlap=self.cli_args.patch_overlap,
                    mode=self.cli_args.sliding_window_mode
                )

                if self.cli_args.unet_arch == 'unet':
                    seg_map = robust_sigmoid(seg_map)
                else:
                    msg = f"currently model is {self.cli_args.unet_arch}.\n If the model is not unet, it is necessary to check the value range of seg_map before applying any thresholding."
                    self.logger.error(msg)
                    raise NotImplementedError(msg)

                # discrete
                seg_map_th = torch.where(seg_map > 0.5, True, False)
##### logger name only when forward testset
                for bat_idx, _ in enumerate(image):
                    bat_list=[]
                    bat_list+=[
                        f"[{self.cli_args.job_name.upper()}][{mode.upper()}]({((i+1)/len(test_loader)*100):3.0f}%)",
                        f"{name[bat_idx]}",
                    ]
                    self.logger.info(" ".join(bat_list))
#####
                # output seg map
                if save_infer:
                    modality = self.cli_args.input_channel_names
                    # scale = 255
                    label_map = self.cli_args.label_index
                    # img_name = self.cli_args.img_name
                    # seg_name = self.cli_args.seg_name
                    save_img_nifti(image,      "", "",                        affine, modality,    save_val_path, name)
                    save_seg_nifti(seg_map_th, "", self.cli_args.seg_postfix, affine, label_map,   save_val_path, name)
        return

    def run_forward(self, test_mode='test'):
        test_dict = load_subjects_list(
            self.cli_args.rounds, 
            self.cli_args.round, 
            self.cli_args.cases_split, 
            self.cli_args.inst_ids, 
            TrainOrVal=[test_mode], 
            mode=test_mode
        )
        if self.cli_args.weight_path == None:
            _, self.cli_args.weight_path = self.initModel(self.cli_args.weight_path, mode='INIT')
        # assert self.cli_args.weight_path is not None, f"run_infer must have weight_path for model to infer."
        test_setup = self.initializer(test_dict, mode=test_mode)
        inst_root = self.cli_args.inst_root
        self.forward(
            inst_root, 
            test_setup['model'], 
            test_setup['test_loader'], 
            mode=test_mode,
            save_infer=self.cli_args.save_infer
        )
        return



#     def compare(self, inst_root, test_loader, mode: str, save_infer: bool = True):
#         # model.eval()
#         seg_names = self.cli_args.label_names

#         save_val_path = os.path.join("states", self.job_name, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}", inst_root)
#         os.makedirs(save_val_path, exist_ok=True)

#         with torch.no_grad():
#             for i, (image, _, _, name, affine, label_names) in enumerate(test_loader):
#                 # label_name = [el[0] for el in label_names]
#                 if self.cli_args.use_gpu:
#                     image = image.float().cuda()
#                 else:
#                     image = image.float()

#                 # seg_map = 
#                 # sliding_window_inference(
#                 #     inputs=image,
#                 #     predictor=model,
#                 #     roi_size=self.cli_args.patch_size,
#                 #     sw_batch_size=self.cli_args.sw_batch_size,
#                 #     overlap=self.cli_args.patch_overlap,
#                 #     mode=self.cli_args.sliding_window_mode
#                 # )

#                 # if self.cli_args.unet_arch == 'unet':
#                 #     seg_map = robust_sigmoid(seg_map)
#                 # else:
#                 #     msg = f"currently model is {self.cli_args.unet_arch}.\n If the model is not unet, it is necessary to check the value range of seg_map before applying any thresholding."
#                 #     self.logger.error(msg)
#                 #     raise NotImplementedError(msg)

#                 # discrete
#                 seg_map_th = image
# ##### logger name only when forward testset
#                 for bat_idx, _ in enumerate(image):
#                     bat_list=[]
#                     bat_list+=[
#                         f"[{self.cli_args.job_name.upper()}][{mode.upper()}]({((i+1)/len(test_loader)*100):3.0f}%)",
#                         f"{name[bat_idx]}",
#                     ]
#                     self.logger.info(" ".join(bat_list))
# #####
#                 # output seg map
#                 if save_infer:
#                     modality = self.cli_args.input_channel_names
#                     # scale = 255
#                     label_map = self.cli_args.label_index
#                     # img_name = self.cli_args.img_name
#                     # seg_name = self.cli_args.seg_name
#                     save_img_nifti(image,      "", "",                        affine, modality,    save_val_path, name)
#                     save_seg_nifti(seg_map_th, "", self.cli_args.seg_postfix, affine, label_map,   save_val_path, name)
#         return


    # def run_quantification(self, quant_mode='quant'):
    #     quant_dict = load_subjects_list(
    #         self.cli_args.rounds, 
    #         self.cli_args.round, 
    #         self.cli_args.cases_split, 
    #         self.cli_args.inst_ids, 
    #         TrainOrVal=['test'],
    #         mode=quant_mode
    #     )
    #     # if self.cli_args.weight_path == None:
    #     #     _, self.cli_args.weight_path = self.initModel(self.cli_args.weight_path, mode='INIT')
    #     # assert self.cli_args.weight_path is not None, f"run_infer must have weight_path for model to infer."
    #     quant_setup = self.initializer(quant_dict, mode=quant_mode)
    #     inst_root = self.cli_args.inst_root
    #     self.compare(
    #         inst_root, 
    #         # quant_setup['model'], 
    #         quant_setup['test_loader'],     # Manual segmented
    #         # quant_setup['test_loader'],   # Model segmented
    #         mode=quant_mode,
    #         save_infer=self.cli_args.save_infer
    #     )
    #     return