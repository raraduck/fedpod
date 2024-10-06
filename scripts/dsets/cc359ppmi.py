import os
from os.path import join
import torch
import numpy as np
import monai.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from dsets.dataset_utils import nib_load
import nibabel as nib
import glob

class CC359PPMIDataset(Dataset):
    def __init__(self, args, data_root: str, inst_root: str, mode: str, case_names: list = [], input_channel_names: list = [], label_names: list = [], transforms=None, custom_lower_bound=1, custom_upper_bound=99999):
        super(CC359PPMIDataset, self).__init__()

        assert mode.lower() in ['train', 'training', 
                                'infer', 'val', 'validation', 
                                'test', 'testing'], f'Unknown mode: {mode}'
        self.args = args
        self.mode = mode.lower()
        self.data_root = data_root
        self.inst_root = inst_root
        self.input_channel_names = input_channel_names
        self.case_names = case_names
        self.label_names = label_names
        self.custom_len = min(max(custom_lower_bound, len(case_names)), custom_upper_bound)
        # self.custom_max_len = len(case_names) if custom_max_len <= 1 else custom_max_len
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple:
        index = index % len(self.case_names)
        name = self.case_names[index]
        # base_dir = join(self.data_root, 'training', name, name)  # seg/data/brats21/BraTS2021_00000/BraTS2021_00000
        base_dir_list = glob.glob(join(self.data_root, self.inst_root, name))  # seg/data/brats21/BraTS2021_00000/BraTS2021_00000
        assert base_dir_list.__len__() == 1
        base_dir = base_dir_list[0]

        channels_dict = {}
        if 't1' in self.input_channel_names:
            t1, affine = nib_load(join(base_dir, 'brain.nii.gz'))
            channels_dict['t1'] = np.array(t1, dtype='float32')

        if 't1ce' in self.input_channel_names:
            t1ce, _ = nib_load(join(base_dir, 'orig.nii.gz'))
            channels_dict['t1ce'] = np.array(t1ce, dtype='float32')

        if 't2' in self.input_channel_names:
            t2, _ = nib_load(join(base_dir, 'brain.nii.gz'))
            channels_dict['t2'] = np.array(t2, dtype='float32')

        if 'flair' in self.input_channel_names:
            flair, _ = nib_load(join(base_dir, 'brain.nii.gz'))
            channels_dict['flair'] = np.array(flair, dtype='float32')

        if 'pet' in self.input_channel_names:
            pet, _ = nib_load(join(base_dir, 'pet.nii.gz'))
            channels_dict['pet'] = np.array(pet, dtype='float32')

        if 'striatum' in self.input_channel_names:
            striatum, _ = nib_load(join(base_dir, 'striatum_orig.nii.gz'))
            _mask = np.array(striatum, dtype='float32')
            # _mask = np.where(_mask != 0, 100, 0)
            channels_dict['striatum'] = _mask

        mask = np.array(nib_load(join(base_dir, 'striatum_orig.nii.gz'))[0], dtype='uint16')  # ground truth
        channels_dict['label'] = mask
        # _t2, _ = nib_load(join(base_dir, 'brain.nii.gz'))
        # t2 = np.array(_t2, dtype='float32')
        # t1 = np.array(nib_load(join(base_dir, 'brain.nii.gz')), dtype='float32')
        # flair = np.array(nib_load(base_dir + '_flair.nii.gz'), dtype='float32')
        # t1 = np.array(nib_load(base_dir + '-T1.nii.gz'), dtype='float32')
        # data, affine = nib_load(join(base_dir, 'brain.nii.gz'))
        # t1ce = np.array(nib_load(base_dir + '_t1ce.nii.gz'), dtype='float32')
        # t2 = np.array(nib_load(base_dir + '_t2.nii.gz'), dtype='float32')
        # mask = np.array(nib_load(base_dir + '-label.nii.gz'), dtype='uint8')  # ground truth

        item = self.transforms(channels_dict)

        if self.mode.lower() in ['train', 'training']:  # train
            # Assume each item is a dictionary containing multiple samples
            samples = []
            for el in item:
                image_shape = el['image'][0].shape
                el['image'].affine[:3, 3] = torch.tensor(-np.array(image_shape) / 2 * np.diag(el['image'].affine)[:3])
                samples.append((el['image'], el['label'], index, name, el['image'].affine, self.label_names))
            return samples

            # el = item[0]  # [0] for RandCropByPosNegLabeld
            # # 첫번째 샘플만 뽑아서 작업하는중
            # image_shape = el['image'][0].shape  # 이미지의 복셀 크기 가져오기
            # # 이미지 중앙을 원점으로 설정하기
            # el['image'].affine[:3, 3] = torch.tensor(-np.array(image_shape) / 2 * np.diag(el['image'].affine)[:3])
            # return el['image'], el['label'], index, name, el['image'].affine, self.label_names # item['image_meta_dict']['affine']
        else:
            # 이미지 중앙을 원점으로 설정하기
            # affine[:3, 3] = torch.tensor(-np.array(image_shape) / 2 * np.diag(item['image'].affine)[:3])
            # affine[:3, 3] = torch.tensor(-np.array(image_shape) / 2)
            if self.args.zoom or (item['image'][0].shape != t1[0].shape):
                # image_shape = item['image'][0].shape
                # temp_affine = affine # item['image'].affine[:3, 3]
                # affine[:3, 3] = torch.tensor(-np.array(image_shape) / 2 * np.diag(item['image'].affine)[:3])
                temp_affine = affine
            else:
                temp_affine = affine # item['image'].affine[:3, 3]
            return item['image'], item['label'], index, name, temp_affine, self.label_names # item['image_meta_dict']['affine']



    def __len__(self):
        # return len(self.case_names)
        return self.custom_len

    # def count_by_condition(self, condition_func):
    #     count = 0
    #     for data, label in zip(self.data, self.labels):
    #         if condition_func(data, label):
    #             count += 1
    #     return count