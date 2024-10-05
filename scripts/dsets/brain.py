import os
from os.path import join
import torch
import numpy as np
import monai.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from dsets.dataset_utils import nib_load
import nibabel as nib


class BrainDataset(Dataset):
    def __init__(self, args, data_root: str, mode: str, case_names: list = [], input_channel_names: list = [], label_names: list = [], transforms=None, custom_lower_bound=1, custom_upper_bound=99999):
        super(BrainDataset, self).__init__()

        assert mode.lower() in ['support', 'query', 'train', 'infer', 'training', 'validation', 'test', 'testing', 'val',
                                'meta_train', 'meta_val', 'meta_test', 'testing2'], f'Unknown mode: {mode}'
        self.args = args
        self.mode = mode.lower()
        self.data_root = data_root
        self.input_channel_names = input_channel_names
        self.case_names = case_names
        self.label_names = label_names
        self.custom_len = min(max(custom_lower_bound, len(case_names)), custom_upper_bound)
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple:
        # Return a default item if case_names is empty
        if len(self.case_names) == 0:
            raise ValueError("The case_names list is empty. Please provide valid case names.")

        index = index % len(self.case_names)
        name = self.case_names[index]
        base_dir = join(self.data_root, 'center', name)  # seg/data/brats21/BraTS2021_00000/BraTS2021_00000

        channels_dict = {}
        if 't1' in self.input_channel_names:
            t1, affine = nib_load(join(base_dir, f'{name}_t1.nii.gz'))
            channels_dict['t1'] = np.array(t1, dtype='float32')

        if 't1ce' in self.input_channel_names:
            t1ce, _ = nib_load(join(base_dir, f'{name}_t1ce.nii.gz'))
            channels_dict['t1ce'] = np.array(t1ce, dtype='float32')

        if 't2' in self.input_channel_names:
            t2, _ = nib_load(join(base_dir, f'{name}_t2.nii.gz'))
            channels_dict['t2'] = np.array(t2, dtype='float32')

        if 'flair' in self.input_channel_names:
            flair, _ = nib_load(join(base_dir, f'{name}_flair.nii.gz'))
            channels_dict['flair'] = np.array(flair, dtype='float32')

        if self.mode == 'testing2':
            item = self.transforms(channels_dict)
            affine[0][0] = -1.0
            affine[1][1] = -1.0
            affine[2][2] = 1.0
            affine[0][3] = 0.0
            affine[1][3] = 239.0
            affine[2][3] = 0.0

            return item['image'], index, name, affine, self.label_names # item['image_meta_dict']['affine']
        
        mask = np.array(nib_load(join(base_dir, f'{name}_seg.nii.gz'))[0], dtype='uint8')  # ground truth
        channels_dict['label'] = mask
        item = self.transforms(channels_dict)

        if self.mode.lower() in ['support', 'query', 'train', 'training', 'meta_train', 'meta_val']:  # train
            # Assume each item is a dictionary containing multiple samples
            samples = []
            for el in item:
                # image_shape = el['image'][0].shape
                # el['image'].affine[:3, 3] = torch.tensor(-np.array(image_shape) / 2 * np.diag(el['image'].affine)[:3])
                samples.append((el['image'], el['label'], index, name, el['image'].affine, self.label_names))
            return samples

        else:
            # image_shape = item['image'][0].shape  # 이미지의 복셀 크기 가져오기
            # # 이미지 중앙을 원점으로 설정하기
            # item['image'].affine[:3, 3] = torch.tensor(-np.array(image_shape) / 2 * np.diag(item['image'].affine)[:3])
            return item['image'], item['label'], index, name, affine, self.label_names # item['image_meta_dict']['affine']

    def __len__(self):
        return self.custom_len