import os
import numpy as np
import nibabel as nib
import random
from os.path import join
from monai.transforms.transform import MapTransform
import monai.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
MASKS=['seg','ref','sub','label']
def nib_load(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError(file_name)
    proxy = nib.load(file_name)
    # proxy = nib.as_closest_canonical(proxy)
    data = proxy.get_fdata()
    affine = proxy.affine
    # image_shape = canonical_nii.header.get_data_shape()  # 이미지의 복셀 크기 가져오기
    # 이미지 중앙을 원점으로 설정하기
    # affine[:3, 3] = -np.array(image_shape) / 2 * np.diag(affine)[:3]
    # affine[:3, 3] = [0, 0, 0]
    proxy.uncache()
    return np.expand_dims(data, axis=0), affine


class RobustZScoreNormalization(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key] > 0

            lower = np.percentile(d[key][mask], 0.2)
            upper = np.percentile(d[key][mask], 99.8)

            d[key][mask & (d[key] < lower)] = float(lower)
            d[key][mask & (d[key] > upper)] = float(upper)

            y = d[key][mask]
            d[key] -= y.mean()
            d[key] /= y.std()

        return d


class ConvertToMultiChannel(transforms.MapTransform):
    def __init__(self, keys, labels):
        super().__init__(keys)
        self.labels = labels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label_map = np.squeeze(d[key], axis=0)
            channels = []
            for label_group in self.labels:
                # Initialize channel with False
                channel = np.zeros_like(label_map, dtype=np.bool_)
                # Logical OR for each label in the group
                for label in label_group:
                    channel = np.logical_or(channel, label_map == label)
                channels.append(channel)
            d[key] = np.stack(channels, axis=0)  # Stack along a new channel dimension
        return d


def calculate_new_pixdim(args, data):
    # crop된 이미지의 형태를 얻습니다.
    cropped_shape = data[args.input_channel_names[0]][0].shape  # 예: (120, 120, 120)
    # target_shape = (256, 256, 256)  # 원본 크기를 예로 들었습니다.
    # ratio = [c / t for c, t in zip(cropped_shape, target_shape)]
    # data['new_pixdim'] = ratio  # 계산된 pixdim을 데이터 딕셔너리에 저장
    data['cropped_shape'] = cropped_shape
    return data

def apply_spacing_transform(args, data, selected_keys):
    max_size = max(data['cropped_shape'])
    ratio = max_size/args.resize * 1 # args.pixdim
    # if 'label' in selected_keys:
    #     interpolations = [*(['bilinear'] * args.input_channels), 'nearest']
    # else:
    #     interpolations = [*(['bilinear'] * len(selected_keys))]
    
    interpolations = []
    for key in selected_keys:
        if key in MASKS:  # 특정 키에 대해 nearest interpolation
            interpolations.append('nearest')
        else:
            interpolations.append('bilinear')  # 나머지 키는 bilinear interpolation

    # selected_keys = [*args.input_channel_names, 'label']
    # 정의된 함수에서 Spacingd 변환 적용
    transform = transforms.Spacingd(
        keys=selected_keys,
        pixdim=(ratio, ratio, ratio),  # new_pixdim을 이미 계산한 값으로 가정
        mode=interpolations,
        # recompute_affine=True
    )
    return transform(data)


# def apply_spacing_transform2(data, ratio):
#     src_pixdim = max(data['new_pixdim'])
#     new_pixdim = src_pixdim * ratio
#     # 정의된 함수에서 Spacingd 변환 적용
#     transform = transforms.Spacingd(
#         keys=['t1', 'label'],
#         pixdim=(new_pixdim, new_pixdim, new_pixdim),  # new_pixdim을 이미 계산한 값으로 가정
#         mode=['bilinear', 'nearest'],
#         # recompute_affine=True
#     )
#     return transform(data)

def get_base_transform(args, label_groups=None):
    if label_groups == None:
        label_groups = args.label_groups
    selected_keys = [*args.input_channel_names, 'label']
    base_transform_1 = [
        transforms.EnsureTyped(keys=selected_keys),  # 데이터를 MetaTensor로 변환
        # transforms.Orientationd(keys=selected_keys, axcodes="RAS"),
    ]
    zoom_transform = [
        transforms.CropForegroundd(
            keys=selected_keys,
            source_key=selected_keys[0],
            margin=(10, 10, 10),
            k_divisible=[1, 1, 1],
        ),
        transforms.Lambda(func=lambda data: calculate_new_pixdim(args, data)),
        transforms.Lambda(func=lambda data: apply_spacing_transform(args, data, selected_keys)),
        transforms.SpatialPadd(keys=selected_keys, spatial_size=(args.resize, args.resize, args.resize), mode='constant'),
    ]
    base_transform_2 = [
        # RobustZScoreNormalization(keys=(lambda x: x[:-1] if len(x) > 1 else x)(args.input_channel_names)),
        # RobustZScoreNormalization(keys=args.input_channel_names),
        RobustZScoreNormalization(keys=[el for el in args.input_channel_names if el not in MASKS]),
        transforms.ConcatItemsd(keys=args.input_channel_names, name='image', dim=0),
        transforms.DeleteItemsd(keys=args.input_channel_names),
        ConvertToMultiChannel(keys=["label"], labels=label_groups)
    ]
    if args.zoom:
        return base_transform_1 + zoom_transform + base_transform_2
    else:
        return base_transform_1 + base_transform_2
    
def get_forward_transform(args):
    selected_keys = [*args.input_channel_names]
    base_transform_1 = [
        transforms.EnsureTyped(keys=selected_keys),  # 데이터를 MetaTensor로 변환
        transforms.Orientationd(keys=selected_keys, axcodes="RAS"),
    ]
    zoom_transform = [
        transforms.CropForegroundd(
            keys=selected_keys,
            source_key=selected_keys[0],
            margin=(10, 10, 10),
            k_divisible=[1, 1, 1],
        ),
        transforms.Lambda(func=lambda data: calculate_new_pixdim(args, data)),
        transforms.Lambda(func=lambda data: apply_spacing_transform(args, data, selected_keys)),
        transforms.SpatialPadd(keys=selected_keys, spatial_size=(args.resize, args.resize, args.resize), mode='constant'),
    ]
    base_transform_2 = [
        # RobustZScoreNormalization(keys=(lambda x: x[:-1] if len(x) > 1 else x)(args.input_channel_names)),
        # RobustZScoreNormalization(keys=args.input_channel_names),
        RobustZScoreNormalization(keys=[el for el in args.input_channel_names if el not in MASKS]),
        transforms.ConcatItemsd(keys=args.input_channel_names, name='image', dim=0),
        transforms.DeleteItemsd(keys=args.input_channel_names),
    ]
    if args.zoom:
        return base_transform_1 + zoom_transform + base_transform_2
    else:
        return base_transform_1 + base_transform_2

def get_aug_transform(args):
    aug_patch_crop = [
        # crop
        transforms.RandCropByPosNegLabeld(
            keys=["image", 'label'],
            label_key='label',
            spatial_size=[args.patch_size] * 3,
            pos=args.pos_ratio,
            neg=args.neg_ratio,
            num_samples=args.multi_batch_size),
    ]
    aug_flip_lr = [
        # spatial aug
        # RAS orientation 의 경우
        # axis 0는 Left->Right (LR:0)
        # FeTS2022 일때, 
        # axis 1은 Inf->Sup (IS:1)
        # axis 2는 Post->Ant (PA:2)
        # CC359와 PPMI 일때, 
        # axis 1는 Post->Ant (PA:1)
        # axis 2은 Inf->Sup (IS:2)
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),
    ]
    aug_rest = [
        # rotate
        transforms.RandRotated(
            keys=["image", 'label'], prob=0.5, mode=['bilinear', 'nearest'],
            range_x=(0.3, 0.3), range_y=(0.3, 0.3), range_z=(0.3, 0.3)),

        # intensity aug
        transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

        # other stuff
        transforms.EnsureTyped(keys=["image", 'label']),
    ]
    if args.flip_lr:
        return aug_patch_crop + aug_flip_lr + aug_rest 
    else: 
        return aug_patch_crop + aug_rest

def custom_collate(batch):
    # 리스트 컴프리헨션을 사용하여 각 요소를 분리합니다.
    images = []
    labels = []
    indices = []
    names = []
    affines = []
    label_names_list = []

    # 배치 내의 모든 샘플을 반복 처리합니다.
    for sample in batch:
        for data in sample:
            images.append(data[0])  # el['image']
            labels.append(data[1])  # el['label']
            indices.append(data[2])  # index
            names.append(data[3])   # name
            affines.append(data[4])  # el['image'].affine
            label_names_list.append(data[5])  # self.label_names

    # PyTorch 텐서로 변환합니다. `torch.stack`은 동일한 크기의 텐서들을 쌓아서 새로운 차원을 추가합니다.
    # 모든 이미지와 라벨들이 동일한 크기를 가져야 `torch.stack`을 사용할 수 있습니다.
    images = torch.stack(images)
    labels = torch.stack(labels)

    # 이미지와 라벨, 그 외의 정보들을 튜플로 묶어 반환합니다.
    return images, labels, indices, names, affines, label_names_list