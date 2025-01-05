import os
import shutil
import nibabel as nib
import sys
import numpy as np
import torch
import torch.nn.functional as F
import glob
import metrics as metrics
from misc import *


def get_bp(pet_vol, sub_vol):
    means = []
    for i in range(sub_vol.shape[1]):  # 12 채널 순회
        mask = sub_vol[:,i].unsqueeze(0)  # i번째 채널의 마스크 (shape: [1, 128, 128, 128])
        masked_pet = pet_vol[mask > 0]  # 마스크가 1(True)인 PET 값만 추출
        mean_value = masked_pet.mean() if masked_pet.numel() > 0 else torch.tensor(0.0)  # 평균값 계산 (비어있으면 0)
        means.append(mean_value.item())
    return means


def get_suvr(pet_vol, ref_vol):
    pet_ref = pet_vol[ref_vol > 0].mean()
    return pet_vol / pet_ref


def load_nii(path):
    data = nib.load(path).get_fdata()

    data[data == 26] = 1
    data[data == 11] = 2
    data[data == 12] = 3

    data[data == 58] = 4
    data[data == 50] = 5
    data[data == 51] = 6
    return data

def main(src_base, trg_base):
    c_path = os.getcwd()
    src_path = os.path.join(c_path, src_base)
    src_list = os.listdir(src_path)
    trg_path = os.path.join(c_path, trg_base)
    trg_list = os.listdir(trg_path)
    print(c_path)
    print(src_path)
    print(trg_path)
    print(src_list)
    print(trg_list)
    assert len(src_list) == len(trg_list), f"src_list and trg_list len must be same"

    seg_names = "[LV,LC,LP,RV,RC,RP]".strip("[]").split(",")
    case_metrics_meter = CaseSegMetricsMeter(seg_names, metrics_list=['DICE','HD95','PVDC', 'SUV1', 'SUV2'])
    for pid in src_list:
        src_dir = os.path.join(src_path, pid)
        trg_dir = os.path.join(trg_path, pid)
        # print(trg_dir)
        # check_list = os.listdir(trg_dir)
        src_seg_file = os.path.join(src_dir, f"{pid}_seg.nii.gz")
        trg_seg_file = os.path.join(trg_dir, f"{pid}_seg.nii.gz")
        trg_ref_file = os.path.join(trg_dir, f"{pid}_ref.nii.gz")
        trg_pet_file = os.path.join(trg_dir, f"{pid}_pt.nii.gz")

        assert os.path.exists(src_seg_file), f"{src_seg_file} not exists."
        assert os.path.exists(trg_seg_file), f"{trg_seg_file} not exists."
        assert os.path.exists(trg_ref_file), f"{trg_ref_file} not exists."
        assert os.path.exists(trg_pet_file), f"{trg_pet_file} not exists."

        src_seg_vol = torch.from_numpy(load_nii(src_seg_file)).long().unsqueeze(0)
        trg_seg_vol = torch.from_numpy(load_nii(trg_seg_file)).long().unsqueeze(0)
        trg_ref_vol = torch.from_numpy(nib.load(trg_ref_file).get_fdata()).bool().long().unsqueeze(0)
        trg_pet_vol = torch.from_numpy(nib.load(trg_pet_file).get_fdata()).long().unsqueeze(0)



        # 원-핫 인코딩 (num_classes=13, 라벨 1~12)
        src_seg_vol = F.one_hot(src_seg_vol, num_classes=7)[..., 1:].permute(0, 4, 1, 2, 3).to(torch.float)
        trg_seg_vol = F.one_hot(trg_seg_vol, num_classes=7)[..., 1:].permute(0, 4, 1, 2, 3).to(torch.float)
        trg_ref_vol = F.one_hot(trg_ref_vol, num_classes=2)[..., 1:].permute(0, 4, 1, 2, 3).to(torch.float)
        trg_pet_vol = trg_pet_vol.unsqueeze(0).to(torch.float)

        dice = metrics.dice(src_seg_vol, trg_seg_vol)
        hd95 = metrics.hd95(src_seg_vol, trg_seg_vol)
        pvdc = metrics.pvdc(src_seg_vol, trg_seg_vol)
        norm_pet = get_suvr(trg_pet_vol, trg_ref_vol)
        suvr_src = metrics.suvr(norm_pet, src_seg_vol)
        suvr_trg = metrics.suvr(norm_pet, trg_seg_vol)
        # print(suvr_src)
        # print(suvr_trg)

        case_metrics_meter.update(dice, hd95, pvdc, [pid], 1, suv1=suvr_src, suv2=suvr_trg)
        print(pid, dice.mean())
    # save_val_path = os.path.join(src_path)
    # save_val_path = os.path.dirname(src_path)
    case_metrics_meter.output(c_path)



if __name__ == '__main__':
    # main()
    if len(sys.argv) == 3:
        src_base = os.path.normpath(str(sys.argv[1]))
        trg_base = os.path.normpath(str(sys.argv[2]))
        main(src_base, trg_base)
    else:
        print("Usage: python script.py <src_base_path> <trg_base_path>")