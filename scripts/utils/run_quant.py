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

    seg_names = "[LVS,LAC,LAP,LAP,LPP,LVP,RVS,RAC,RAP,RAP,RPP,RVP]".strip("[]").split(",")
    case_metrics_meter = CaseSegMetricsMeter(seg_names, metrics_list=['DICE','HD95','PVDC'])
    for pid in src_list:
        src_dir = os.path.join(src_path, pid)
        trg_dir = os.path.join(trg_path, pid)
        # print(trg_dir)
        # check_list = os.listdir(trg_dir)
        src_sub_file = os.path.join(src_dir, f"{pid}_sub.nii.gz")
        trg_sub_file = os.path.join(trg_dir, f"{pid}_sub.nii.gz")
        trg_pet_file = os.path.join(trg_dir, f"{pid}_pt.nii.gz")

        src_sub_vol = torch.from_numpy(nib.load(src_sub_file).get_fdata()).long().unsqueeze(0)
        trg_sub_vol = torch.from_numpy(nib.load(trg_sub_file).get_fdata()).long().unsqueeze(0)
        # trg_pet_vol = torch.from_numpy(nib.load(trg_pet_file).get_fdata()).float().unsqueeze(0)

        # 원-핫 인코딩 (num_classes=13, 라벨 1~12)
        src_sub_vol = F.one_hot(src_sub_vol, num_classes=13)[..., 1:].permute(0, 4, 1, 2, 3)
        trg_sub_vol = F.one_hot(trg_sub_vol, num_classes=13)[..., 1:].permute(0, 4, 1, 2, 3)

        dice = metrics.dice(src_sub_vol, trg_sub_vol)
        hd95 = metrics.hd95(src_sub_vol, trg_sub_vol)
        pvdc = metrics.pvdc(src_sub_vol, trg_sub_vol)

        case_metrics_meter.update(dice, hd95, pvdc, [pid], 1)
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