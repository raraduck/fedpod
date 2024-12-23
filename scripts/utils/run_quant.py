import os
import shutil
import nibabel as nib
import sys
import numpy as np
import torch
import torch.nn.functional as F
import glob
import metrics as metrics

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

        # print(src_sub_file, src_sub_vol.shape)
        # print(trg_sub_file, trg_sub_vol.shape)
        # print(trg_pet_file, trg_pet_vol.shape)

        dice = metrics.dice(src_sub_vol, trg_sub_vol)
        print(pid, dice.mean())



if __name__ == '__main__':
    # main()
    if len(sys.argv) == 3:
        src_base = str(sys.argv[1])
        trg_base = str(sys.argv[2])
        main(src_base, trg_base)
    else:
        print("Usage: python script.py <src_base_path> <trg_base_path>")