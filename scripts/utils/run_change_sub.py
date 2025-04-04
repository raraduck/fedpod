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
    print(c_path)
    print(src_path)
    print(src_list)
    print(trg_path)
    for pid in src_list:
        src_dir = os.path.join(src_path, pid)
        trg_dir = os.path.join(trg_path, pid)
        src_seg_file = os.path.join(src_dir, f"{pid}_sub.nii.gz")
        trg_sub_file = os.path.join(trg_dir, f"{pid}_sub.nii.gz")
        assert os.path.exists(src_seg_file), f"{src_seg_file} not exists."

        proxy = nib.load(src_seg_file)
        affine = proxy.affine
        data = proxy.get_fdata()

        active_data = np.zeros_like(data)

        active_data[data == 1] =  7 # LVS
        active_data[data == 2] =  8 # LAC
        active_data[data == 3] =  9 # LPC
        active_data[data == 4] = 10 # LAP
        active_data[data == 5] = 11 # LPP
        active_data[data == 6] = 12 # LVP

        active_data[data == 7] =  1 # RVS
        active_data[data == 8] =  2 # RAC
        active_data[data == 9] =  3 # RPC
        active_data[data ==10] =  4 # RAP
        active_data[data ==11] =  5 # RPP
        active_data[data ==12] =  6 # RVP

        final_img = nib.Nifti1Image(active_data, affine)
        nib.save(final_img, trg_sub_file)
        print(f"{src_seg_file} -> {trg_sub_file}")


if __name__ == '__main__':
    # main()
    if len(sys.argv) == 3:
        src_base = os.path.normpath(str(sys.argv[1]))
        trg_base = os.path.normpath(str(sys.argv[2]))
        main(src_base, trg_base)
    else:
        print("Usage: python script.py <src_base_path> <trg_base_path>")