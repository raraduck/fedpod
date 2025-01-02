import os
import shutil
import nibabel as nib
import sys
import numpy as np
import glob

def main(src_base, postfix):
    base = os.path.dirname(src_base)
    pid = os.path.basename(src_base)
    dst_folder = os.path.join(base, pid)
    sub_file = f'{pid}_sub.nii.gz'
    seg_file = f'{pid}_{postfix}.nii.gz'
    # print(os.path.join(dst_folder, sub_file))
    # print(f"{sub_file} not exists.")
    assert os.path.exists(os.path.join(dst_folder, sub_file)), f"{sub_file} not exists."
    # assert not os.path.exists(os.path.join(dst_folder, seg_file)), f"{seg_file} already exists."

    src_file = os.path.join(base, pid, sub_file)

    proxy = nib.load(src_file)
    affine = proxy.affine
    data = proxy.get_fdata()

    active_data = np.zeros_like(data)

    active_data[data == 1] = 26 # LVS
    active_data[data == 2] = 11 # LAC
    active_data[data == 3] = 11 # LPC
    active_data[data == 4] = 12 # LAP
    active_data[data == 5] = 12 # LPP
    active_data[data == 6] = 12 # LVP

    active_data[data == 7] = 58 # RVS
    active_data[data == 8] = 50 # RAC
    active_data[data == 9] = 50 # RPC
    active_data[data ==10] = 51 # RAP
    active_data[data ==11] = 51 # RPP
    active_data[data ==12] = 51 # RVP

    final_img = nib.Nifti1Image(active_data, affine)
    nib.save(final_img, os.path.join(dst_folder, seg_file))
    print(f"{src_file} -> {seg_file}")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        src_base = os.path.normpath(sys.argv[1])
        postfix = str(sys.argv[2])
        main(src_base, postfix)
    else:
        print("Usage: python script.py <src_base_path> <postfix>")