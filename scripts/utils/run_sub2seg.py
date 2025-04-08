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

    active_data[data == 1] = 1 # LVS
    active_data[data == 2] = 1 # LAC
    active_data[data == 3] = 1 # LPC
    active_data[data == 4] = 1 # LAP
    active_data[data == 5] = 1 # LPP
    active_data[data == 6] = 1 # LVP

    active_data[data == 7] = 2 # RVS
    active_data[data == 8] = 2 # RAC
    active_data[data == 9] = 2 # RPC
    active_data[data ==10] = 2 # RAP
    active_data[data ==11] = 2 # RPP
    active_data[data ==12] = 2 # RVP

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