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
    seg_file = f'{pid}_seg.nii.gz'
    seg1_file = f'{pid}_seg1.nii.gz'
    seg2_file = f'{pid}_seg2.nii.gz'
    # sub_file = f'{pid}_sub.nii.gz'
    result_file = f'{pid}_{postfix}.nii.gz'
    # print(os.path.join(dst_folder, sub_file))
    # print(f"{sub_file} not exists.")
    assert os.path.exists(os.path.join(dst_folder, seg_file)), f"{seg_file} not exists."
    # assert os.path.exists(os.path.join(dst_folder, sub_file)), f"{sub_file} not exists."
    # assert not os.path.exists(os.path.join(dst_folder, seg_file)), f"{seg_file} already exists."

    seg_path = os.path.join(base, pid, seg_file)
    seg1_path = os.path.join(base, pid, seg1_file)
    seg2_path = os.path.join(base, pid, seg2_file)
    # sub_path = os.path.join(base, pid, sub_file)

    seg_proxy = nib.load(seg_path)
    seg_affine = seg_proxy.affine
    seg_data = seg_proxy.get_fdata()

    seg1_proxy = nib.load(seg1_path)
    seg1_affine = seg1_proxy.affine
    seg1_data = seg1_proxy.get_fdata()

    seg2_proxy = nib.load(seg2_path)
    seg2_affine = seg2_proxy.affine
    seg2_data = seg2_proxy.get_fdata()

    # sub_proxy = nib.load(sub_path)
    # sub_affine = sub_proxy.affine
    # sub_data = sub_proxy.get_fdata()

    active_data = np.zeros_like(seg_data)  # seg_data의 복사본 생성

    active_data[seg1_data == 4] = 4 # PC,PP
    active_data[seg2_data == 5] = 5 # VP
    active_data[seg1_data == 1] = 1 # VS
    active_data[seg1_data == 2] = 2 # AP
    active_data[seg1_data == 3] = 3 # AC

    final_img = nib.Nifti1Image(active_data, seg_affine)
    nib.save(final_img, os.path.join(dst_folder, result_file))
    print(f"{seg_path} -> {result_file}")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        src_base = os.path.normpath(sys.argv[1])
        postfix = str(sys.argv[2])
        main(src_base, postfix)
    else:
        print("Usage: python script.py <src_base_path> <postfix>")