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


def entropy_from_probmap(prob_map, mask=None, eps=1e-10):
    """
    prob_map: [H,W,D], values in [0,1]
    mask: True인 위치만 엔트로피 계산 (배경 제외용)
    """
    p = np.clip(prob_map, eps, 1 - eps)

    if mask is None:
        mask = np.ones_like(p, dtype=bool)

    p_sel = p[mask]
    if p_sel.size == 0:
        return 0.0

    entropy_voxel = -p_sel * np.log(p_sel) - (1 - p_sel) * np.log(1 - p_sel)
    return float(np.mean(entropy_voxel))

def compute_entropy_from_volume(seg_mean, lower=30, upper=255, max_value=255.0):
    # 1. Scale to [0,1] if needed
    seg_mean = seg_mean / max_value

    # 2. Thresholding
    mask = (seg_mean >= lower / max_value) & (seg_mean <= upper / max_value)
    roi_values = seg_mean[mask]

    # 3. Normalize
    total = np.sum(roi_values)
    if total == 0:
        return 0
    probs = roi_values / total

    # 4. Entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

def load_nii(path):
    data = nib.load(path).get_fdata()
    return data
    
def case_load(filename, case_dir):
    src_file = os.path.join(case_dir, filename)
    src_vol = load_nii(src_file)
    return src_vol

def merge_roi_probmaps(src_base):
    c_path = os.getcwd()
    src_path = os.path.join(c_path, src_base)
    src_list = os.listdir(src_path)

    # ROI 리스트
    seg_names_left = "LVS,LAC,LPC,LAP,LPP,LVP".split(",")
    seg_names_right = "RVS,RAC,RPC,RAP,RPP,RVP".split(",")

    for pid in src_list:
        print(f"\n▶ Processing {pid}...")

        src_dir = os.path.join(src_path, pid)

        # =============================
        # ① LEFT ROI probability maps
        # =============================
        left_probs = []
        for roi in seg_names_left:
            fname = os.path.join(src_dir, f"{pid}_{roi}_prb.nii.gz")
            prob = nib.load(fname).get_fdata()  # float prob map
            left_probs.append(prob)

        # shape: (6, H, W, D)
        left_stack = np.stack(left_probs, axis=0)
        left_max = np.max(left_stack, axis=0)   # (H, W, D)
        # left_max = np.argmax(left_stack, axis=0) + 1  # 1~6 label

        # Save LEFT merged mask
        out_left = os.path.join(src_dir, f"{pid}_LS_prb.nii.gz")
        # nib.save(nib.Nifti1Image(left_max.astype(np.int16), affine=nib.load(fname).affine), out_left)
        nib.save(nib.Nifti1Image(left_max.astype(np.float32), affine=nib.load(fname).affine),out_left)
        print(f"  ✓ Saved {out_left}")

        # =============================
        # ② RIGHT ROI probability maps
        # =============================
        right_probs = []
        for roi in seg_names_right:
            fname = os.path.join(src_dir, f"{pid}_{roi}_prb.nii.gz")
            prob = nib.load(fname).get_fdata()
            right_probs.append(prob)

        right_stack = np.stack(right_probs, axis=0)
        right_max = np.max(right_stack, axis=0)
        # right_max = np.argmax(right_stack, axis=0) + 1

        # Save RIGHT merged mask
        out_right = os.path.join(src_dir, f"{pid}_RS_prb.nii.gz")
        # nib.save(nib.Nifti1Image(right_max.astype(np.int16), affine=nib.load(fname).affine), out_right)
        nib.save(nib.Nifti1Image(right_max.astype(np.float32), affine=nib.load(fname).affine),out_right)
        print(f"  ✓ Saved {out_right}")


def main1(src_base):
    """
    src_base 폴더에 pid 폴더들이 있고,
    각 pid 폴더 안에 merge_roi_probmaps()가 생성한
    - pid_LS_prb.nii.gz
    - pid_RS_prb.nii.gz
    파일이 있다고 가정
    """
    base_path = os.path.join(os.getcwd(), src_base)
    pid_list = sorted(os.listdir(base_path))

    print(f"=== Entropy Scores for {len(pid_list)} subjects ===")

    for pid in pid_list:
        pid_dir = os.path.join(base_path, pid)

        # 파일 경로
        file_LS = os.path.join(pid_dir, f"{pid}_LS_prb.nii.gz")
        file_RS = os.path.join(pid_dir, f"{pid}_RS_prb.nii.gz")

        # LS
        if os.path.exists(file_LS):
            vol_LS = nib.load(file_LS).get_fdata()
            entropy_LS = compute_entropy_from_volume(
                vol_LS, lower=30, upper=200
            )
        else:
            entropy_LS = None

        # RS
        if os.path.exists(file_RS):
            vol_RS = nib.load(file_RS).get_fdata()
            entropy_RS = compute_entropy_from_volume(
                vol_RS, lower=30, upper=200
            )
        else:
            entropy_RS = None

        # 출력
        print(f"[{pid}] LS Entropy: {entropy_LS:.4f} | RS Entropy: {entropy_RS:.4f}")


def main2(src_base):

    base_path = os.path.join(os.getcwd(), src_base)
    pid_list = sorted(os.listdir(base_path))

    print(f"=== Entropy Scores (Foreground-only) for {len(pid_list)} subjects ===\n")

    for pid in pid_list:
        pid_dir = os.path.join(base_path, pid)

        file_LS = os.path.join(pid_dir, f"{pid}_LS_prb.nii.gz")
        file_RS = os.path.join(pid_dir, f"{pid}_RS_prb.nii.gz")

        entropy_LS = entropy_RS = None

        # -------------------
        # LS entropy (배경 제외)
        # -------------------
        if os.path.exists(file_LS):
            vol_LS = nib.load(file_LS).get_fdata()
            vol_LS = vol_LS / 255.0  # [0,1] 스케일

            # 배경 제외 마스크 (확률이 거의 0인 voxel 제거)
            mask_LS = vol_LS > 0.01
            entropy_LS = entropy_from_probmap(vol_LS, mask=mask_LS)

        # -------------------
        # RS entropy (배경 제외)
        # -------------------
        if os.path.exists(file_RS):
            vol_RS = nib.load(file_RS).get_fdata()
            vol_RS = vol_RS / 255.0

            mask_RS = vol_RS > 0.01
            entropy_RS = entropy_from_probmap(vol_RS, mask=mask_RS)

        print(f"[{pid}] LS Entropy: {entropy_LS:.4f} | RS Entropy: {entropy_RS:.4f}")


if __name__ == '__main__':
    if len(sys.argv) == 2:
        src_base = os.path.normpath(str(sys.argv[1]))
        # merge_roi_probmaps(src_base)
        main1(src_base)
    else:
        print("Usage: python script.py <src_base_path>")
