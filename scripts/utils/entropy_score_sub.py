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
import math

seg_names_left  = "LVS,LAC,LPC,LAP,LPP,LVP".split(",")
seg_names_right = "RVS,RAC,RPC,RAP,RPP,RVP".split(",")

def multiclass_entropy_from_rois(pid_dir, pid, roi_names, thr=0.01, eps=1e-10):
    prob_maps = []
    for roi in roi_names:
        fname = os.path.join(pid_dir, f"{pid}_{roi}_prb.nii.gz")
        vol = nib.load(fname).get_fdata() / 255.0  # [0,1]로 스케일
        prob_maps.append(vol)

    # shape: (K, H, W, D)
    probs = np.stack(prob_maps, axis=0)

    # 혹시 합이 1이 안 될 수도 있으니 정규화
    sum_probs = np.sum(probs, axis=0, keepdims=True) + eps
    probs = probs / sum_probs

    # foreground mask: 어느 클래스든 thr 이상인 voxel만
    fg_mask = np.any(probs > thr, axis=0)

    if not np.any(fg_mask):
        return 0.0

    p_sel = probs[:, fg_mask]  # (K, N_fg)

    # multi-class entropy: -sum_k p_k log p_k
    entropy_vox = -np.sum(p_sel * np.log(p_sel + eps), axis=0)  # (N_fg,)

    # log(K)로 나서 [0,1]로 정규화 (해석용)
    H_norm = entropy_vox / math.log(probs.shape[0])

    return float(np.mean(H_norm))


def main(src_base):
    base_path = os.path.join(os.getcwd(), src_base)
    pid_list = sorted(os.listdir(base_path))

    print(f"=== Multi-class normalized entropy (0~1) for {len(pid_list)} subjects ===\n")

    for pid in pid_list:
        pid_dir = os.path.join(base_path, pid)

        H_L = multiclass_entropy_from_rois(pid_dir, pid, seg_names_left)
        H_R = multiclass_entropy_from_rois(pid_dir, pid, seg_names_right)

        print(f"[{pid}] LS H_mc: {H_L:.3f} | RS H_mc: {H_R:.3f}")


# def entropy_from_probmap_uncertain(vol, low=0.1, high=0.9, eps=1e-10):
#     p = np.clip(vol, eps, 1 - eps)
#     mask = (p > low) & (p < high)
#     p_sel = p[mask]
#     if p_sel.size == 0:
#         return 0.0
#     ent = -p_sel * np.log(p_sel) - (1 - p_sel) * np.log(1 - p_sel)
#     return float(np.mean(ent))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        src_base = os.path.normpath(str(sys.argv[1]))
        # merge_roi_probmaps(src_base)
        main(src_base)
    else:
        print("Usage: python script.py <src_base_path>")
