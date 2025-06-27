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

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    node_path = os.path.join(base_path, 'data', 'sol1')
    save_path = os.path.join(node_path, "sol1-output")
    node_names = [
        "sol1-node1_0_forward", 
        "sol1-node2_0_forward", 
        "sol1-node3_0_forward", 
        "sol1-node4_0_forward", 
        "sol1-node5_0_forward", 
        "sol1-node6_0_forward"
    ]
    case_names = os.listdir(os.path.join(node_path, node_names[0]))
    case_names = ['cc0182', 'cc0183', 'cc0184', 'cc0185']

    seg_names = [lambda x: f"{x}_LS_prb.nii.gz",  lambda x: f"{x}_RS_prb.nii.gz"]
    for seg_name in seg_names:
        for idx, case_name in enumerate(case_names):
            if idx < 50:
                customized_node_names = [*node_names[1:]]
            elif idx < 100:
                customized_node_names = [*node_names[:1], *node_names[2:]]
            elif idx < 150:
                customized_node_names = [*node_names[:2], *node_names[3:]]
            elif idx < 200:
                customized_node_names = [*node_names[:3], *node_names[4:]]
            elif idx < 250:
                customized_node_names = [*node_names[:4], *node_names[5:]]
            else:
                customized_node_names = [*node_names[:5]]
            
            # if idx in [0, 50, 100, 150, 200, 250]:
            #     print(customized_node_names)

            seg_vols = []
            for node_name in customized_node_names:
                case_dir = os.path.join(node_path, node_name, case_name)
                seg_vol = case_load(seg_name(case_name), case_dir)
                seg_vols.append(seg_vol)
            # 평균 계산
            seg_mean = np.mean(seg_vols, axis=0)

            # NIfTI 저장을 위한 헤더 참조 (첫 번째 파일 사용)
            ref_nii = nib.load(os.path.join(case_dir, seg_name(case_name)))
            seg_nii = nib.Nifti1Image(seg_mean, affine=ref_nii.affine, header=ref_nii.header)

            # 저장 경로 생성
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"avg_{seg_name(case_name)}")
            nib.save(seg_nii, save_file)
            # print(f"Averaged segmentation saved to: {save_file}")

            entropy_score = compute_entropy_from_volume(seg_mean, lower=30, upper=200)
            print(f"{seg_name(case_name)}: Entropy of ROI: {entropy_score:.4f}")
    
if __name__ == '__main__':
    main()