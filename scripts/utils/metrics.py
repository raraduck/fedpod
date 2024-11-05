import numpy as np
from medpy.metric import hd95 as hd95_medpy
from torch import Tensor
import torch


def cdsc(output: Tensor, target: Tensor) -> np.ndarray:
    """Calculate root mean squared error across the spatial dimensions of image batches."""
    target = target.float()
    masked_output = target * output
    squared_error = (target - masked_output) ** 2
    mse = squared_error.mean(dim=(2, 3, 4))  # 평균을 각 샘플의 spatial dimensions (D, H, W)에 대해서 계산
    sum = squared_error.sum(dim=(2, 3, 4))  # 평균을 각 샘플의 spatial dimensions (D, H, W)에 대해서 계산
    rmse = torch.sqrt(mse)/sum  # 각 샘플에 대한 RMSE 계산
    return rmse.cpu().numpy()  # 결과를 CPU로 이동시킨 후 NumPy 배열로 변환

def pvdc(output: Tensor, target: Tensor, eps: float = 1e-5) -> np.ndarray:
    """calculate multilabel batch dice"""
    target = target.float()
    num = 2 * (output != target).sum(dim=(2, 3, 4)) + eps
    den = output.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + eps
    pvd = num / den

    return pvd.cpu().numpy()

def dice(output: Tensor, target: Tensor, eps: float = 1e-5) -> np.ndarray:
    """calculate multilabel batch dice"""
    target = target.float()
    num = 2 * (output * target).sum(dim=(2, 3, 4)) + eps
    den = output.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + eps
    dsc = num / den

    return dsc.cpu().numpy()


def hd95(output: Tensor, target: Tensor, spacing=None) -> np.ndarray:
    """ output and target should all be boolean tensors"""
    output = output.bool().cpu().numpy()
    target = target.bool().cpu().numpy()

    B, C = target.shape[:2]
    hd95 = np.zeros((B, C), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            pred, gt = output[b, c], target[b, c]

            # reward if gt all background, pred all background
            if (not gt.sum()) and (not pred.sum()):
                hd95[b, c] = 0.0
            # penalize if gt all background, pred has foreground
            elif (not gt.sum()) and (pred.sum()):
                hd95[b, c] = 373.1287
            # penalize if gt has forground, but pred has no prediction
            elif (gt.sum()) and (not pred.sum()):
                hd95[b, c] = 373.1287
            else:
                hd95[b, c] = hd95_medpy(pred, gt, voxelspacing=spacing)

    return hd95