import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
# import torch.nn.functional as F

def robust_sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=0.0, max=1.0)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdims=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):  # gt (b, x, y(, z))
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))  # gt (b, 1, x, y(, z))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)  # (b, 1, ...) --> (b, c, ...)

    # shape: (b, c, ...)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceWithLogitsLoss(nn.Module):
    ''' Adapt from nnUNet repo'''

    def __init__(self, channel_weights=None, nonlinear='sigmoid', smooth=1.0):
        super(SoftDiceWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.nonlinear = nonlinear
        # Channel-specific weights for the Dice loss
        self.channel_weights = channel_weights

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        axes = list(range(2, len(shp_x)))

        if self.nonlinear == 'sigmoid':
            x = robust_sigmoid(x)
        else:
            raise NotImplementedError(self.nonlinear)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        dc_loss = (1 - dc)

        # Apply channel weights if provided
        if self.channel_weights is None:
            return dc_loss
        else:
            weighted_dc_loss = dc_loss * self.channel_weights
            return weighted_dc_loss


class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(CustomBCEWithLogitsLoss, self).__init__()
        if pos_weight is not None:
            self.pos_weight = pos_weight
        else:
            self.pos_weight = None

    def forward(self, inputs, targets):
        if self.pos_weight is not None:
            # pos_weight를 inputs와 같은 크기로 확장
            # 예시: inputs가 [batch, channels, height, width] 형태라면,
            # pos_weight를 [1, channels, 1, 1]로 reshape하고 이를 inputs 크기에 맞추어 확장
            pos_weight = self.pos_weight.view(1, -1, 1, 1, 1).expand_as(inputs)
            max_val = (-inputs).clamp(min=0)
            log_weight = 1 + (pos_weight.exp() - 1) * targets
            loss = inputs - inputs * targets + log_weight * ((-max_val).exp() + (-inputs - max_val).exp()).log()
        else:
            max_val = (-inputs).clamp(min=0)
            loss = inputs - inputs * targets + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()

        return loss.mean()


class SoftDiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, channel_weights=None, dice_smooth=1.0):
        """Binary Cross Entropy & Soft Dice Loss

        Seperately return BCEWithLogitsloss and Dice loss.

        BCEWithLogitsloss is more numerically stable than Sigmoid + BCE
        """
        super(SoftDiceBCEWithLogitsLoss, self).__init__()

        # BCEWithLogitsLoss 인스턴스 생성, 라벨별 가중치 적용
        self.bce = nn.BCEWithLogitsLoss()
        # self.bce = CustomBCEWithLogitsLoss(pos_weight=channel_weights)
        self.dsc = SoftDiceWithLogitsLoss(
            channel_weights=channel_weights,
            nonlinear='sigmoid', smooth=dice_smooth)

    def forward(self, net_output: Tensor, target: Tensor):
        """Compute Binary Cross Entropy & Region Dice Loss

        Args:
            net_output (Tensor): [B, C, ...]
            target (Tensor): [B, C, ...]
        """
        bce_loss = self.bce(net_output, target)
        dsc_loss_by_channels = self.dsc(net_output, target)

        return bce_loss, dsc_loss_by_channels
