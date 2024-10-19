import math
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR


def get_scheduler(args, optimizer: torch.optim):
    """Generate the learning rate scheduler for **every epoch**

    Args:
        optimizer (torch.optim): Optimizer
        epochs (int): training epochs

    Returns:
        lr_scheduler
    """
    epochs = args.epochs
    try:
        scheduler = args.scheduler
        milestones = args.milestones
        lr_gamma = args.lr_gamma
    except Exception as e:
        # self.logger.error(msg)  # 로깅 시 스택 트레이스를 포함시킵니다
        # scheduler = 'step'
        # milestones = [3]
        # lr_gamma = 0.1
        raise e


    if scheduler == 'warmup_cosine':
        warmup = args.warmup_epochs
        warmup_cosine_lr = (lambda epoch: epoch / warmup if epoch <= warmup else 0.5 * (math.cos((epoch - warmup) / (epochs - warmup) * math.pi) + 1))
        lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
    elif scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler == 'step':
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)
    elif scheduler == 'poly':
        lr_scheduler = LambdaLR(optimizer, lambda epoch: (1 - epoch / epochs) ** 0.9)
    elif scheduler == 'none':
        lr_scheduler = None
    else:
        raise NotImplementedError(f"LR scheduler {scheduler} is not implemented.")

    return lr_scheduler