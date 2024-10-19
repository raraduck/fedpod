import os
import time
import argparse
from .tools import *


def parse_args(argv):
    """args of segmentation tasks"""
    parser = argparse.ArgumentParser()
    # meta
    # parser.add_argument('--use_gpu', action='store_true', default=False, help='enable gpu (default: False)')
    parser.add_argument('--save_infer', type=int, choices=[0, 1], default=1, help='whether save individual prediction')
    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=0, help='Enable GPU (0 = No, 1 = Yes, default: 0)')
    parser.add_argument('--job_name', type=str, default=None, help='create job_name folder to save results')
    
    # data
    parser.add_argument('--cases_split', type=str, default=None, help='file path for split csv')
    parser.add_argument('--inst_ids', type=parse_1d_int_list, default=[])
    parser.add_argument('--label_groups', type=parse_2d_int_list, default=[[11, 50],[12, 51]], help='2D list of label groups')
    parser.add_argument('--label_names', type=parse_1d_str_list, default=['caud', 'puta'], help='1D list of label names')
    parser.add_argument('--label_index', type=parse_1d_int_list, default=[10, 20], help='1D list of label index')
    parser.add_argument('--input_channel_names', type=parse_1d_str_list, default=['t1'], help='1D list of input channel names')
    parser.add_argument('--dataset', type=str, default='CC359PPMI', help='dataset list',
        choices=['FeTS2022', 'iSeg-2019', 'gaain', 'CC359', 'CC359PPMI'])
    parser.add_argument('--data_root', type=str, default='data/', help='root dir of dataset')
    parser.add_argument('--inst_root', type=str, default='inst_01/', help='root dir of inst')
    
    # data augmentation
    parser.add_argument('--zoom', type=int, choices=[0, 1], default=0, help='enable zoom to crop (default: False)')
    parser.add_argument('--flip_lr', type=int, choices=[0, 1], default=0, help='enable flip to left and right (default: False)')
    parser.add_argument('--resize', type=int, default=256, help='resize')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size')
    parser.add_argument('--pos_ratio', type=float, default=1.0,
        help="prob of picking positive patch (center in foreground)")
    parser.add_argument('--neg_ratio', type=float, default=0.0,
        help="prob of picking negative patch (center in background)")
    parser.add_argument('--min_dlen', type=int, default=1)
    parser.add_argument('--max_dlen', type=int, default=99999)

    # train
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load data')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--optim', type=str, default='adam', help='optimizer',
        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--clip_grad', type=int, choices=[0, 1], default=0, help='whether to clip gradient')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--amp', type=int, choices=[0, 1], default=0, help='using mixed precision')

    parser.add_argument('--scheduler', type=str, default='step', help='scheduler',
                        choices=['warmup_cosine', 'cosine', 'step', 'poly', 'none'])
    parser.add_argument('--milestones', type=int, nargs="+", default=[10],
        help='milestones for multistep decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
        help='decay factor for multistep decay')
    # parser.add_argument('--warmup_epochs', type=int, default=5, help='warm up epochs')

    # infer
    parser.add_argument('--patch_overlap', type=float, default=0.5,
        help="overlap ratio between patches")
    parser.add_argument('--sw_batch_size', type=int, default=1, help="sliding window batch size")
    parser.add_argument('--sliding_window_mode', type=str, default='constant',
        choices=['constant', 'gaussian'], help='sliding window importance map mode')
    
    # model (u-net)
    parser.add_argument('--weight_path', type=str, required=True, help='The model path must be provided.')
    parser.add_argument('--unet_arch', type=str, default='unet',
        choices=['unet3d', 'unet'], help='Architecuture of the U-Net')
    parser.add_argument('--channels_list', type=parse_1d_int_list, default=[64, 128, 256],
        help="#channels of every levels of decoder in a top-down order")
    parser.add_argument('--deep_supervision', type=int, choices=[0, 1], default=0, help='whether use deep supervision')
    parser.add_argument('--block', type=str, default='plain', choices=['plain', 'res'],
        help='Type of convolution block')
    parser.add_argument('--ds_layer', type=int, default=4,
        help='last n layer to use deep supervision')
    parser.add_argument('--kernel_size', type=int, default=3, help="size of conv kernels")
    parser.add_argument('--dropout_prob', type=float, default=0.0, help="prob of dropout")
    parser.add_argument('--norm', type=str, default='instance',
        choices=['instance', 'batch', 'group'], help='type of norm')
    
    
    args = parser.parse_args(argv)
    return args