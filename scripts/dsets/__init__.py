# from .blocks import PlainBlock, ResidualBlock
# from .unet import UNet, MultiEncoderUNet
# from monai.networks.nets import UNETR

# block_dict = {
#     'plain': PlainBlock,
#     'res': ResidualBlock
# }

import os
# from .gaain import GaainDataset
# from .cc359 import CC359Dataset
from .cc359ppmi import CC359PPMIDataset
from .fets1470 import FETS1470Dataset
from .dataset_utils import get_base_transform, get_forward_transform, get_aug_transform, custom_collate
def get_dataset(args, case_names, _transforms, mode, label_names, custom_min_len=1, custom_max_len=99999, index_filter=None):
    # kwargs = {
    #     "input_channels": args.input_channels,
    #     "output_classes": args.num_classes,
    #     "channels_list": args.channels_list,
    #     "deep_supervision": args.deep_supervision,
    #     "ds_layer": args.ds_layer,
    #     "kernel_size": args.kernel_size,
    #     "dropout_prob": args.dropout_prob,
    #     "norm_key": args.norm,
    #     "block": block_dict[args.block],
    # }

    # if args.dataset == 'gaain':
    #     return GaainDataset(
    #         data_root=os.path.join(args.data_root, args.dataset),
    #         mode=mode,
    #         case_names=case_names,
    #         label_names=label_names,
    #         transforms=_transforms)
    # elif args.dataset == 'CC359':
    #     return CC359Dataset(
    #         data_root=os.path.join(args.data_root, args.dataset),
    #         mode=mode,
    #         case_names=case_names,
    #         input_channel_names=args.input_channel_names,
    #         label_names=label_names,
    #         custom_lower_bound=custom_min_len,
    #         custom_upper_bound=custom_max_len,
    #         transforms=_transforms)
    if args.dataset == 'CC359PPMI':
        return CC359PPMIDataset(
            args,
            # data_root=os.path.join(args.data_root, args.dataset),
            data_root='data', # args.data_root,
            inst_root=args.inst_root,
            mode=mode,
            case_names=case_names,
            input_channel_names=args.input_channel_names,
            label_names=label_names,
            custom_lower_bound=custom_min_len,
            custom_upper_bound=custom_max_len,
            transforms=_transforms, 
            index_filter=index_filter)
    elif args.dataset == 'FETS1470':
        return FETS1470Dataset(
            args,
            # data_root=os.path.join(args.data_root, args.dataset),
            data_root='data', # args.data_root,
            inst_root=args.inst_root,
            mode=mode,
            case_names=case_names,
            input_channel_names=args.input_channel_names,
            label_names=label_names,
            custom_lower_bound=custom_min_len,
            custom_upper_bound=custom_max_len,
            transforms=_transforms, 
            index_filter=index_filter)
    else:
        raise NotImplementedError(args.dataset + " is not implemented.")