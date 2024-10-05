from .Unet import UNet
from .blocks import PlainBlock, ResidualBlock

block_dict = {
    'plain': PlainBlock,
    'res': ResidualBlock
}

def get_unet(args):
    kwargs = {
        "input_channels"   : args.input_channels,
        "output_classes"   : args.num_classes,
        "channels_list"    : args.channels_list,

        "deep_supervision" : args.deep_supervision,
        "block"            : block_dict[args.block],
        "ds_layer"         : args.ds_layer,
        "kernel_size"      : args.kernel_size,
        "dropout_prob"     : args.dropout_prob,
        "norm_key"         : args.norm,
    }

    if args.unet_arch == 'unet':
        return UNet(**kwargs)
    else:
        raise NotImplementedError(args.unet_arch + " is not implemented.")