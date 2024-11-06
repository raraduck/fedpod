import sys
from Unet3DApp import Unet3DApp

if __name__ == '__main__':
    args = sys.argv[1:]
    App_args = Unet3DApp(args)
    App_args.run_forward(test_mode=App_args.cli_args.test_mode)
