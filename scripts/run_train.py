import sys
from Unet3DApp import Unet3DApp
import torch
import torch.multiprocessing

# [이 코드를 추가하세요]
# 공유 메모리(/dev/shm) 제한을 우회하여 파일 시스템을 통신에 사용합니다.
torch.multiprocessing.set_sharing_strategy('file_system')
if __name__ == '__main__':
    args = sys.argv[1:]
    App_args = Unet3DApp(args)
    App_args.run_train()
