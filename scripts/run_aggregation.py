import os
import time
import argparse
import sys
import glob
from utils.tools import *
from Aggregator import *

parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--method', type=str, default="fedavg", 
    choices=['fedavg', 'fedpid', 'fedpod'], help='type of avg')
parser.add_argument('--job_id', type=str, default="")
parser.add_argument('--inst_id', type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    if args.method == "fedavg":
        base_dir = os.path.join('/','fedpod','states')
        # base_dir = os.path.join('.','states')
        round_dir = os.path.join(base_dir, f"R{args.rounds:02}r{args.round:02}")
        inst_dir = os.path.join(round_dir, f"{args.job_id}*")
        pattern = os.path.join(inst_dir, 'models', '*_last.pth')
        pth_path = glob.glob(pattern)
        print(pattern)
        print(pth_path)
        global_state_dict = fedavg(pth_path)
        state = {'model': global_state_dict}

        # exp_folder = f"{self.cli_args.exp_name}"
        # save_model_path1 = os.path.join("states", exp_folder)
        # save_model_path2 = f"round_{self.round:02d}"
        # save_model_path3 = f"epoch_{from_epoch:03d}"
        next_round_dir = os.path.join(base_dir, f"R{args.rounds:02}r{args.round+1:02}")
        os.makedirs(next_round_dir, exist_ok=True)
        save_model_path = os.path.join(next_round_dir, f"R{args.rounds:02}r{args.round+1:02}.pth")
        torch.save(state, save_model_path)
    else:
        raise NotImplementedError(f"{args.method.upper()} is not implemented on Aggregator()")
    

