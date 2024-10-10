import os
import shutil
import time
import argparse
import sys
import glob
from utils.tools import *
from Aggregator import *

parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--algorithm', type=str, default="fedavg", 
    choices=['fedavg', 'fedpid', 'fedpod'], help='type of avg')
parser.add_argument('--job_id', type=str, default="")
parser.add_argument('--inst_id', type=int, default=0)
parser.add_argument('--weight_path', type=str, default="None",
    help='path to pretrained encoder or decoder weight, None for train-from-scratch')

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    prev_round = args.round - 1
    curr_round = args.round
    base_dir = os.path.join('/','fedpod','states')
    if args.algorithm == "fedavg":
        if prev_round < 0:
            orig_file = args.weight_path
            curr_round_dir = os.path.join(base_dir, f"R{args.rounds:02}r{curr_round:02}")
            os.makedirs(curr_round_dir, exist_ok=True)
            save_model_path = os.path.join(curr_round_dir, f"R{args.rounds:02}r{curr_round:02}.pth")
            shutil.copy2(orig_file, save_model_path)
        else:
            # base_dir = os.path.join('.','states')
            prev_round_dir = os.path.join(base_dir, f"R{args.rounds:02}r{prev_round:02}")
            inst_dir = os.path.join(prev_round_dir, f"{args.job_id}*")
            pattern = os.path.join(inst_dir, 'models', '*_last.pth')
            pth_path = glob.glob(pattern)
            # print(pattern)
            # print(pth_path)
            global_state_dict = fedavg(pth_path)
            state = {'model': global_state_dict}
            # exp_folder = f"{self.cli_args.exp_name}"
            # save_model_path1 = os.path.join("states", exp_folder)
            # save_model_path2 = f"round_{self.round:02d}"
            # save_model_path3 = f"epoch_{from_epoch:03d}"
            curr_round_dir = os.path.join(base_dir, f"R{args.rounds:02}r{curr_round:02}")
            os.makedirs(curr_round_dir, exist_ok=True)
            save_model_path = os.path.join(curr_round_dir, f"R{args.rounds:02}r{curr_round:02}.pth")
            torch.save(state, save_model_path)
    else:
        raise NotImplementedError(f"{args.algorithm.upper()} is not implemented on Aggregator()")
    

