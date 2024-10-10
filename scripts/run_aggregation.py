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
parser.add_argument('--inst_ids', type=parse_1d_int_list, default=[])

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    if args.method == "fedavg":
        # base_path = os.path.join("states", f"R{args.rounds:02}r{args.round:02}")
        # print(base_path)
        # pth_list = [os.path.join(base_path, job, 'models', '*_last.pth') for idx, job in enumerate(args.job_ids,1) if idx in args.inst_ids]
        # # pth_list = [os.path.join(base_path, inst) for inst in args.job_ids]
        # # pth_list = [os.path.join(base_path, job) for idx, job in enumerate(args.job_ids,1) if str(idx) in args.inst_ids]
        # # # fedavg()
        # print(pth_list)
        pattern = os.path.join(os.getcwd(), 'states', f"R{args.rounds:02}r{args.round:02}", 'Job[1-9]', 'models', '*_last.pth')
        trg_pth = glob.glob(pattern)
        print(pattern)
        print(trg_pth)
        fedavg()
    else:
        raise NotImplementedError(f"{args.method.upper()} is not implemented on Aggregator()")
    

