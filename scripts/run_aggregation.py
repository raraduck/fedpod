import os
import shutil
import time
import argparse
import sys
import glob
import natsort
from utils.tools import *
from Aggregator import *
from utils.misc import *


parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--algorithm', type=str, default="fedavg", 
    choices=['fedavg', 'fedwavg', 'fedpid', 'fedpod'], help='type of avg')
parser.add_argument('--job_prefix', type=str, default="")
parser.add_argument('--inst_id', type=int, default=0)
parser.add_argument('--weight_path', type=str, required=True,
    help='path to pretrained encoder or decoder weight, None for train-from-scratch')


def main(args):
    log_filename = f"{args.job_prefix}_R{args.rounds:02}r{args.round:02}.log"
    logger = initialization_logger(args, log_filename)
    prev_round = args.round - 1
    curr_round = args.round
    base_dir = os.path.join('/','fedpod','states')
    # base_dir = os.path.join('.','states')
    
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] aggregation algorithm is {args.algorithm.upper()}...")
    if prev_round < 0:
        logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] initial model setup from {args.weight_path}...")
        orig_file = args.weight_path
        center_dir = os.path.join(base_dir, f"{args.job_prefix}_{args.inst_id}")
        curr_round_dir = os.path.join(center_dir, f"R{args.rounds:02}r{curr_round:02}")
        models_dir = os.path.join(curr_round_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        save_model_path = os.path.join(models_dir, f"R{args.rounds:02}r{curr_round:02}.pth")
        shutil.copy2(orig_file, save_model_path)
        logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] initial model setup to {save_model_path}...")
        return
    
    inst_dir = os.path.join(base_dir, f"{args.job_prefix}_*") # inst0 also included 
    prev_round_dir = os.path.join(inst_dir, f"R{args.rounds:02}r{prev_round:02}")
    pattern = os.path.join(prev_round_dir, 'models', '*_last.pth') # but, _last.pth removes inst0 because inst0 never has _last.pth file on it
    pth_path = natsort.natsorted(glob.glob(pattern))
    for pth in pth_path:
        logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] local models are from {pth}...")
    local_models_with_dlen = [torch.load(m) for m in pth_path]

    if args.algorithm == "fedavg":
        P = [1 for el in local_models_with_dlen]
        W = [el/sum(P) for el in P]
        M = [el['model'] for el in local_models_with_dlen]
        aggregated_model = fedavg(W, M)
    elif args.algorithm == "fedwavg":
        P = [el['P'] for el in local_models_with_dlen]
        W = [el/sum(P) for el in P]
        M = [el['model'] for el in local_models_with_dlen]
        aggregated_model = fedwavg(W, M)
        # averaged_loss = lossavg()
    else:
        raise NotImplementedError(f"{args.algorithm.upper()} is not implemented on Aggregator()")
    state = {
        'model': aggregated_model, 
        'pre_metrics':{
            'DSCL_AVG':0.11,
            'DICE_AVG':0.12,
            'HD95_AVG':0.13
        }, 
        'post_metrics':{
            'DSCL_AVG':0.21,
            'DICE_AVG':0.22,
            'HD95_AVG':0.23
        }, 
    }
    center_dir = os.path.join(base_dir, f"{args.job_prefix}_{args.inst_id}")
    curr_round_dir = os.path.join(center_dir, f"R{args.rounds:02}r{curr_round:02}")
    models_dir = os.path.join(curr_round_dir, 'models')
    # curr_round_dir = os.path.join(base_dir, f"R{args.rounds:02}r{curr_round:02}")
    # center_dir = os.path.join(curr_round_dir, f"{args.job_prefix}_{args.inst_id}")
    # models_dir = os.path.join(center_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_model_path = os.path.join(models_dir, f"R{args.rounds:02}r{curr_round:02}.pth")
    torch.save(state, save_model_path)
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] models are aggregated to {save_model_path}...")





        
    # averaged_loss = lossavg()
    mean_prev_DSCL_AVG = np.mean([el['pre_metrics']['DSCL_AVG'] for el in local_models_with_dlen])
    mean_prev_DICE_AVG = np.mean([el['pre_metrics']['DICE_AVG'] for el in local_models_with_dlen])
    mean_prev_HD95_AVG = np.mean([el['pre_metrics']['HD95_AVG'] for el in local_models_with_dlen])
    
    local_prev_DSCL_AVG = {
        f"PREV_{int(el['args'].job_name.split('_')[-1]):04d}_DSCL": np.mean([
                v for (k,v) in el['pre_metrics'].items() if 'DSCL' in k
                # el['pre_metrics']['DSCL_LA'],
                # el['pre_metrics']['DSCL_LC'],
                # el['pre_metrics']['DSCL_LP'],
                # el['pre_metrics']['DSCL_RA'],
                # el['pre_metrics']['DSCL_RC'],
                # el['pre_metrics']['DSCL_RP'],
        ]) for el in local_models_with_dlen
    }
    local_prev_DICE_AVG = {
        f"PREV_{int(el['args'].job_name.split('_')[-1]):04d}_DICE": np.mean([
                v for (k,v) in el['pre_metrics'].items() if 'DICE' in k
        ]) for el in local_models_with_dlen
    }
    local_prev_HD95_AVG = {
        f"PREV_{int(el['args'].job_name.split('_')[-1]):04d}_HD95": np.mean([
                v for (k,v) in el['pre_metrics'].items() if 'HD95' in k
        ]) for el in local_models_with_dlen
    }
    # local_pre_DSCL_LA = np.mean([el['pre_metrics']['DSCL_LA'] for el in local_models_with_dlen])
    # local_pre_DSCL_LC = np.mean([el['pre_metrics']['DSCL_LC'] for el in local_models_with_dlen])
    # local_pre_DSCL_LP = np.mean([el['pre_metrics']['DSCL_LP'] for el in local_models_with_dlen])
    # local_pre_DSCL_RA = np.mean([el['pre_metrics']['DSCL_RA'] for el in local_models_with_dlen])
    # local_pre_DSCL_RC = np.mean([el['pre_metrics']['DSCL_RC'] for el in local_models_with_dlen])
    # local_pre_DSCL_RP = np.mean([el['pre_metrics']['DSCL_RP'] for el in local_models_with_dlen])

    mean_post_DSCL_AVG = np.mean([el['post_metrics']['DSCL_AVG'] for el in local_models_with_dlen])
    mean_post_DICE_AVG = np.mean([el['post_metrics']['DICE_AVG'] for el in local_models_with_dlen])
    mean_post_HD95_AVG = np.mean([el['post_metrics']['HD95_AVG'] for el in local_models_with_dlen])

    local_post_DSCL_AVG = {
        f"POST_{int(el['args'].job_name.split('_')[-1]):04d}_DSCL": np.mean([
                v for (k,v) in el['post_metrics'].items() if 'DSCL' in k
                # el['post_metrics']['DSCL_LA'],
                # el['post_metrics']['DSCL_LC'],
                # el['post_metrics']['DSCL_LP'],
                # el['post_metrics']['DSCL_RA'],
                # el['post_metrics']['DSCL_RC'],
                # el['post_metrics']['DSCL_RP'],
        ]) for el in local_models_with_dlen
    }
    local_post_DICE_AVG = {
        f"POST_{int(el['args'].job_name.split('_')[-1]):04d}_DICE": np.mean([
                v for (k,v) in el['post_metrics'].items() if 'DICE' in k
        ]) for el in local_models_with_dlen
    }
    local_post_HD95_AVG = {
        f"POST_{int(el['args'].job_name.split('_')[-1]):04d}_HD95": np.mean([
                v for (k,v) in el['post_metrics'].items() if 'HD95' in k
        ]) for el in local_models_with_dlen
    }
    # local_pre_DSCL = [el['pre_metrics']['DSCL_AVG'] for el in local_models_with_dlen]


    # mean_pre_metrics    = ', '.join(map(str, [mean_pre_DSCL_AVG,   mean_pre_DICE_AVG,  mean_pre_HD95_AVG   ]))
    # mean_post_metrics   = ', '.join(map(str, [mean_post_DSCL_AVG,  mean_post_DICE_AVG, mean_post_HD95_AVG  ]))

    prev_metrics    = {
        "PREV_mean_DSCL": mean_prev_DSCL_AVG,  
        **local_prev_DSCL_AVG,
        "PREV_mean_DICE": mean_prev_DICE_AVG,  
        **local_prev_DICE_AVG,
        "PREV_mean_HD95": mean_prev_HD95_AVG,  
        **local_prev_HD95_AVG,
    }
    post_metrics   = {
        "POST_mean_DSCL": mean_post_DSCL_AVG, 
        **local_post_DSCL_AVG, 
        "POST_mean_DICE": mean_post_DICE_AVG,  
        **local_post_DICE_AVG,
        "POST_mean_HD95": mean_post_HD95_AVG, 
        **local_post_HD95_AVG,
    }


    logs_dir = os.path.join('/','fedpod','logs')
    # logs_dir = os.path.join('.','logs')
    job_dir = os.path.join(logs_dir, f"{args.job_prefix}_{args.inst_id}")
    os.makedirs(job_dir, exist_ok=True)
    
    # 컬럼 헤더 정의
    header = ", ".join([
        "round", 
        *list(prev_metrics.keys()),
        *list(post_metrics.keys()),
        # "PREVmDSCL", "POSTmDSCL", 
        # "PREVmDICE", "POSTmDICE", 
        # "PREVmHD95", "POSTmHD95",
        # "\n",
    ])  # 실제 컬럼명에 맞게 수정하세요.

    # Pre-metrics 파일 작성
    pre_metrics_file = os.path.join(job_dir, f'{args.job_prefix}_metrics.csv')
    if not os.path.exists(pre_metrics_file):
        with open(pre_metrics_file, 'a') as f:
            f.write(f"{header}\n")  # 파일이 없으면 헤더 추가
    with open(pre_metrics_file, 'a') as f:
        f.write(
            ', '.join([
                f"{args.round:5d}",
                *[f"{el:14.4f}" for el in list(prev_metrics.values())],
                *[f"{el:14.4f}" for el in list(post_metrics.values())],
                # f"{mean_prev_metrics[0]:9.4f}", f"{mean_post_metrics[0]:9.4f}", 
                # f"{mean_prev_metrics[1]:9.4f}", f"{mean_post_metrics[1]:9.4f}", 
                # f"{mean_prev_metrics[2]:9.4f}", f"{mean_post_metrics[2]:9.4f}\n",
            ])+"\n"
        )

    # # Post-metrics 파일 작성
    # post_metrics_file = os.path.join(job_dir, f'{args.job_prefix}_post_metrics.csv')
    # if not os.path.exists(post_metrics_file):
    #     with open(post_metrics_file, 'a') as f:
    #         f.write(f"{header}\n")  # 파일이 없으면 헤더 추가
    #         # f.write(header + '\n')  # 파일이 없으면 헤더 추가
    # with open(post_metrics_file, 'a') as f:
    #     f.write(f"{args.round},\t{mean_post_metrics}\n")
    #     # f.write(str(args.round) + '\t' + mean_post_metrics + '\n')

if __name__ == '__main__': 
    args = parser.parse_args(sys.argv[1:])
    main(args)  # Call the main function with parsed arguments
    