import os
import shutil
import time
import argparse
import sys
import glob
import natsort
import json
from collections import OrderedDict
from utils.tools import *
from Aggregator import *
from utils.misc import *


parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--algorithm', type=str, default="fedavg", 
    choices=['fedavg', 'fedwavg', 'fedpid', 'fedpod'], help='type of avg')
parser.add_argument('--job_prefix', type=str, default="")
parser.add_argument('--inst_id', type=int, default=0)
parser.add_argument('--weight_path', type=str, required=True,
    help='path to pretrained encoder or decoder weight, None for train-from-scratch')


def fed_round_to_json(args, logger, local_dict, filename):
    logs_dir = os.path.join('/','fedpod','logs')
    job_dir = os.path.join(logs_dir, f"{args.job_prefix}_{args.inst_id}")
    os.makedirs(job_dir, exist_ok=True)

    # last-metrics 파일 작성
    metrics_file = os.path.join(job_dir, filename)

    # 기존 데이터 로드 또는 초기화
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as file:
            json_metrics_dict = json.load(file)
            # json_metrics_dict = json.load(file, object_pairs_hook=OrderedDict)
    else:
        json_metrics_dict = {}

    # local_last_dict의 각 job_name과 round_dict 순회
    for job_name, round_dict in local_dict.items():
        # job_name과 round_num 키가 없으면 자동으로 초기화
        job_dict = json_metrics_dict.setdefault(job_name, {})
        # round_dict 병합
        for round_num, metrics in round_dict.items():
            round_metrics = job_dict.setdefault(str(round_num), {})
        
            # 기존 데이터와 새로운 metrics 병합
            round_metrics.update(metrics)

    json_metrics_dict = OrderedDict(sorted(json_metrics_dict.items()))
    # 업데이트된 데이터를 JSON 파일에 저장
    with open(metrics_file, 'w', encoding='utf-8') as file:
        json.dump(json_metrics_dict, file, ensure_ascii=False, indent=4)

    # 로깅 정보 출력
    logger.info(f"Updated metrics saved to {metrics_file}")

def fed_print_to_csv(args, logger, local_models_with_dlen):
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
    prev_logger_metrics = {
        "PREV_mean_DSCL": mean_prev_DSCL_AVG, 
        "PREV_mean_DICE": mean_prev_DICE_AVG, 
        "PREV_mean_HD95": mean_prev_HD95_AVG,  
    }
    post_logger_metrics = { 
        "POST_mean_DSCL": mean_post_DSCL_AVG, 
        "POST_mean_DICE": mean_post_DICE_AVG,  
        "POST_mean_HD95": mean_post_HD95_AVG, 
    }
    for key1 in prev_logger_metrics.keys():
        prev_value = prev_logger_metrics[key1]
        key2 = "POST" + key1[4:]
        post_value = post_logger_metrics[key2]  # POST prefix 붙인 값 가져오기
        logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] {key1:>14} -> {key2:>14}: {prev_value:8.4f} -> {post_value:8.4f}")

    # # Post-metrics 파일 작성
    # post_metrics_file = os.path.join(job_dir, f'{args.job_prefix}_post_metrics.csv')
    # if not os.path.exists(post_metrics_file):
    #     with open(post_metrics_file, 'a') as f:
    #         f.write(f"{header}\n")  # 파일이 없으면 헤더 추가
    #         # f.write(header + '\n')  # 파일이 없으면 헤더 추가
    # with open(post_metrics_file, 'a') as f:
    #     f.write(f"{args.round},\t{mean_post_metrics}\n")
    #     # f.write(str(args.round) + '\t' + mean_post_metrics + '\n')

def solo_print_to_csv():
    pass

def solo_processing(args, base_dir, curr_round, next_round, logger):
    inst_dir = os.path.join(base_dir, f"{args.job_prefix}_{args.inst_id}") # inst0 also included 
    curr_round_dir = os.path.join(inst_dir, f"R{args.rounds:02}r{curr_round:02}")
    models_dir = os.path.join(curr_round_dir, 'models')
    best_path = os.path.join(models_dir, f"R{args.rounds:02}r{args.round:02}_best.pth")
    last_path = os.path.join(models_dir, f"R{args.rounds:02}r{args.round:02}_last.pth")
    assert os.path.exists(best_path), f"File not found: {best_path}"
    assert os.path.exists(last_path), f"File not found: {last_path}"
    assert args.rounds == next_round, f"solo post processing requires the end of rounds at the moment. Currently, next round is specified as {next_round} in case when total rounds is {args.rounds}"
    next_round_dir = os.path.join(inst_dir, f"R{args.rounds:02}r{next_round:02}")
    models_dir = os.path.join(next_round_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_best_path = os.path.join(models_dir, f"R{args.rounds:02}r{next_round:02}_best.pth")
    save_last_path = os.path.join(models_dir, f"R{args.rounds:02}r{next_round:02}_last.pth")
    # solo_print_to_csv(args, logger, local_models_with_dlen)
    # solo_print_to_csv(args, logger, local_models_with_dlen)
    shutil.copy2(best_path, save_best_path)
    shutil.copy2(last_path, save_last_path)
    logger.info(f"[{args.job_prefix.upper()}][SOLO] saved best model to {save_best_path}...")
    logger.info(f"[{args.job_prefix.upper()}][SOLO] saved last model to {save_last_path}...")
    return


def fed_processing(args, base_dir, curr_round, next_round, logger):
    inst_dir = os.path.join(base_dir, f"{args.job_prefix}_*") # inst0 also included 
    curr_round_dir = os.path.join(inst_dir, f"R{args.rounds:02}r{curr_round:02}")
    
    # prev to json
    prev_pattern = os.path.join(curr_round_dir, 'models', '*_prev.pth') # but, _last.pth removes inst0 because inst0 never has _last.pth file on it
    prev_pth_path = natsort.natsorted(glob.glob(prev_pattern))
    local_prev_dict = {
        state['args'].job_name: {
            state['args'].round: {
                'prev_metrics': state['pre_metrics']['DSCL_AVG']
            }
        } for el in prev_pth_path for state in [torch.load(el)]
    }
    fed_round_to_json(args, logger, local_prev_dict, f'{args.job_prefix}.json')


    # last to json
    last_pattern = os.path.join(curr_round_dir, 'models', '*_last.pth') # but, _last.pth removes inst0 because inst0 never has _last.pth file on it
    last_pth_path = natsort.natsorted(glob.glob(last_pattern))

    # local_last_dict = {state['args'].job_name: state for el in last_pth_path for state in [torch.load(el)]}
    local_last_dict = {
        state['args'].job_name: {
            state['args'].round: {
                'post_metrics': state['post_metrics']['DSCL_AVG']
            }
        } for el in last_pth_path for state in [torch.load(el)]
    }
    fed_round_to_json(args, logger, local_last_dict, f'{args.job_prefix}.json')


    # TODO: 여기서는 pth last와 prev 를 읽어서 cli_args 내 정보를 바탕으로 pandas 형태로 저장한 뒤 csv에 저장하기
    # 이후 round 에서도 csv를 읽을 때 pandas로 읽어들여서 column과 row를 관리해야함 (json으로 저장해서 dict 로 호환해도 됨)
    # pandas는 sql 형식이고, json은 nosql 형식임

    for pth in last_pth_path:
        logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] local models are from {pth}...")

    local_models_with_dlen = [torch.load(m) for m in last_pth_path]
    # local_last_dict = {torch.load(el)['args'].job_name: torch.load(el) for el in pth_path}
    fed_print_to_csv(args, logger, local_models_with_dlen)


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

    # save aggregated model with metrics
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
    next_round_dir = os.path.join(center_dir, f"R{args.rounds:02}r{next_round:02}")
    models_dir = os.path.join(next_round_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_model_path = os.path.join(models_dir, f"R{args.rounds:02}r{next_round:02}.pth")
    torch.save(state, save_model_path)
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] models are aggregated to {save_model_path}...")
    return

    #     save_model_path = os.path.join(models_dir, f"R{args.rounds:02}r{curr_round:02}.pth")
    #     shutil.copy2(orig_file, save_model_path)
    #     logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] initial model setup to {save_model_path}...")
    #     return

def init_processing(args, base_dir, curr_round, logger):
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] initial model setup from {args.weight_path}...")
    orig_file = args.weight_path
    inst_dir = os.path.join(base_dir, f"{args.job_prefix}_{args.inst_id}")
    curr_round_dir = os.path.join(inst_dir, f"R{args.rounds:02}r{curr_round:02}")
    models_dir = os.path.join(curr_round_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_model_path = os.path.join(models_dir, f"R{args.rounds:02}r{curr_round:02}.pth")
    shutil.copy2(orig_file, save_model_path)
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] initial model setup to {save_model_path}...")
    return


def main(args):
    log_filename = f"{args.job_prefix}_R{args.rounds:02}r{args.round:02}.log"
    logger = initialization_logger(args, log_filename)
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] aggregation algorithm is {args.algorithm.upper()}...")
    
    # prev_round = args.round - 1
    curr_round = args.round
    next_round = args.round + 1
    base_dir = os.path.join('/','fedpod','states')
    args.weight_path = None if args.weight_path == "None" else args.weight_path

    if args.weight_path:
        assert curr_round == 0, f"init_processing must be called at round 0, currently it is {curr_round}"
        init_processing(args, base_dir, curr_round, logger)
    else:
        if args.inst_id != 0:
            solo_processing(args, base_dir, curr_round, next_round, logger)
        else:
            fed_processing(args, base_dir, curr_round, next_round, logger)

    # 현재 라운드를 가져오고 1을 더한 후 두 자리 형식으로 변환
    next_round_formatted = f"{next_round:02d}"
    # /tmp/next_round.txt 파일에 저장
    with open("/tmp/next_round.txt", "w") as f:
        f.write(next_round_formatted)

    to_epoch = args.epoch + args.epochs
    next_epoch_formatted = f"{to_epoch:03d}"
    with open("/tmp/next_epoch.txt", "w") as f:
        f.write(next_epoch_formatted)

if __name__ == '__main__': 
    args = parser.parse_args(sys.argv[1:])
    main(args)  # Call the main function with parsed arguments
    