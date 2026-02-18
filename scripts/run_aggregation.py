import os
import shutil
import time
import argparse
import sys
import glob
import natsort
import json
import re
from torch.utils.tensorboard import SummaryWriter
from utils.tools import *
from Aggregator import *
from utils.misc import *


parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--cases_split', type=str, default=None, help='file path for split csv')
parser.add_argument('--algorithm', type=str, default="fedavg", 
    choices=['fedavg', 'fedwavg', 'fedpid', 'fedpod', 'fedprox'], help='type of avg')
parser.add_argument('--job_prefix', type=str, default="")
parser.add_argument('--inst_id', type=int, default=0)
parser.add_argument('--weight_path', type=str, required=True,
    help='path to pretrained encoder or decoder weight, None for train-from-scratch')


def fed_round_to_json(args, logger, local_dict, filename):
    logs_dir = os.path.join('/', 'fedpod', 'states')
    job_dir = os.path.join(logs_dir, f"{args.job_prefix}_{args.inst_id}")
    os.makedirs(job_dir, exist_ok=True)

    # last-metrics 파일 작성
    metrics_file = os.path.join(job_dir, filename) # f'{args.job_prefix}.json'

    # 기존 데이터 로드 또는 초기화
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as file:
            json_metrics_dict = json.load(file)
    else:
        json_metrics_dict = {}

    json_metrics_dict[str(args.round)] = local_dict

    # 업데이트된 데이터를 JSON 파일에 저장
    with open(metrics_file, 'w', encoding='utf-8') as file:
        json.dump(json_metrics_dict, file, ensure_ascii=False, indent=4)

    # 로깅 정보 출력
    logger.info(f"Updated metrics saved to {metrics_file}")
    return json_metrics_dict

def fed_processing(args, base_dir, base_logs_dir, curr_round, next_round, logger, final_status):
    # config for log files
    inst_logs_dir = os.path.join(base_logs_dir, f"{args.job_prefix}") # inst0 also included 
    curr_inst_pattern = os.path.join(inst_logs_dir, f"{args.job_prefix}_*_R{args.rounds:02}r{curr_round:02}.log")
    curr_inst_log_path = natsort.natsorted(glob.glob(curr_inst_pattern))
    trn_pattern = re.compile(r'\[TRN\].*?N:\s*[\(\[](.*?)[\)\]]')

    trn_files_json_path = os.path.join(inst_logs_dir, f"{args.job_prefix}.json")
    # 기존 JSON 파일 불러오기 (없으면 빈 dict로 시작)
    if os.path.exists(trn_files_json_path):
        with open(trn_files_json_path, 'r', encoding='utf-8') as f:
            preserved_dict = json.load(f)
    else:
        preserved_dict = {}

    # trn_files_dict = {}
    for log_file_path in curr_inst_log_path:
        file_list = []
        log_file_name = os.path.basename(log_file_path)
        head_of_log_file_name = log_file_name.split('.log')[0]
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '[TRN]' in line:
                    match = trn_pattern.search(line)
                    if match:
                        # 리스트 안의 파일들을 ,로 구분된 문자열로 간주하고 따옴표 제거
                        filenames = [name.strip().strip("'\"") for name in match.group(1).split(',')]
                        file_list.extend(filenames)
        # 중복 제거 (선택사항)
        file_list = sorted(set(file_list))
        preserved_dict[head_of_log_file_name] = file_list
        # trn_files_dict.update(f"{head_of_log_file_name}", file_list)
        logger.info(f"[{args.job_prefix.upper()}][{log_file_name}] trained files {file_list}...")
    # JSON으로 저장
    with open(trn_files_json_path, 'w', encoding='utf-8') as f:
        json.dump(preserved_dict, f, indent=2)

    if final_status:
        logger.info(f"[{args.job_prefix.upper()}][{head_of_log_file_name}] final_statues is TRUE and save json to csv...")
        # args.cases_split 경로에서 csv 파일을 읽어서 R1...마지막 Round까지 csv 에 채워서 logs 경로에 csv 새로 저장하기
        split_path = args.cases_split
        df = pd.read_csv(split_path)
        #  "fedavgniid2-node6-0415_6_R20r01
        # Update the DataFrame based on the JSON
        df["Subject_ID"] = df["Subject_ID"].astype(str).str.strip()
        for key, subject_list in preserved_dict.items():
            match = re.search(r'_R\d{2}r(\d+)', key)
            if match:
                round_idx = int(match.group(1))
                col_name = f"R{round_idx}"
                for subject_id in subject_list:
                    subject_id = subject_id.strip()
                    if subject_id:
                        df.loc[df["Subject_ID"] == subject_id, col_name] = subject_id
        # Save the updated DataFrame to a new CSV file
        output_path = os.path.join(inst_logs_dir, f"{args.job_prefix}.csv")
        df.to_csv(output_path, index=False)
    

    # config for pth files
    inst_dir = os.path.join(base_dir, f"{args.job_prefix}_*") # inst0 also included 
    curr_round_dir = os.path.join(inst_dir, f"R{args.rounds:02}r{curr_round:02}")
    
    # prev to json
    prev_pattern = os.path.join(curr_round_dir, 'models', '*_prev.pth') # but, _last.pth removes inst0 because inst0 never has _last.pth file on it
    prev_pth_path = natsort.natsorted(glob.glob(prev_pattern))
    assert len(prev_pth_path) > 0, f"*_prev.pth does not exist"
    local_dict = {
        state['args'].job_name: {
            'prev': state['pre_metrics']
        }
        for el in prev_pth_path for state in [torch.load(el)]
    }
    # DSCL_AVG 값을 저장할 리스트 초기화
    # avg_values = {'DSCL':[], 'DICE':[], 'HD95':[]}
    mean_values = {k: [] for k, v in list(local_dict.values())[0]['prev'].items()} 

    # local_dict를 순회하며 DSCL_AVG 값 추출
    for job_info in local_dict.values():
        # avg_values['DSCL'].append(job_info['prev']['DSCL_AVG'])
        # avg_values['DICE'].append(job_info['prev']['DICE_AVG'])
        # avg_values['HD95'].append(job_info['prev']['HD95_AVG'])
        for k, v in job_info['prev'].items():
            mean_values[k].append(v)

    local_dict = {
        **local_dict,
        f"{args.job_prefix}_{args.inst_id}": {
            # 'prev': {
            #     'DSCL_AVG': sum(avg_values['DSCL']) / len(avg_values['DSCL']),
            #     'DICE_AVG': sum(avg_values['DICE']) / len(avg_values['DICE']),
            #     'HD95_AVG': sum(avg_values['HD95']) / len(avg_values['HD95'])
            # },
            'prev':{k: sum(v)/len(v) for k, v in mean_values.items()} 
        }
    }

    # last to json
    last_pattern = os.path.join(curr_round_dir, 'models', '*_last.pth') # but, _last.pth removes inst0 because inst0 never has _last.pth file on it
    last_pth_path = natsort.natsorted(glob.glob(last_pattern))
    assert len(last_pth_path) > 0, f"*_last.pth does not exist"

    for el in last_pth_path:
        state = torch.load(el)
        job_name = state['args'].job_name
        local_dict[job_name].update({
            'post': state['post_metrics']
        })
    # DSCL_AVG 값을 저장할 리스트 초기화
    # avg_values = {'DSCL':[], 'DICE':[], 'HD95':[]}
    mean_values = {k: [] for k, v in list(local_dict.values())[0]['prev'].items()} 

    # local_dict를 순회하며 DSCL_AVG 값 추출
    for job_info in local_dict.values():
        if 'post' in job_info:
            # avg_values['DSCL'].append(job_info['post']['DSCL_AVG'])
            # avg_values['DICE'].append(job_info['post']['DICE_AVG'])
            # avg_values['HD95'].append(job_info['post']['HD95_AVG'])
            for k, v in job_info['post'].items():
                mean_values[k].append(v)
            
    # if len(last_pth_path) > 0:
    local_dict = {
        **local_dict,
        f"{args.job_prefix}_{args.inst_id}": {
            **local_dict[f"{args.job_prefix}_{args.inst_id}"],
            # 'post':{
            #     'DSCL_AVG': sum(avg_values['DSCL']) / len(avg_values['DSCL']),
            #     'DICE_AVG': sum(avg_values['DICE']) / len(avg_values['DICE']),
            #     'HD95_AVG': sum(avg_values['HD95']) / len(avg_values['HD95'])
            # },
            'post':{k: sum(v)/len(v) for k, v in mean_values.items()} 
        }
    }
        # local_dict[f"{args.job_prefix}_{args.inst_id}"] = {
        #     'post':{
        #         'DSCL_AVG': sum(avg_values['DSCL']) / len(avg_values['DSCL']),
        #         'DICE_AVG': sum(avg_values['DICE']) / len(avg_values['DICE']),
        #         'HD95_AVG': sum(avg_values['HD95']) / len(avg_values['HD95'])
        #     }
        # }

    json_metrics_dict = fed_round_to_json(args, logger, local_dict, f'{args.job_prefix}.json')
    
    for jobname, job_dict in local_dict.items():
        writer = SummaryWriter(os.path.join('runs', args.job_prefix, jobname))
        for prev_post, metric_dict in job_dict.items():
            for metric_name, value in metric_dict.items():
                writer.add_scalar(f"{prev_post}/{metric_name}", value, args.round)
        writer.flush()
        writer.close()

    for pth in last_pth_path:
        logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] local models are from {pth}...")

    local_models_with_dlen = [torch.load(m) for m in last_pth_path]
    ## TODO: 이 부분을 local_models_with_dlen에 의존하지 않고 
    ## json_metrics_dict[str(args.round)]의 key 만 list로 추출하여 사용할 수 있음
    JOB_NAME = [el['args'].job_name for el in local_models_with_dlen]
    ## TODO: 또한 PID 정보를 이전 round 에서도 추출할 수 있음
    ## 예를 들면, json_metrics_dict[str(args.round-1)] 으로 접근하면 이전 post와 현재 post의 PID 추출 가능
    ## 하지만, round 1일 때에는 하면 안되고, round 2부터 해야함
    ## FedPID의 경우에는 기존 PID를 추출하면 안됨, (기존 PID는 post-prev 와 같이 이미 계산이 완료된 것들임)
    ## 
    ## fedpid 일 경우에만 
    # json_metrics_dict[str(args.round)], json_metrics_dict[str(args.round-1)] 접근하여 
    # post 정보 추출하기
    # 참고용 정보
    # state = {
    #         'model': train_setup['model'].state_dict(), 
    #         'args': self.cli_args,
    #         'train_tb_dict': train_tb_dict,
    #         'pre_metrics': pre_metrics,
    #         'post_metrics': post_metrics,
    #         'P': train_setup['train_loader'].dataset.__len__(),
    #         'I': (Pre_LOSS + Post_LOSS)/2,
    #         'D': (Pre_LOSS - Post_LOSS),
    #         'time': time.time() - time_in_total,
    #     }
    # torch.save(state, os.path.join(save_model_path, f"R{self.cli_args.rounds:02}r{self.cli_args.round:02}_last.pth"))

    if args.algorithm == "fedavg":
        P = [1 for el in local_models_with_dlen]
        W = [p/sum(P) for p in P]
        M = [el['model'] for el in local_models_with_dlen]
        aggregated_model = fedavg(W, M)
        for p, w, j in zip(P, W, JOB_NAME):
            logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}][{j}][P,W][{p:.2f},{w:.2f}]")
    elif args.algorithm == "fedprox":
        # FedProx의 Aggregation은 FedAvg와 동일 (가중 평균)
        P = [el['P'] for el in local_models_with_dlen] # 데이터 개수 비례
        W = [p/sum(P) for p in P]
        M = [el['model'] for el in local_models_with_dlen]
        
        # 별도의 fedprox 함수를 만들 필요 없이 fedwavg 로직 재사용
        aggregated_model = fedwavg(W, M) 
        
        for p, w, j in zip(P, W, JOB_NAME):
            logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}][{j}][P,W][{p:.2f},{w:.2f}]")
    elif args.algorithm == "fedwavg":
        P = [el['P'] for el in local_models_with_dlen]
        W = [p/sum(P) for p in P]
        M = [el['model'] for el in local_models_with_dlen]
        aggregated_model = fedwavg(W, M)
        for p, w, j in zip(P, W, JOB_NAME):
            logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}][{j}][P,W][{p:.2f},{w:.2f}]")
    elif args.algorithm == "fedpod":
        P = [el['P'] for el in local_models_with_dlen]
        I = [el['I'] for el in local_models_with_dlen]
        D = [el['D'] for el in local_models_with_dlen]
        if sum(D) == 0:
            if sum(I) == 0:
                W = [p/sum(P) for p in P]
                logger.warn(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] I or D term is zero")
            else:
                alpha = 0.8
                beta = 0.2
                # gamma = 0.7
                W = [alpha*p/sum(P) + beta*i/sum(I) for p, i, d in zip(P, I, D)]
                logger.warn(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] D term is zero")
        else:
            alpha = 0.2
            beta = 0.1
            gamma = 0.7
            W = [alpha*p/sum(P) + beta*i/sum(I) + gamma*d/sum(D) for p, i, d in zip(P, I, D)]
        M = [el['model'] for el in local_models_with_dlen]
        aggregated_model = fedPOD(W, M)
        for p, i, d, w, j in zip(P, I, D, W, JOB_NAME):
            logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}][{j}][P,I,D,W][{p:.2f},{i:.2f},{d:.2f},{w:.2f}]")
    elif args.algorithm == "fedpid":
        # json_metrics_dict[str(args.round-1)]
        # json_metrics_dict[str(args.round)]
        # 에서 Client Selection을 적용한 경우에는 client 수가 변했을 때 문제가 발생함
        # 따라서 pid 는 all client participation 상황이어야함
        if args.round < 2: # round 1일 때는 round 0에서 PID를 추출할 수 없으므로 round 2부터 FedPID를 적용해야함
            P = [el['P'] for el in local_models_with_dlen]
            W = [p/sum(P) for p in P]
            M = [el['model'] for el in local_models_with_dlen]
            aggregated_model = fedwavg(W, M)
            for p, w, j in zip(P, W, JOB_NAME):
                logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}][{j}][P,W][{p:.2f},{w:.2f}]")
        else:
            local_metric_at_round_prev = json_metrics_dict[str(args.round-1)]
            local_metric_at_round_post = json_metrics_dict[str(args.round)]
            # DSCL_AVG_prev  = [float(el['post']['DSCL_AVG']) for (job_name, el) in local_metric_at_round_prev.items()]
            DSCL_AVG_prev = [float(local_metric_at_round_prev[el]['post']['DSCL_AVG']) for el in JOB_NAME]
            # DSCL_AVG_post  = [float(el['post']['DSCL_AVG']) for (job_name, el) in local_metric_at_round_post.items()]
            DSCL_AVG_post = [float(local_metric_at_round_post[el]['post']['DSCL_AVG']) for el in JOB_NAME]

            P = [el['P'] for el in local_models_with_dlen]
            I = [(prev+post)/2 for (prev, post) in zip(DSCL_AVG_prev, DSCL_AVG_post)] # (DSCL_AVG_prev + DSCL_AVG_post)/2
            D = [max(0, prev - post) for (prev, post) in zip(DSCL_AVG_prev, DSCL_AVG_post)] # max(0, DSCL_AVG_prev - DSCL_AVG_post)
            if sum(D) == 0:
                if sum(I) == 0:
                    W = [p/sum(P) for p in P]
                    logger.warn(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] I or D term is zero")
                else:
                    alpha = 0.8
                    beta = 0.2
                    # gamma = 0.7
                    W = [alpha*p/sum(P) + beta*i/sum(I) for p, i, d in zip(P, I, D)]
                    logger.warn(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] D term is zero")
            else:
                alpha = 0.2
                beta = 0.1
                gamma = 0.7
                W = [alpha*p/sum(P) + beta*i/sum(I) + gamma*d/sum(D) for p, i, d in zip(P, I, D)]
            M = [el['model'] for el in local_models_with_dlen]
            aggregated_model = fedPID(W, M)
            for p, i, d, w, j in zip(P, I, D, W, JOB_NAME):
                logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}][{j}][P,I,D,W][{p:.2f},{i:.2f},{d:.2f},{w:.2f}]")
    else:
        raise NotImplementedError(f"{args.algorithm.upper()} is not implemented on Aggregator()")


    # save aggregated model with metrics
    state = {
        'model': aggregated_model, 
    }
    center_dir = os.path.join(base_dir, f"{args.job_prefix}_{args.inst_id}")
    next_round_dir = os.path.join(center_dir, f"R{args.rounds:02}r{next_round:02}")
    models_dir = os.path.join(next_round_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_model_path = os.path.join(models_dir, f"R{args.rounds:02}r{next_round:02}_agg.pth")
    torch.save(state, save_model_path)
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] models are aggregated to {save_model_path}...")
    return

def init_processing(args, base_dir, curr_round, logger):
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] initial model setup from {args.weight_path}...")
    orig_file = args.weight_path
    inst_dir = os.path.join(base_dir, f"{args.job_prefix}_{args.inst_id}")
    curr_round_dir = os.path.join(inst_dir, f"R{args.rounds:02}r{curr_round:02}")
    models_dir = os.path.join(curr_round_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_model_path = os.path.join(models_dir, f"R{args.rounds:02}r{curr_round:02}_agg.pth")
    shutil.copy2(orig_file, save_model_path)
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] initial model setup to {save_model_path}...")
    return


def main(args):
    log_filename = f"{args.job_prefix}_R{args.rounds:02}r{args.round:02}.log"
    logger = initialization_logger(args, args.job_prefix, log_filename)
    logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] aggregation algorithm is {args.algorithm.upper()}...")

    curr_round = args.round
    next_round = args.round + 1
    final_status = (args.rounds == next_round)
    base_dir = os.path.join('/','fedpod','states')
    base_logs_dir = os.path.join('/','fedpod','logs')
    args.weight_path = None if args.weight_path == "None" else args.weight_path

    if args.weight_path:
        assert curr_round == 0, f"init_processing must be called at round 0, currently it is {curr_round}"
        init_processing(args, base_dir, curr_round, logger)
    else:
        fed_processing(args, base_dir, base_logs_dir, curr_round, next_round, logger, final_status)

    # 현재 라운드를 가져오고 1을 더한 후 두 자리 형식으로 변환
    next_round_formatted = f"{next_round:02d}"
    with open("/tmp/next_round.txt", "w") as f:
        f.write(next_round_formatted)

    to_epoch = args.epoch + args.epochs
    next_epoch_formatted = f"{to_epoch:03d}"
    with open("/tmp/next_epoch.txt", "w") as f:
        f.write(next_epoch_formatted)

if __name__ == '__main__': 
    args = parser.parse_args(sys.argv[1:])
    main(args)
    



# def fed_print_to_csv(args, logger, local_models_with_dlen):
#     # averaged_loss = lossavg()
#     mean_prev_DSCL_AVG = np.mean([el['pre_metrics']['DSCL_AVG'] for el in local_models_with_dlen])
#     mean_prev_DICE_AVG = np.mean([el['pre_metrics']['DICE_AVG'] for el in local_models_with_dlen])
#     mean_prev_HD95_AVG = np.mean([el['pre_metrics']['HD95_AVG'] for el in local_models_with_dlen])
    
#     local_prev_DSCL_AVG = {
#         f"PREV_{int(el['args'].job_name.split('_')[-1]):04d}_DSCL": np.mean([
#                 v for (k,v) in el['pre_metrics'].items() if 'DSCL' in k
#                 # el['pre_metrics']['DSCL_LA'],
#                 # el['pre_metrics']['DSCL_LC'],
#                 # el['pre_metrics']['DSCL_LP'],
#                 # el['pre_metrics']['DSCL_RA'],
#                 # el['pre_metrics']['DSCL_RC'],
#                 # el['pre_metrics']['DSCL_RP'],
#         ]) for el in local_models_with_dlen
#     }
#     local_prev_DICE_AVG = {
#         f"PREV_{int(el['args'].job_name.split('_')[-1]):04d}_DICE": np.mean([
#                 v for (k,v) in el['pre_metrics'].items() if 'DICE' in k
#         ]) for el in local_models_with_dlen
#     }
#     local_prev_HD95_AVG = {
#         f"PREV_{int(el['args'].job_name.split('_')[-1]):04d}_HD95": np.mean([
#                 v for (k,v) in el['pre_metrics'].items() if 'HD95' in k
#         ]) for el in local_models_with_dlen
#     }
#     # local_pre_DSCL_LA = np.mean([el['pre_metrics']['DSCL_LA'] for el in local_models_with_dlen])
#     # local_pre_DSCL_LC = np.mean([el['pre_metrics']['DSCL_LC'] for el in local_models_with_dlen])
#     # local_pre_DSCL_LP = np.mean([el['pre_metrics']['DSCL_LP'] for el in local_models_with_dlen])
#     # local_pre_DSCL_RA = np.mean([el['pre_metrics']['DSCL_RA'] for el in local_models_with_dlen])
#     # local_pre_DSCL_RC = np.mean([el['pre_metrics']['DSCL_RC'] for el in local_models_with_dlen])
#     # local_pre_DSCL_RP = np.mean([el['pre_metrics']['DSCL_RP'] for el in local_models_with_dlen])

#     mean_post_DSCL_AVG = np.mean([el['post_metrics']['DSCL_AVG'] for el in local_models_with_dlen])
#     mean_post_DICE_AVG = np.mean([el['post_metrics']['DICE_AVG'] for el in local_models_with_dlen])
#     mean_post_HD95_AVG = np.mean([el['post_metrics']['HD95_AVG'] for el in local_models_with_dlen])

#     local_post_DSCL_AVG = {
#         f"POST_{int(el['args'].job_name.split('_')[-1]):04d}_DSCL": np.mean([
#                 v for (k,v) in el['post_metrics'].items() if 'DSCL' in k
#                 # el['post_metrics']['DSCL_LA'],
#                 # el['post_metrics']['DSCL_LC'],
#                 # el['post_metrics']['DSCL_LP'],
#                 # el['post_metrics']['DSCL_RA'],
#                 # el['post_metrics']['DSCL_RC'],
#                 # el['post_metrics']['DSCL_RP'],
#         ]) for el in local_models_with_dlen
#     }
#     local_post_DICE_AVG = {
#         f"POST_{int(el['args'].job_name.split('_')[-1]):04d}_DICE": np.mean([
#                 v for (k,v) in el['post_metrics'].items() if 'DICE' in k
#         ]) for el in local_models_with_dlen
#     }
#     local_post_HD95_AVG = {
#         f"POST_{int(el['args'].job_name.split('_')[-1]):04d}_HD95": np.mean([
#                 v for (k,v) in el['post_metrics'].items() if 'HD95' in k
#         ]) for el in local_models_with_dlen
#     }
#     # local_pre_DSCL = [el['pre_metrics']['DSCL_AVG'] for el in local_models_with_dlen]


#     # mean_pre_metrics    = ', '.join(map(str, [mean_pre_DSCL_AVG,   mean_pre_DICE_AVG,  mean_pre_HD95_AVG   ]))
#     # mean_post_metrics   = ', '.join(map(str, [mean_post_DSCL_AVG,  mean_post_DICE_AVG, mean_post_HD95_AVG  ]))

#     prev_metrics    = {
#         "PREV_mean_DSCL": mean_prev_DSCL_AVG,  
#         **local_prev_DSCL_AVG,
#         "PREV_mean_DICE": mean_prev_DICE_AVG,  
#         **local_prev_DICE_AVG,
#         "PREV_mean_HD95": mean_prev_HD95_AVG,  
#         **local_prev_HD95_AVG,
#     }
#     post_metrics   = {
#         "POST_mean_DSCL": mean_post_DSCL_AVG, 
#         **local_post_DSCL_AVG, 
#         "POST_mean_DICE": mean_post_DICE_AVG,  
#         **local_post_DICE_AVG,
#         "POST_mean_HD95": mean_post_HD95_AVG, 
#         **local_post_HD95_AVG,
#     }


#     logs_dir = os.path.join('/','fedpod','logs')
#     # logs_dir = os.path.join('.','logs')
#     job_dir = os.path.join(logs_dir, f"{args.job_prefix}_{args.inst_id}")
#     os.makedirs(job_dir, exist_ok=True)
    
#     # 컬럼 헤더 정의
#     header = ", ".join([
#         "round", 
#         *list(prev_metrics.keys()),
#         *list(post_metrics.keys()),
#         # "PREVmDSCL", "POSTmDSCL", 
#         # "PREVmDICE", "POSTmDICE", 
#         # "PREVmHD95", "POSTmHD95",
#         # "\n",
#     ])  # 실제 컬럼명에 맞게 수정하세요.

#     # Pre-metrics 파일 작성
#     pre_metrics_file = os.path.join(job_dir, f'{args.job_prefix}_metrics.csv')
#     if not os.path.exists(pre_metrics_file):
#         with open(pre_metrics_file, 'a') as f:
#             f.write(f"{header}\n")  # 파일이 없으면 헤더 추가
#     with open(pre_metrics_file, 'a') as f:
#         f.write(
#             ', '.join([
#                 f"{args.round:5d}",
#                 *[f"{el:14.4f}" for el in list(prev_metrics.values())],
#                 *[f"{el:14.4f}" for el in list(post_metrics.values())],
#                 # f"{mean_prev_metrics[0]:9.4f}", f"{mean_post_metrics[0]:9.4f}", 
#                 # f"{mean_prev_metrics[1]:9.4f}", f"{mean_post_metrics[1]:9.4f}", 
#                 # f"{mean_prev_metrics[2]:9.4f}", f"{mean_post_metrics[2]:9.4f}\n",
#             ])+"\n"
#         )
#     prev_logger_metrics = {
#         "PREV_mean_DSCL": mean_prev_DSCL_AVG, 
#         "PREV_mean_DICE": mean_prev_DICE_AVG, 
#         "PREV_mean_HD95": mean_prev_HD95_AVG,  
#     }
#     post_logger_metrics = { 
#         "POST_mean_DSCL": mean_post_DSCL_AVG, 
#         "POST_mean_DICE": mean_post_DICE_AVG,  
#         "POST_mean_HD95": mean_post_HD95_AVG, 
#     }
#     for key1 in prev_logger_metrics.keys():
#         prev_value = prev_logger_metrics[key1]
#         key2 = "POST" + key1[4:]
#         post_value = post_logger_metrics[key2]  # POST prefix 붙인 값 가져오기
#         logger.info(f"[{args.job_prefix.upper()}][{args.algorithm.upper()}] {key1:>14} -> {key2:>14}: {prev_value:8.4f} -> {post_value:8.4f}")
