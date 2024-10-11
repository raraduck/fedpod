import torch
from copy import deepcopy

def fedavg(weights, models):
    first_model = models[0]  # 첫 번째 모델을 기준 구조로 사용

    # 파라미터 평균화를 위한 빈 껍데기 딕셔너리 생성
    agg_m = {key: torch.zeros_like(value, dtype=torch.float) for key, value in first_model.items()}

    # 모든 로컬 모델 파라미터의 합 계산
    for w, m in zip(weights, models):
        for key in agg_m.keys():
            agg_m[key] += m[key] * w
    return agg_m

def fedwavg(weights, models):
    first_model = models[0]  # 첫 번째 모델을 기준 구조로 사용

    # 파라미터 평균화를 위한 빈 껍데기 딕셔너리 생성
    agg_m = {key: torch.zeros_like(value, dtype=torch.float) for key, value in first_model.items()}

    # 모든 로컬 모델 파라미터의 합 계산
    for w, m in zip(weights, models):
        for key in agg_m.keys():
            agg_m[key] += m[key] * w
    return agg_m


# def fedwavg(path_list):
#     local_models_with_dlen = [torch.load(m) for m in path_list]
#     local_dlen = [el['P'] for el in local_models_with_dlen]
#     local_dprob = [el/sum(local_dlen) for el in local_dlen]
#     local_models = [el['model'] for el in local_models_with_dlen]
#     first_model = local_models[0]  # 첫 번째 모델을 기준 구조로 사용

#     # 파라미터 평균화를 위한 빈 껍데기 딕셔너리 생성
#     global_state_dict = {key: torch.zeros_like(value, dtype=torch.float) for key, value in first_model.items()}

#     # 모든 로컬 모델 파라미터의 합 계산
#     for dprob, model_state_dict in zip(local_dprob, local_models):
#         local_state_dict = model_state_dict
#         for key in global_state_dict.keys():
#             global_state_dict[key] += local_state_dict[key] * dprob
#     return global_state_dict