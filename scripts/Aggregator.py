import torch
from copy import deepcopy

def fedavg(path_list):
    local_models_with_dlen = [torch.load(m) for m in path_list]
    # local_dlen = [el['P'] for el in local_models_with_dlen]
    local_dlen = [1 for el in local_models_with_dlen]
    local_dprob = [el/sum(local_dlen) for el in local_dlen]
    local_models = [el['model'] for el in local_models_with_dlen]
    first_model = local_models[0]  # 첫 번째 모델을 기준 구조로 사용

    # 파라미터 평균화를 위한 빈 껍데기 딕셔너리 생성
    global_state_dict = {key: torch.zeros_like(value, dtype=torch.float) for key, value in first_model.items()}

    # 모든 로컬 모델 파라미터의 합 계산
    for dprob, model_state_dict in zip(local_dprob, local_models):
        local_state_dict = model_state_dict
        for key in global_state_dict.keys():
            global_state_dict[key] += local_state_dict[key] * dprob
    return global_state_dict

# def fedavg(local_models_with_dlen):
#     """ 모든 로컬 모델의 파라미터를 평균화합니다. """
#     local_dlen = [1.0 for el in local_models_with_dlen]
#     local_dprob = [el/sum(local_dlen) for el in local_dlen]
#     local_models = [el[1] for el in local_models_with_dlen]
#     first_model = local_models[0]  # 첫 번째 모델을 기준 구조로 사용

#     # 파라미터 평균화를 위한 빈 껍데기 딕셔너리 생성
#     global_state_dict = {key: torch.zeros_like(value, dtype=torch.float) for key, value in first_model.state_dict().items()}

#     # 모든 로컬 모델 파라미터의 합 계산
#     for dprob, model in zip(local_dprob, local_models):
#         local_state_dict = model.state_dict()
#         for key in global_state_dict.keys():
#             global_state_dict[key] += local_state_dict[key] * dprob
#     return global_state_dict

