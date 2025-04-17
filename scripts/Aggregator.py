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


def fedPOD(weights, models):
    first_model = models[0]  # 첫 번째 모델을 기준 구조로 사용

    # 파라미터 평균화를 위한 빈 껍데기 딕셔너리 생성
    agg_m = {key: torch.zeros_like(value, dtype=torch.float) for key, value in first_model.items()}

    # 모든 로컬 모델 파라미터의 합 계산
    for w, m in zip(weights, models):
        for key in agg_m.keys():
            agg_m[key] += m[key] * w
    return agg_m

def fedPID(weights, models):
    first_model = models[0]  # 첫 번째 모델을 기준 구조로 사용

    # 파라미터 평균화를 위한 빈 껍데기 딕셔너리 생성
    agg_m = {key: torch.zeros_like(value, dtype=torch.float) for key, value in first_model.items()}

    # 모든 로컬 모델 파라미터의 합 계산
    for w, m in zip(weights, models):
        for key in agg_m.keys():
            agg_m[key] += m[key] * w
    return agg_m