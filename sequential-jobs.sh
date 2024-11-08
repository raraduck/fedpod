#!/bin/bash

kubectl create -f kube-object/fedpod-fed1J0R12.yaml -n argo

# WORKFLOW_NAME=$(kubectl get wf -o jsonpath='{.items[0].metadata.name}' -n argo)
WORKFLOW_NAME=$(kubectl get wf -n argo --sort-by='{.metadata.creationTimestamp}' -o jsonpath='{.items[-1:].metadata.name}')

# 첫 번째 워크플로우의 상태 확인
while true; do
  STATUS=$(kubectl get wf $WORKFLOW_NAME -o jsonpath='{.status.phase}' -n argo)
  if [[ $STATUS == "Succeeded" ]]; then
    # 두 번째 워크플로우 제출
    kubectl create -f kube-object/fedpod-sol1J6R12.yaml -n argo
    # 두 번째 워크플로우 이름 갱신
    WORKFLOW_NAME=$(kubectl get wf -n argo --sort-by='{.metadata.creationTimestamp}' -o jsonpath='{.items[-1:].metadata.name}')
    break
  fi
  sleep 300
done

# 워크플로우의 상태 확인
while true; do
  STATUS=$(kubectl get wf $WORKFLOW_NAME -o jsonpath='{.status.phase}' -n argo)
  if [[ $STATUS == "Succeeded" ]]; then
    # 워크플로우 제출
    kubectl create -f kube-object/fedpod-sol2J6R12.yaml -n argo
    # 워크플로우 이름 갱신
    WORKFLOW_NAME=$(kubectl get wf -n argo --sort-by='{.metadata.creationTimestamp}' -o jsonpath='{.items[-1:].metadata.name}')
    break
  fi
  sleep 300
done


# 워크플로우의 상태 확인
while true; do
  STATUS=$(kubectl get wf $WORKFLOW_NAME -o jsonpath='{.status.phase}' -n argo)
  if [[ $STATUS == "Succeeded" ]]; then
    # 워크플로우 제출
    kubectl create -f kube-object/fedpod-fed2J0R12.yaml -n argo
    # 워크플로우 이름 갱신
    WORKFLOW_NAME=$(kubectl get wf -n argo --sort-by='{.metadata.creationTimestamp}' -o jsonpath='{.items[-1:].metadata.name}')
    break
  fi
  sleep 300
done


# 워크플로우의 상태 확인
while true; do
  STATUS=$(kubectl get wf $WORKFLOW_NAME -o jsonpath='{.status.phase}' -n argo)
  if [[ $STATUS == "Succeeded" ]]; then
    # 워크플로우 제출
    kubectl create -f kube-object/fedpod-trf1J0R12.yaml -n argo
    # 워크플로우 이름 갱신
    WORKFLOW_NAME=$(kubectl get wf -n argo --sort-by='{.metadata.creationTimestamp}' -o jsonpath='{.items[-1:].metadata.name}')
    break
  fi
  sleep 300
done


# 워크플로우의 상태 확인
while true; do
  STATUS=$(kubectl get wf $WORKFLOW_NAME -o jsonpath='{.status.phase}' -n argo)
  if [[ $STATUS == "Succeeded" ]]; then
    # 워크플로우 제출
    kubectl create -f kube-object/fedpod-trf2J0R12.yaml -n argo
    # 워크플로우 이름 갱신
    WORKFLOW_NAME=$(kubectl get wf -n argo --sort-by='{.metadata.creationTimestamp}' -o jsonpath='{.items[-1:].metadata.name}')
    break
  fi
  sleep 300
done