#!/bin/bash

kubectl create -f kube-object/fedpod-test-sol1.yaml -n argo

WORKFLOW_NAME=$(kubectl get wf -o jsonpath='{.items[0].metadata.name}' -n argo)

while true; do
  STATUS=$(kubectl get wf $WORKFLOW_NAME -o jsonpath='{.status.phase}' -n argo)	
  if [[ $STATUS == "Succeeded" ]]; then
    kubectl create -f kube-object/fedpod-test-fed.yaml -n argo
    break
  fi
  sleep 10
done
