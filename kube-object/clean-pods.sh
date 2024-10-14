# kubectl get pods -n argo --field-selector=status.phase=Succeeded
kubectl delete pod -n argo --field-selector=status.phase=Succeeded
sudo rm -rf /fedpod/logs/* /fedpod/states/*

