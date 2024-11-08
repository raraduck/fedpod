argo submit kube-object/fedpod-test-sol1.yaml -n argo
while true; do
  STATUS=$(argo get @latest --no-color | grep Status: | awk '{print $2}')
  if [[ $STATUS == "Succeeded" ]]; then
    argo submit kube-object/fedpod-test-fed.yaml -n argo
    break
  fi
  sleep 10
done