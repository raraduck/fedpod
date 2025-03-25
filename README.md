# FedPOD
### requirements
```
python > 3.7.9
dvc[s3] == 2.10.2
nibabel == 4.0.2
numpy == 1.21.6
```
```
### packages installation
```shell
> python --version 3.7.9
> pip install SimpleITK==2.2.1
> pip install jupyterlab medpy matplotlib natsort nibabel numpy pandas pillow tensorboard torch tqdm torchsummary monai
```
### run container
```
> docker run -v ./cc359ppmi128:/fedpod/cc359ppmi128 -v ./states/R00:/fedpod/states -it fedpod:v0.3 ./run_train.sh -j inst_01 -R 1 -r 0 -E 1 -i 1
```

### install kubernetes 1.24 on ubuntu 18.04
#### SWAP-On 시 퍼포먼스 이슈가 있어 일반적으로 제거
```
sudo swapoff /swap.img
sudo sed -i -e '/swap.img/d' /etc/fstab
```
#### Pre-Setting – All node
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl enable --now docker && sudo systemctl status docker --no-pager
sudo usermod -aG docker worker
sudo docker container ls

# cri-docker Install
VER=$(curl -s https://api.github.com/repos/Mirantis/cri-dockerd/releases/latest|grep tag_name | cut -d '"' -f 4|sed 's/v//g')
echo $VER
wget https://github.com/Mirantis/cri-dockerd/releases/download/v${VER}/cri-dockerd-${VER}.amd64.tgz
tar xvf cri-dockerd-${VER}.amd64.tgz
sudo mv cri-dockerd/cri-dockerd /usr/local/bin/

# cri-docker Version Check
cri-dockerd --version

wget https://raw.githubusercontent.com/Mirantis/cri-dockerd/master/packaging/systemd/cri-docker.service
wget https://raw.githubusercontent.com/Mirantis/cri-dockerd/master/packaging/systemd/cri-docker.socket
sudo mv cri-docker.socket cri-docker.service /etc/systemd/system/
sudo sed -i -e 's,/usr/bin/cri-dockerd,/usr/local/bin/cri-dockerd,' /etc/systemd/system/cri-docker.service

sudo systemctl daemon-reload
sudo systemctl enable cri-docker.service
sudo systemctl enable --now cri-docker.socket

# cri-docker Active Check
sudo systemctl restart docker && sudo systemctl restart cri-docker
sudo systemctl status cri-docker.socket --no-pager 

# Docker cgroup Change Require to Systemd
sudo mkdir /etc/docker
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2"
}
EOF

sudo systemctl restart docker && sudo systemctl restart cri-docker
sudo docker info | grep Cgroup

# Kernel Forwarding 
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
br_netfilter
EOF

cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
EOF

sudo sysctl --system
```
#### Packages Install – All node
```
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl
sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
```
##### 이부분 교체 (kubernetes v1.24 부터 변경)
```
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list

>>>변경>>>

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.24/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.24/deb/ /" | sudo tee /etc/apt/sources.list.d/kubernetes.list
```

##### Update 해야 패키지 인식함
```
sudo apt-get update
```

##### k8s 설치
```
sudo apt-get install -y kubelet kubeadm kubectl
```

##### 버전 확인하기
```
kubectl version --short

Client Version: v1.24.3
Kustomize Version: v4.5.4
```

##### 버전 고정하기
```
sudo apt-mark hold kubelet kubeadm kubectl
```

##### 명령 실행 방법 (example)
> cd ~/fedpod/kube-object/fets
```
argo -n argo submit fed-v0.4.18.08-from00to04.yaml --parameter-file params/FedPODnSel2.yaml -p job-prefix="fedpodnsel2"
```