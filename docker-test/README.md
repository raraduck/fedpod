# Install Minikube

## Prerequisites
- Linux
- Latest NVIDIA GPU drivers
- minikube v1.32.0-beta.0 or later (docker driver only)
## 0. 보안업데이트 중단 (apt lock 방지)
```bash
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer
```
## 1. Install Docker using the apt repository
### 1.1. Set up Docker's apt repository.
```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings # 키링 설치
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc # 도커 공식 GPG 키 다운
sudo chmod a+r /etc/apt/keyrings/docker.asc # 키 권한 변경

# Add the repository to Apt sources (리포지토리 추가):
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
### 1.2. Install the Docker packages.
```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl status docker
sudo systemctl start docker

sudo usermod -aG docker $USER
docker ps
```

## 2 Configuring Docker (Using the docker driver)
[cuda1]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
Ensure you have an NVIDIA driver installed, you can check if one is installed by running nvidia-smi, if one is not installed follow the [**NVIDIA Driver Installation Guide**][cuda1]
### 2.1. Check if bpf_jit_harden is set to 0
```bash
sudo sysctl net.core.bpf_jit_harden
```
#### 2.1.1. If it’s not 0 run:
```bash
echo "net.core.bpf_jit_harden=0" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```
- Install the [**NVIDIA Container Toolkit**][docker1] on your host machine
- or
- Enable [**NVIDIA CDI resources**][docker2] on your host machine.

[docker1]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[docker2]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
## 3. install docker for NVIDIA Container Toolkit (With apt: Ubuntu, Debian) [link][docker1]
### 3.1. Configure the production repository: 
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
### 3.2. Update the packages list from the repository:
```bash
sudo apt-get update
```
### 3.3. Install the NVIDIA Container Toolkit packages:
```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```
### 3.4. Configure the container runtime by using the nvidia-ctk command:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
# INFO[0000] Config file does not exist; using empty config
# INFO[0000] Wrote updated config to /etc/docker/daemon.json
# INFO[0000] It is recommended that docker daemon be restarted.
```
The **nvidia-ctk** command modifies the **/etc/docker/daemon.json** file on the host. The file is updated so that Docker can use the NVIDIA Container Runtime.
### 3.5. Restart the Docker daemon:
```bash
sudo systemctl restart docker
```
### 3.6. Run a sample CUDA container:
```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## 4. Install Minikube (v1.32.0-beta.0 or later)
To install the latest minikube **stable** release on **x86-64 Linux** using **Debian package**:
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo dpkg -i minikube_latest_amd64.deb
```

## 5. Run minikube with gpu and mount disk
```bash
minikube start --driver=docker \
  --cpus=24 \
  --memory=200g \
  --container-runtime=docker \
  --gpus=all \
  --mount --mount-string="/home2/dwnusa/{workspace}/fedpod:/fedpod" \
  --profile={minikube-context-name} # if minikube context is required
```

### 5.1. Install kubectl
ref: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/
in ~/.bashrc
```bash
alias kubectl="minikube kubectl --"
```

### 5.2. Minikube and Kubectl Context with Config and Env
```bash
<kubectl context for k3d or minikube>
kubectl config view --minify --raw > ~/.kube/config-<context> # for minikube
k3d kubeconfig get mycluster > ~/.kube/config-<context> # for k3d
export KUBECONFIG=~/.kube/config-<context>
kubectl get nodes

<minikube context>
kubectl config view --minify --raw > ~/.kube/config-<context> # for minikube
export MINIKUBE_HOME=~/.minikube-<context>
minikube status --profile=<context>
```

## 6. Install Argo Workflow
```bash
ARGO_WORKFLOWS_VERSION="v3.5.8"
kubectl create namespace argo
kubectl apply -n argo -f "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_WORKFLOWS_VERSION}/quick-start-minimal.yaml"
or
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
```
```bash
# Download the binary
curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.4.8/argo-linux-amd64.gz

# Unzip
gunzip argo-linux-amd64.gz

# Make binary executable
chmod +x argo-linux-amd64

# Move binary to path
sudo mv ./argo-linux-amd64 /usr/local/bin/argo

# Test installation
argo version
```

## 7. Open Argo-server
Open Port on minikube cluster side.
```bash
minikube kubectl edit svc argo-server
type: ClusterIP --> NodePort
```
Port-forwarding for host access.
```bash
minikube service argo-server -n argo
minikube service list
#  argo        │ argo-server │ web/2746     │ http://192.168.*.*:32634
```
```bash
curl https://192.168.*.*:32634 -k
```
or
browser access: https://192.168.*.*:32634

## 8. Configure reverse proxy using Nginx
### 8.1. routing to https
```bash
sudo apt update
sudo apt install -y nginx
sudo vim /etc/nginx/sites-available/argo.conf
```
in /etc/nginx/sites-available/argo.conf
```bash
server {
    listen 8080;
    server_name _;

    location / {
        proxy_pass https://192.168.49.2:32634;
        proxy_ssl_verify off;           # self-signed TLS 무시
        proxy_ssl_server_name on;
        proxy_ssl_protocols TLSv1.2 TLSv1.3;

        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```
```bash
sudo ln -s /etc/nginx/sites-available/argo.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```
### 8.2. check and open firewall
```bash
sudo ufw status verbose
sudo ufw allow 8080/tcp
sudo ufw reload
sudo ufw status
```
