# Install Minikube

## Prerequisites
- Linux
- Latest NVIDIA GPU drivers
- minikube v1.32.0-beta.0 or later (docker driver only)

## 1. Install Docker using the apt repository
### 1.1. Set up Docker's apt repository.
```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
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
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \ 
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
### 3.2. Update the packages list from the repository:
```bash
$ sudo apt-get update
```
### 3.3. Install the NVIDIA Container Toolkit packages:
```bash
$ export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
$ sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```
### 3.4. Configure the container runtime by using the nvidia-ctk command:
```bash
$ sudo nvidia-ctk runtime configure --runtime=docker
```
The **nvidia-ctk** command modifies the **/etc/docker/daemon.json** file on the host. The file is updated so that Docker can use the NVIDIA Container Runtime.
### 3.5. Restart the Docker daemon:
```bash
$ sudo systemctl restart docker
```
### 3.6. Run a sample CUDA container:
```bash
$ sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## 4. Install Minikube (v1.32.0-beta.0 or later)
To install the latest minikube **stable** release on **x86-64 Linux** using **Debian package**:
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo dpkg -i minikube_latest_amd64.deb
```
in ~/.bashrc
```bash
alias kubectl="minikube kubectl --"
```
```bash
kubectl start --driver docker --container-runtime docker --gpus all
```
