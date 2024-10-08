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
> docker run -v ./cc359ppmi128:/fedpod/cc359ppmi128 -v ./states/R00:/fedpod/states -it fedpod:v0.3
```
