# Docker usage

Docker allows to create independent and isolated environments to launch and deploy applications.

## Installation on Linux

Before install Docker Engine you need to set up the Docker repository.
To setup the repository run the following commands

```bash
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
$ sudo apt update

```
To install docker engine run the following commands
```bash
 $ sudo apt-get update
 $ sudo apt-get install docker-ce docker-ce-cli containerd.io
```
Check the status with the command
```bash
$ sudo systemctl status docker
```

Check if you can access images and download them from Docker Hub, type the following

```bash
$ sudo docker run hello-world
```
## Installation on Windows
Download Docker Desktop for Windows [here](https://hub.docker.com/editions/community/docker-ce-desktop-windows/) then double-click Docker Desktop Installer.exe to run the installer, make sure to follow the instructions on the installation wizard to authorize the installer and proceed with the install
#### System Requirements for Windows

* Windows 10 64 bits Pro, Enterprice or Education.
* Hyper-V and Containers Windows features must be enabled.
* 64 bit processor
* 4GB system RAM
* BIOS hardware virtualization support

## CLONE THE GIT REPO ON BRACH NPDB-DEMO-ON AND BUILD THE DOCKERFILE
```bash
$ git clone --single-branch --branch=npdb-demo-on https://github.com/Asymm-Developers/nowinsurance-loss-runs.git

$ cd nowinsurance-loss-runs

$ docker build -t=lossruns .
```
## COPY THE MODEL TO THE CONTAINER (OPTIONAL, YOU CAN POINT TO THE MODEL PATH OUTSIDE THE CONTEINER)
```bash
$ docker cp <NPDB_ner_model> <container ID>:/home/app/models/
```
## RUN THE DOCKER IMAGE
```bash
$ docker lossruns [params]
$ docker lossruns --h --f=/home/app/NPDB/data/NPDBQA1.pdf --m=/home/app/models/NPDB_ner_model --d=false # i.e.
```

## IGNORE UNTIL MAINTAINMENT (programed to 2021 Jan 8-11)
## Create the Docker image
To create the image run the following command in root directory of the project
```bash
$ docker build -t lossrun .
```
-t param defines the name of the image, in this case 'lossrun' but you can put what you want.

## Run the Docker image
Here we put the name of the image after docker run and then execute the python script
```bash
$ docker run lossrun python3 main_npdb.py [params]
```
