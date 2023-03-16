# How to start the download script in the docker container
## Make the image from the Dockerfile
In the command line type:
```bash
$ docker build <DOCKERFILE_PATH> --tag <IMAGE_NAME>
```
* `<DOCKERFILE_PATH>` - path to the `Dockerfile` (`.` - current directory)
* `<IMAGE_NAME>` - name of the created image 
## Run a container from the created image
In the command line type:
```bash
$ docker run -it --rm  -v <MOUNTED_DIR>:<CONTAINER_WORKDIR> <IMAGE_NAME> /bin/bash
```
* `<MOUNTED_DIR>` - path to the mounted directory where the files `auto_download.py` and `context.json` are located
* `<CONTAINER_WORKDIR>` - path to the work directory in the container
## Install esgf-pyclient environment
Now all commands are executer in the running container.

In the command line type:
```bash
$ conda create -n esgf-pyclient -c conda-forge esgf-pyclient
```
Run the environment
```bash
$ source activate esgf-pyclient
```
## Run the download script in the container
```bash
(esgf-pyclinet)$ python3 auto_download.py <PATH_TO_CONTEXT> <PATH_TO_DOWNLOAD_DIR>
```
* `<PATH_TO_CONTEXT>` - path to the file `context.json`
* `<PATH_TO_DOWNLOAD_DIR>` - path to the directory where the data will be downloaded

## After downloading
Deactivate the environment
```bash
(esgf-pyclinet)$ source deactivate
```
Exit the container
```bash
$ exit
```