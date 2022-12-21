# Installation

## Prerequisites

CUDA must be installed on the system. This library has been tested with version 11.3. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

## Create environment

Nerfstudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before preceding.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

```

## Dependencies

Install pytorch with CUDA (this repo has been tested with CUDA 11.3) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

```

## Installing nerfstudio

**From pip**

```bash
pip install nerfstudio
```

**From source**
Optional, use this command if you want the latest development version.

```bash
git clone git@github.com:nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

:::{admonition} Note
:class: info
Below are optional installations, but makes developing with nerfstudio much more convenient.
:::

**Tab completion (bash & zsh)**

This needs to be rerun when the CLI changes, for example if nerfstudio is updated.

```bash
ns-install-cli
```

**Development packages**

```bash
pip install -e .[dev]
pip install -e .[docs]
```

## Use docker image
Instead of installing and compiling prerequisites, setting up the environment and installing dependencies, a ready to use docker image is provided. \
### Prerequisites
Docker ([get docker](https://docs.docker.com/get-docker/)) and nvidia GPU drivers ([get nvidia drivers](https://www.nvidia.de/Download/index.aspx?lang=de)), capable of working with CUDA 11.7, must be installed.
The docker image can then either be pulled from [here](https://hub.docker.com/r/dromni/nerfstudio/tags) (replace <version> with the actual version, e.g. 0.1.10)
```bash
docker pull dromni/nerfstudio:<version>
```
or be built from the repository using
```bash
docker build --tag nerfstudio -f Dockerfile .
```
### Using an interactive container
The docker container can be launched with an interactive terminal where nerfstudio commands can be entered as usual. Some parameters are required and some are strongly reocmmended for usage as following:
```bash
docker run --gpus all \                                         # Give the container access to nvidia GPU (required).
            -v /folder/of/your/data:/workspace/ \               # Mount a folder from the local machine into the container to be able to process them (required).
            -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ \   # Mount cache folder to avoid re-downloading of models everytime (recommended).
            -p 7007:7007 \                                      # Map port from local machine to docker container (required to access the web interface/UI).
            --rm \                                              # Remove container after it is closed (recommended).
            -it \                                               # Start container in interactive mode.
            nerfstudio                                          # Docker image name
```
### Call nerfstudio commands directly
Besides, the container can also directly be used by adding the nerfstudio command to the end.
```bash
docker run --gpus all -v /folder/of/your/data:/workspace/ -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ -p 7007:7007 --rm -it # Parameters.
            nerfstudio \                                        # Docker image name
            ns-process-data video --data /workspace/video.mp4   # Smaple command of nerfstudio.
```
### Note
- The container works on Linux and Windows, depending on your OS some additional setup steps might be required to provide access to your GPU inside containers.
- Paths on Windows use backslash '\\' while unix based systems use a frontslash '/' for paths, where backslashes might require an escape character depending on where they are used (e.g. C:\\\\folder1\\\\folder2...). Ensure to use the correct paths when mounting folders or providing paths as parameters.
- Everything inside the container, what is not in a mounted folder (workspace in the above example), will be permanently removed after destroying the container. Always do all your tasks and output folder in workdir!
- The user inside the container is called 'user' and is mapped to the local user with ID 1000 (usually the first non-root user on Linux systems).
- The container currently is based on nvidia/cuda:11.7.1-devel-ubuntu22.04, consequently it comes with CUDA 11.7 which must be supported by the nvidia driver. No local CUDA installation is required or will be affected by using the docker image.
- The docker image (respectively Ubuntu 22.04) comes with Python3.10, no older version of Python is installed.
- If you call the container with commands directly, you still might want to add the interactive terminal ('-it') flag to get live log outputs of the nerfstudio scripts. In case the container is used in an automated environment the flag should be discarded.


## Installation FAQ

- [TinyCUDA installation errors out with cuda mismatch](tiny-cuda-error)
- [Installation errors, File "setup.py" not found](pip-install-error)
- [Runtime errors, "len(sources) > 0".](cuda-sources-error)

 <br />

(tiny-cuda-error)=

**TinyCUDA installation errors out with cuda mismatch**

While installing tiny-cuda, you run into: `The detected CUDA version mismatches the version that was used to compile PyTorch (10.2). Please make sure to use the same CUDA versions.`

**Solution**:

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

 <br />

(pip-install-error)=

**Installation errors, File "setup.py" not found**

When installing dependencies and nerfstudio with `pip install -e .`, you run into: `ERROR: File "setup.py" not found. Directory cannot be installed in editable mode`

**Solution**:
This can be fixed by upgrading pip to the latest version:

```
python -m pip install --upgrade pip
```

 <br />

(cuda-sources-error)=

**Runtime errors: "len(sources) > 0", "ctype = \_C.ContractionType(type.value) ; TypeError: 'NoneType' object is not callable".**

When running `train.py `, an error occurs when installing cuda files in the backend code.

**Solution**:
This is a problem with not being able to detect the correct CUDA version, and can be fixed by updating the CUDA path environment variables:

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```
