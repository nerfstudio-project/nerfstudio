# Installation

## Prerequisites
::::::{tab-set}
:::::{tab-item} Linux

Install CUDA. This library has been tested with version 11.3. You can find CUDA download links [here](https://developer.nvidia.com/cuda-toolkit-archive) and more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

:::::
:::::{tab-item} Windows
Install [Git](https://git-scm.com/downloads).

Install CUDA. This library and the Windows installation process has been tested with version 11.3.

::::{tab-set}
:::{tab-item} 11.3 with Visual Studio 2019 v16.9

Install Visual Studio 2019 ver 16.9. This must be done before installing CUDA. The necessary components are included in the `Desktop Development with C++` workflow (also called `C++ Build Tools` in the BuildTools edition). You can find older versions of Visual Studio 2019 [here](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers). The latest version, 16.11, may cause errors when installing tiny-cuda-nn.

Install CUDA 11.3. You can find CUDA download links [here](https://developer.nvidia.com/cuda-toolkit-archive) and more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

:::
:::{tab-item} 11.7 with Visual Studio 2022

Install Visual Studio 2022. This must be done before installing CUDA. The necessary components are included in the `Desktop Development with C++` workflow (also called `C++ Build Tools` in the BuildTools edition).

Install CUDA 11.7. You can find CUDA download links [here](https://developer.nvidia.com/cuda-toolkit-archive) and more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

:::
::::
:::::
::::::

## Create environment

Nerfstudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

```

## Dependencies

(pytorch)=
### pytorch

::::{tab-set}
:::{tab-item} Torch 1.12.1 with CUDA 11.3

- To install 1.12.1 with CUDA 11.3:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

:::
:::{tab-item} Torch 1.13.1 with CUDA 11.7

- To install 1.13.1 with CUDA 11.7:

Note that if a pytorch version prior to 1.13 is installed, 
it should be uninstalled first to avoid upgrade issues (e.g. with functorch)

```bash
pip uninstall torch torchvision functorch
```

Install pytorch 1.13.1 with CUDA and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```bash
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
```

:::
::::

### tinycudann

After pytorch and ninja, install the torch bindings for tinycudann:

```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
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
Instead of installing and compiling prerequisites, setting up the environment and installing dependencies, a ready to use docker image is provided.
### Prerequisites
Docker ([get docker](https://docs.docker.com/get-docker/)) and nvidia GPU drivers ([get nvidia drivers](https://www.nvidia.de/Download/index.aspx?lang=de)), capable of working with CUDA 11.8, must be installed.
The docker image can then either be pulled from [here](https://hub.docker.com/r/dromni/nerfstudio/tags) (replace <version> with the actual version, e.g. 0.1.18)
```bash
docker pull dromni/nerfstudio:<version>
```
or be built from the repository using
```bash
docker build --tag nerfstudio -f Dockerfile .
```

To restrict to only CUDA architectures that you have available locally, use the `CUDA_ARCHITECTURES`
build arg and look up [the compute capability for your GPU](https://developer.nvidia.com/cuda-gpus).
For example, here's how to build with support for GeForce 30xx series GPUs:
```bash
docker build --build-arg CUDA_ARCHITECTURES=86 --tag nerfstudio-86 -f Dockerfile .
```

### Using an interactive container
The docker container can be launched with an interactive terminal where nerfstudio commands can be entered as usual. Some parameters are required and some are strongly recommended for usage as following:
```bash
docker run --gpus all \                                         # Give the container access to nvidia GPU (required).
            -v /folder/of/your/data:/workspace/ \               # Mount a folder from the local machine into the container to be able to process them (required).
            -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ \   # Mount cache folder to avoid re-downloading of models everytime (recommended).
            -p 7007:7007 \                                      # Map port from local machine to docker container (required to access the web interface/UI).
            --rm \                                              # Remove container after it is closed (recommended).
            -it \                                               # Start container in interactive mode.
            --shm-size=12gb \                                   # Increase memory assigned to container to avoid memory limitations, default is 64 MB (recommended).
            dromni/nerfstudio:<tag>                             # Docker image name if you pulled from docker hub.
            <--- OR --->
            nerfstudio                                          # Docker image tag if you built the image from the Dockerfile by yourself using the command from above. 
```
### Call nerfstudio commands directly
Besides, the container can also directly be used by adding the nerfstudio command to the end.
```bash
docker run --gpus all -v /folder/of/your/data:/workspace/ -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ -p 7007:7007 --rm -it --shm-size=12gb  # Parameters.
            dromni/nerfstudio:<tag> \                           # Docker image name
            ns-process-data video --data /workspace/video.mp4   # Smaple command of nerfstudio.
```
### Note
- The container works on Linux and Windows, depending on your OS some additional setup steps might be required to provide access to your GPU inside containers.
- Paths on Windows use backslash '\\' while unix based systems use a frontslash '/' for paths, where backslashes might require an escape character depending on where they are used (e.g. C:\\\\folder1\\\\folder2...). Alternatively, mounts can be quoted (e.g. ```-v 'C:\local_folder:/docker_folder'```). Ensure to use the correct paths when mounting folders or providing paths as parameters.
- Always use full paths, relative paths are known to create issues when being used in mounts into docker.
- Everything inside the container, what is not in a mounted folder (workspace in the above example), will be permanently removed after destroying the container. Always do all your tasks and output folder in workdir!
- The user inside the container is called 'user' and is mapped to the local user with ID 1000 (usually the first non-root user on Linux systems).
- The container currently is based on nvidia/cuda:11.8.0-devel-ubuntu22.04, consequently it comes with CUDA 11.8 which must be supported by the nvidia driver. No local CUDA installation is required or will be affected by using the docker image.
- The docker image (respectively Ubuntu 22.04) comes with Python3.10, no older version of Python is installed.
- If you call the container with commands directly, you still might want to add the interactive terminal ('-it') flag to get live log outputs of the nerfstudio scripts. In case the container is used in an automated environment the flag should be discarded.
- The current version of docker is built for multi-architecture (CUDA architectures) use. The target architecture(s) must be defined at build time for Colmap and tinyCUDNN to be able to compile properly. If your GPU architecture is not covered by the following table you need to replace the number in the line ```ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37``` to your specific architecture. It also is a good idea to remove all architectures but yours (e.g. ```ARG CUDA_ARCHITECTURES=86```) to speedup the docker build process a lot.
- To avoid memory issues or limitations during processing, it is recommended to use either ```--shm-size=12gb``` or ```--ipc=host``` to increase the memory available to the docker container. 12gb as in the example is only a suggestion and may be replaced by other values depending on your hardware and requirements. 

**Currently supported CUDA architectures in the docker image**

GPU | CUDA arch
-- | --
H100 | 90
40X0 | 89
30X0 | 86
A100 | 80
20X0 | 75 
TITAN V / V100 | 70
10X0 / TITAN Xp | 61
9X0 | 52
K80 | 37

## Installation FAQ

- [TinyCUDA installation errors out with cuda mismatch](tiny-cuda-mismatch-error)
- [TinyCUDA installation errors out with no CUDA toolset found](tiny-cuda-integration-error)
- [TinyCUDA installation errors out with syntax errors](tiny-cuda-syntax-error)
- [Installation errors, File "setup.py" not found](pip-install-error)
- [Runtime errors, "len(sources) > 0".](cuda-sources-error)

 <br />

(tiny-cuda-mismatch-error)=

**TinyCUDA installation errors out with cuda mismatch**

While installing tiny-cuda, you run into: `The detected CUDA version mismatches the version that was used to compile PyTorch (10.2). Please make sure to use the same CUDA versions.`

**Solution**:

Reinstall pytorch with the correct CUDA version.
See [pytorch](pytorch) under Dependencies, above.

 <br />

(tiny-cuda-integration-error)=

**(Windows) TinyCUDA installation errors out with no CUDA toolset found**

While installing tiny-cuda on Windows, you run into: `No CUDA toolset found.`

**Solution**:

Confirm that you have Visual Studio installed (CUDA 11.3 is only compatible up to 2019 ver 16.9, not 16.11 or 2022).

Make sure CUDA Visual Studio integration is enabled. This should be done automatically by the CUDA installer if it is run after Visual Studio is installed. You can also manually enable integration.

::::{tab-set}
:::{tab-item} Visual Studio 2019

To manually enable integration for Visual Studio 2019, copy all 4 files from

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\extras\visual_studio_integration\MSBuildExtensions
```

to

```
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

:::
:::{tab-item} Visual Studio 2022

To manually enable integration for Visual Studio 2022, copy all 4 files from

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\extras\visual_studio_integration\MSBuildExtensions
```

to

```
C:\Program Files\Microsoft Visual Studio\2022\[Community, Professional, Enterprise, or BuildTools]\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

:::
::::

 <br />

(tiny-cuda-syntax-error)=

**(Windows) TinyCUDA installation errors out with syntax errors**

While installing tiny-cuda on Windows, you run into: `Expected a "("`

**Solution**:

If you are using CUDA 11.3 with Visual Studio 2019, confirm that Visual Studio is on version 16.9 and not 16.11.

If you are using a different version of CUDA and Visual Studio, try switching to CUDA 11.3 with Visual Studio 2019 ver 16.9.

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
