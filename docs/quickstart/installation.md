# Installation

## Prerequisites

::::::{tab-set}
:::::{tab-item} Linux

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

:::::
:::::{tab-item} Windows

:::{admonition} Note
:class: info
Nerfstudio on Windows is less tested and more fragile, due to way more moving parts outside of Nerfstudio's control.  
The instructions also tend to break over time as updates to different Windows packages happen.  
Installing Nerfstudio on Linux instead is recommended if you have the option.  
Alternatively, installing Nerfstudio under WSL2 (temporary unofficial guide [here](https://gist.github.com/SharkWipf/0a3fc1be3ea88b0c9640db6ce15b44b9), not guaranteed to work) is also an option, but this comes with its own set of caveats.
:::

Install [Git](https://git-scm.com/downloads).

Install Visual Studio 2022. This must be done before installing CUDA. The necessary components are included in the `Desktop Development with C++` workflow (also called `C++ Build Tools` in the BuildTools edition).

Install Visual Studio Build Tools. If MSVC 143 does not work (usually will fail if your version > 17.10), you may also need to install MSVC 142 for Visual Studio 2019. Ensure your CUDA environment is set up properly.

Activate your Visual C++ environment:
Navigate to the directory where `vcvars64.bat` is located. This path might vary depending on your installation. A common path is:

```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build
```

Run the following command:
```bash
./vcvars64.bat
```

If the above command does not work, try activating an older version of VC:
```bash
./vcvarsall.bat x64 -vcvars_ver=<your_VC++_compiler_toolset_version>
```
Replace `<your_VC++_compiler_toolset_version>` with the version of your VC++ compiler toolset. The version number should appear in the same folder.

For example:
```bash
./vcvarsall.bat x64 -vcvars_ver=14.29
```
:::{admonition} Note
:class: info
When updating, or if you close your terminal before you finish the installation and run your first `splatfacto`, you have to re-do this environment activation step.
:::

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

:::::
::::::

## Create environment

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

```

## Dependencies

(pytorch)=

### PyTorch

Note that if a PyTorch version prior to 2.0.1 is installed,
the previous version of pytorch, functorch, and tiny-cuda-nn should be uninstalled.

```bash
pip uninstall torch torchvision functorch tinycudann
```

::::{tab-set}
:::{tab-item} Torch 2.1.2 with CUDA 11.8 (recommended)

Install PyTorch 2.1.2 with CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

To build the necessary CUDA extensions, `cuda-toolkit` is also required. We
recommend installing with conda:

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

:::
:::{tab-item} Torch 2.0.1 with CUDA 11.7

Install PyTorch 2.0.1 with CUDA 11.7:

```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

To build the necessary CUDA extensions, `cuda-toolkit` is also required. We
recommend installing with conda:

```bash
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
```

:::
::::

### Install tiny-cuda-nn

::::::{tab-set}
:::::{tab-item} Linux

After pytorch and ninja, install the torch bindings for tiny-cuda-nn:
```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

:::::
:::::{tab-item} Windows

Install the torch bindings for tiny-cuda-nn:
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

:::::
::::::

## Installing nerfstudio

**From pip**

```bash
pip install nerfstudio
```

**From source**
Optional, use this command if you want the latest development version.

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
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

## Using Pixi
[Pixi](https://pixi.sh/latest/) is a fast software package manager built on top of the existing conda ecosystem. Spins up development environments quickly on Windows, macOS and Linux. (Currently only linux is supported for nerfstudio)

### Prerequisites
Make sure to have pixi installed, detailed instructions [here](https://pixi.sh/latest/)

TLDR for linux:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Install Pixi Environmnent 
After Pixi is installed, you can run
```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pixi run post-install
pixi shell
```
This will fetch the latest Nerfstudio code, install all enviroment dependencies including colmap, tinycudann and hloc, and then activate the pixi environment (similar to conda).  
From now on, each time you want to run Nerfstudio in a new shell, you have to navigate to the nerfstudio folder and run `pixi shell` again.

You could also run

```bash
pixi run post-install
pixi run train-example-nerf
```

to download an example dataset and run nerfacto straight away.

Note that this method gets you the very latest upstream Nerfstudio version, if you want to use a specific release, you have to first checkout a specific version or commit in the nerfstudio folder, i.e.:
```
git checkout tags/v1.1.3
```

Similarly, if you want to update, you want to update the git repo in your nerfstudio folder:
```
git pull
```
Remember that if you ran a checkout on a specific tag before, you have to manually specify a new tag or `git checkout main` to see the new changes.

## Use docker image

Instead of installing and compiling prerequisites, setting up the environment and installing dependencies, a ready to use docker image is provided.

### Prerequisites

Docker ([get docker](https://docs.docker.com/get-docker/)) and nvidia GPU drivers ([get nvidia drivers](https://www.nvidia.de/Download/index.aspx?lang=de)), capable of working with CUDA 11.8, must be installed.
The docker image can then either be pulled from [here](https://github.com/nerfstudio-project/nerfstudio/pkgs/container/nerfstudio) (`latest` can be replaced with a fixed version, e.g., `1.1.3`)

```bash
docker pull ghcr.io/nerfstudio-project/nerfstudio:latest
```

or be built from the repository using

```bash
docker build --tag nerfstudio -f Dockerfile .
```

To restrict to only CUDA architectures that you have available locally, use the `CUDA_ARCHITECTURES`
build arg and look up [the compute capability for your GPU](https://developer.nvidia.com/cuda-gpus).
For example, here's how to build with support for GeForce 30xx series GPUs:

```bash
docker build \
    --build-arg CUDA_ARCHITECTURES=86 \
    --tag nerfstudio-86 \
    --file Dockerfile .
```

### Using an interactive container

The docker container can be launched with an interactive terminal where nerfstudio commands can be entered as usual. Some parameters are required and some are strongly recommended for usage as following:

```bash
docker run --gpus all \                                         # Give the container access to nvidia GPU (required).
            -u $(id -u) \                                       # To prevent abusing of root privilege, please use custom user privilege to start.
            -v /folder/of/your/data:/workspace/ \               # Mount a folder from the local machine into the container to be able to process them (required).
            -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ \   # Mount cache folder to avoid re-downloading of models everytime (recommended).
            -p 7007:7007 \                                      # Map port from local machine to docker container (required to access the web interface/UI).
            --rm \                                              # Remove container after it is closed (recommended).
            -it \                                               # Start container in interactive mode.
            --shm-size=12gb \                                   # Increase memory assigned to container to avoid memory limitations, default is 64 MB (recommended).
            ghcr.io/nerfstudio-project/nerfstudio:<tag>         # Docker image name if you pulled from GitHub.
            <--- OR --->
            nerfstudio                                          # Docker image tag if you built the image from the Dockerfile by yourself using the command from above.
```

### Call nerfstudio commands directly

Besides, the container can also directly be used by adding the nerfstudio command to the end.

```bash
docker run --gpus all -u $(id -u) -v /folder/of/your/data:/workspace/ -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ -p 7007:7007 --rm -it --shm-size=12gb  # Parameters.
            ghcr.io/nerfstudio-project/nerfstudio:<tag> \       # Docker image name if you pulled from GitHub.
            ns-process-data video --data /workspace/video.mp4   # Smaple command of nerfstudio.
```

### Note

- The container works on Linux and Windows, depending on your OS some additional setup steps might be required to provide access to your GPU inside containers.
- Paths on Windows use backslash '\\' while unix based systems use a frontslash '/' for paths, where backslashes might require an escape character depending on where they are used (e.g. C:\\\\folder1\\\\folder2...). Alternatively, mounts can be quoted (e.g. `-v 'C:\local_folder:/docker_folder'`). Ensure to use the correct paths when mounting folders or providing paths as parameters.
- Always use full paths, relative paths are known to create issues when being used in mounts into docker.
- Everything inside the container, what is not in a mounted folder (workspace in the above example), will be permanently removed after destroying the container. Always do all your tasks and output folder in workdir!
- The container currently is based on nvidia/cuda:11.8.0-devel-ubuntu22.04, consequently it comes with CUDA 11.8 which must be supported by the nvidia driver. No local CUDA installation is required or will be affected by using the docker image.
- The docker image (respectively Ubuntu 22.04) comes with Python3.10, no older version of Python is installed.
- If you call the container with commands directly, you still might want to add the interactive terminal ('-it') flag to get live log outputs of the nerfstudio scripts. In case the container is used in an automated environment the flag should be discarded.
- The current version of docker is built for multi-architecture (CUDA architectures) use. The target architecture(s) must be defined at build time for Colmap and tinyCUDNN to be able to compile properly. If your GPU architecture is not covered by the following table you need to replace the number in the line `ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37` to your specific architecture. It also is a good idea to remove all architectures but yours (e.g. `ARG CUDA_ARCHITECTURES=86`) to speedup the docker build process a lot.
- To avoid memory issues or limitations during processing, it is recommended to use either `--shm-size=12gb` or `--ipc=host` to increase the memory available to the docker container. 12gb as in the example is only a suggestion and may be replaced by other values depending on your hardware and requirements.

**Currently supported CUDA architectures in the docker image**

(tiny-cuda-arch-list)=

| GPU             | CUDA arch |
| --------------- | --------- |
| H100            | 90        |
| 40X0            | 89        |
| 30X0            | 86        |
| A100            | 80        |
| 20X0            | 75        |
| TITAN V / V100  | 70        |
| 10X0 / TITAN Xp | 61        |
| 9X0             | 52        |
| K80             | 37        |

## Installation FAQ

- [ImportError: DLL load failed while importing \_89_C](tiny-cuda-mismatch-arch)
- [tiny-cuda-nn installation errors out with cuda mismatch](tiny-cuda-mismatch-error)
- [tiny-cuda-nn installation errors out with no CUDA toolset found](tiny-cuda-integration-error)
- [Installation errors, File "setup.py" not found](pip-install-error)
- [Runtime errors, "len(sources) > 0".](cuda-sources-error)

 <br />

(tiny-cuda-mismatch-arch)=

**ImportError: DLL load failed while importing \_89_C**

This occurs with certain GPUs that have CUDA architecture versions (89 in the example above) for which tiny-cuda-nn does not automatically compile support.

**Solution**:

Reinstall tiny-cuda-nn with the following command:

::::::{tab-set}
:::::{tab-item} Linux

```bash
TCNN_CUDA_ARCHITECTURES=XX pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

:::::
:::::{tab-item} Windows

```bash
set TCNN_CUDA_ARCHITECTURES=XX
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

:::::
::::::

Where XX is the architecture version listed [here](tiny-cuda-arch-list). Ie. for a 4090 GPU use `TCNN_CUDA_ARCHITECTURES=89`

 <br />

(tiny-cuda-mismatch-error)=

**tiny-cuda-nn installation errors out with cuda mismatch**

While installing tiny-cuda, you run into: `The detected CUDA version mismatches the version that was used to compile PyTorch (10.2). Please make sure to use the same CUDA versions.`

**Solution**:

Reinstall PyTorch with the correct CUDA version.
See [pytorch](pytorch) under Dependencies, above.

 <br />

(tiny-cuda-integration-error)=

**(Windows) tiny-cuda-nn installation errors out with no CUDA toolset found**

While installing tiny-cuda on Windows, you run into: `No CUDA toolset found.`

**Solution**:

Confirm that you have Visual Studio installed.

Make sure CUDA Visual Studio integration is enabled. This should be done automatically by the CUDA installer if it is run after Visual Studio is installed. You can also manually enable integration.

::::{tab-set}
:::{tab-item} Visual Studio 2019

To manually enable integration for Visual Studio 2019, copy all 4 files from

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\visual_studio_integration\MSBuildExtensions
```

to

```
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

:::
:::{tab-item} Visual Studio 2022

To manually enable integration for Visual Studio 2022, copy all 4 files from

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\visual_studio_integration\MSBuildExtensions
```

to

```
C:\Program Files\Microsoft Visual Studio\2022\[Community, Professional, Enterprise, or BuildTools]\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

:::
::::

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

**Other errors**

(Windows) A lot of errors on Windows can be caused by not having the Visual Studio environment loaded.  
If you run into errors you can't figure out, please try re-activating the Visual Studio environment (as outlined at the top of the Windows instructions on this page) and try again.  
This activation only lasts within your current terminal session and does not extend to other terminals, but this should only be needed on first run and on updates.
