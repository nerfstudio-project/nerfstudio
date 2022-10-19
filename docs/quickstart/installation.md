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
