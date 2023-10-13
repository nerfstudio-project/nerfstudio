# NeRFstudio fork for Stanford NAVLab

This is a fork of the [NeRFstudio](https://github.com/nerfstudio-project/nerfstudio/) project. It is used in the construction of Neural City Maps, a project by the Stanford NAVLab.

## Installation

### Linux
```
git clone https://github.com/Stanford-NavLab/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .
```

### Sherlock (Stanford HPC, preinstalled)
TODO: Improve instructions
1. Activate the Singularity container
2. Activate conda environment
3. cd to Sep_23 folder nerfstudio directory
   
## Training (customized for Sherlock)

From the forked directory
```
python scripts/train_nerfstudio.py <data_directory> <optional flags>
```

For flags, run `python scripts/train_nerfstudio.py --help`
