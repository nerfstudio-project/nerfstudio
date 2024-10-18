# Zip-NeRF

<h4>A pytorch implementation of "Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields"</h4>

```{button-link} https://jonbarron.info/zipnerf/
:color: primary
:outline:
Paper Website
```
```{button-link} https://github.com/SuLvXiangXin/zipnerf-pytorch
:color: primary
:outline:
Code
```
### Installation
First, install nerfstudio and its dependencies. Then run:
```
pip install git+https://github.com/SuLvXiangXin/zipnerf-pytorch#subdirectory=extensions/cuda
pip install git+https://github.com/SuLvXiangXin/zipnerf-pytorch
```
Finally, install torch_scatter corresponding to your cuda version(https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html).


### Running Model

```bash
ns-train zipnerf --data {DATA_DIR/SCENE}
```

## Overview
Zipnerf combines mip-NeRF 360’s overall framework with iNGP’s featurization approach.
Following mip-NeRF, zipnerf assume each pixel corresponds to a cone. Given an interval along the ray, it construct a set of multisamples that approximate the shape of that conical frustum.
Also,  it present an alternative loss that, unlike mip-NeRF 360’s interlevel loss, is continuous and smooth with respect to distance along the ray to prevent z-aliasing.