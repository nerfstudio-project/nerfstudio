# Tetra-NeRF

<h4>Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra</h4>

```{button-link} https://jkulhanek.com/tetra-nerf
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/jkulhanek/tetra-nerf
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://jkulhanek.com/tetra-nerf/resources/intro-video.mp4" type="video/mp4">
</video>

**SfM input pointcloud is triangulated and resulting tetrahedra is used as the radiance field representation**

## Installation

First, make sure to install the following:
```
CUDA (>=11.3)
PyTorch (>=1.12.1)
Nerfstudio (>=0.2.0)
OptiX (>=7.2, preferably 7.6)
CGAL
CMake (>=3.22.1)
```
Follow the [installation section](https://github.com/jkulhanek/tetra-nerf/blob/master/README.md#installation) in the tetra-nerf repository

Finally, you can install **Tetra-NeRF** by running:
```bash
pip install git+https://github.com/jkulhanek/tetra-nerf
```

## Running Tetra-NeRF on custom data
Details for running Tetra-NeRF can be found [here](https://github.com/jkulhanek/tetra-nerf).

```bash
python -m tetranerf.scripts.process_images --path <data folder>
python -m tetranerf.scripts.triangulate --pointcloud <data folder>/sparse.ply --output <data folder>/sparse.th
ns-train tetra-nerf --pipeline.model.tetrahedra-path <data folder>/sparse.th minimal-parser --data <data folder>
```

Three following variants of Tetra-NeRF are provided:

| Method                | Description                            | Memory  | Quality |
| --------------------- | -------------------------------------- | ------- | ------- |
| `tetra-nerf-original` | Official implementation from the paper | ~18GB*  | Good    |
| `tetra-nerf`          | Different sampler - faster and better  | ~16GB*  | Best    |

*Depends on the size of the input pointcloud, estimate is given for a larger scene (1M points)

## Method
![method overview](https://jkulhanek.com/tetra-nerf/resources/overview-white.svg)<br>
The input to Tetra-NeRF is a point cloud which is triangulated to get a set of tetrahedra used to represent the radiance field. Rays are sampled, and the field is queried. The barycentric interpolation is used to interpolate tetrahedra vertices, and the resulting features are passed to a shallow MLP to get the density and colours for volumetric rendering.<br>

[![demo blender lego (sparse)](https://jkulhanek.com/tetra-nerf/resources/images/blender-lego-sparse-100k-animated-cover.gif)](https://jkulhanek.com/tetra-nerf/demo.html?scene=blender-lego-sparse)
[![demo mipnerf360 garden (sparse)](https://jkulhanek.com/tetra-nerf/resources/images/360-garden-sparse-100k-animated-cover.gif)](https://jkulhanek.com/tetra-nerf/demo.html?scene=360-garden-sparse)
[![demo mipnerf360 garden (sparse)](https://jkulhanek.com/tetra-nerf/resources/images/360-bonsai-sparse-100k-animated-cover.gif)](https://jkulhanek.com/tetra-nerf/demo.html?scene=360-bonsai-sparse)
[![demo mipnerf360 kitchen (dense)](https://jkulhanek.com/tetra-nerf/resources/images/360-kitchen-dense-300k-animated-cover.gif)](https://jkulhanek.com/tetra-nerf/demo.html?scene=360-kitchen-dense)


## Existing checkpoints and predictions
For an easier comparisons with Tetra-NeRF, published checkpoints and predictions can be used:

| Dataset  | Checkpoints | Predictions | Input tetrahedra |
| -------- | ----------- | ----------- | ---------------- |
| Mip-NeRF 360 (public scenes) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/mipnerf360-public-checkpoints.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/mipnerf360-public-predictions.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/mipnerf360-public-tetrahedra.tar.gz) |
| Blender | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/blender-checkpoints.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/blender-predictions.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/blender-tetrahedra.tar.gz) |
| Tanks and Temples | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/nsvf-tanks-and-temples-checkpoints.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/nsvf-tanks-and-temples-predictions.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/nsvf-tanks-and-temples-tetrahedra.tar.gz) |

