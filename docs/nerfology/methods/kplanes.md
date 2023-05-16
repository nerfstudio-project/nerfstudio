# K-Planes

<h4>K-Planes:  Explicit Radiance Fields in Space, Time, and Appearance</h4>


```{button-link} https://sarafridov.github.io/K-Planes/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/sarafridov/K-Plane
:color: primary
:outline:
Official Code
```

```{button-link} https://github.com/Giodiro/kplanes_nerfstudio
:color: primary
:outline:
Nerfstudio add-on code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/small_teaser.mp4" type="video/mp4">
</video>

**A unified model for static, dynamic and variable appearance NeRFs.**


## Installation

First, install nerfstudio and its dependencies. Then install the K-Planes add-on
```
pip install git+https://github.com/giodiro/kplanes_nerfstudio
```

## Running K-Planes

Currently, there are two default configurations provided which use the blender and DNeRF dataloaders. However, you can easily extend them to create a new configuration for different datasets.

The default configurations provided are
| Method            | Description                                       | Scene type                     |
| ----------------- | ------------------------------------------------- | ------------------------------ |
| `kplanes`         | Config. tuned for synthetic NeRF (blender) scenes | static, synthetic              |
| `kplanes-dynamic` | Config. tuned for DNeRF dataset                   | dynamic (monocular), synthetic |


for training with these two configurations you should run
```bash
ns-train kplanes --data <data-folder>
```
or
```bash
ns-train kplanes-dynamic --data <data-folder>
```

## Method

![method overview](https://sarafridov.github.io/K-Planes/assets/intro_figure.jpg)<br>
K-planes represents a scene in k dimensions -- where k can be 3 for static 3-dimensional scenes or 4 for scenes which change in time -- 
using k-choose-2 planes (or grids). After ray-sampling, the querying the field at a certain position consists in querying each plane (with interpolation), and combining the resulting features through multiplication.
This factorization of space and time keeps memory usage low, and is very flexible in the kinds of priors and regularizers that can be added.<br>
<br>

We support hybrid models with a small MLP (left) and fully explicit models (right)
<video id="4d_dynamic_mlp" muted autoplay playsinline loop controls width="48%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_salmon_path_mlp.mp4" type="video/mp4">
</video>
<video id="4d_dynamic_linear" muted autoplay playsinline loop controls width="48%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_salmon_path_linear.mp4" type="video/mp4">
</video>

We also support decomposing a scene into its space and time components
<video id="4d_spacetime"  muted autoplay playsinline loop controls width="96%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_cutbeef_spacetime_mlp.mp4" type="video/mp4">
</video>
