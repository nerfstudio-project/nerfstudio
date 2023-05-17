# K-Planes

<h4>Explicit Radiance Fields in Space, Time, and Appearance</h4>


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
pip install kplanes-nerfstudio
```

## Running K-Planes

There are two default configurations provided which use the blender and DNeRF dataloaders. However, you can easily extend them to create a new configuration for different datasets.

The default configurations provided are
| Method            | Description              | Scene type                     | Memory |
| ----------------- | -------------------------| ------------------------------ | ------ |
| `kplanes`         | Tuned for blender scenes | static, synthetic              | ~4GB   |
| `kplanes-dynamic` | Tuned for DNeRF dataset  | dynamic (monocular), synthetic | ~5GB   |


for training with these two configurations you should run
```bash
ns-train kplanes --data <data-folder>
```
or
```bash
ns-train kplanes-dynamic --data <data-folder>
```

:::{admonition} Note
:class: warning

`kplanes` is set up to use blender data, (download it running `ns-download-data blender`), 
`kplanes-dynamic` is set up to use DNeRF data, (download it running `ns-download-data dnerf`).
:::


## Method

![method overview](https://sarafridov.github.io/K-Planes/assets/intro_figure.jpg)<br>
K-planes represents a scene in k dimensions -- where k can be 3 for static 3-dimensional scenes or 4 for scenes which change in time -- 
using k-choose-2 planes (or grids). After ray-sampling, the querying the field at a certain position consists in querying each plane (with interpolation), and combining the resulting features through multiplication.
This factorization of space and time keeps memory usage low, and is very flexible in the kinds of priors and regularizers that can be added.<br>
<br>

We support hybrid models with a small MLP (left) and fully explicit models (right), through the `linear_decoder` [configuration key](https://github.com/Giodiro/kplanes_nerfstudio/blob/db4130605159dadaf180228be5d0335d2c4d21f9/kplanes/kplanes.py#L87)
<br>
<video id="4d_dynamic_mlp" muted autoplay playsinline loop controls width="48%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_salmon_path_mlp.mp4" type="video/mp4">
</video>
<video id="4d_dynamic_linear" muted autoplay playsinline loop controls width="48%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_salmon_path_linear.mp4" type="video/mp4">
</video>

The model also supports decomposing a scene into its space and time components. For more information on how to do this see the [official code repo](https://github.com/sarafridov/K-Plane)
<br>
<video id="4d_spacetime"  muted autoplay playsinline loop controls width="96%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_cutbeef_spacetime_mlp.mp4" type="video/mp4">
</video>
