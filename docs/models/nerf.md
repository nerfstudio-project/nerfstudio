# NeRF

<h4>Neural Radiance Fields</h4>

```{button-link} https://www.matthewtancik.com/nerf
:color: primary
:outline:
Paper Website
```

### Running the Model

```bash
python scripts/run_train.py --config-name=graph_vanilla_nerf.yaml
```

## Method

### Pipeline

```{image} imgs/models_nerf-field-light.png
:align: center
:class: only-light
:width: 500
```

```{image} imgs/models_nerf-field-dark.png
:align: center
:class: only-dark
:width: 500
```

### Overview

If you have arrived to this site, it is likely that you have atleast heard of NeRFs. This page will discuss the original NeRF paper, _"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"_ by Mildenhall, Srinivasan, Tancik et al. (2020). For most tasks, using the original NeRF model is likely not a good choice and hence we provide implementations of various other NeRF related models. It is however useful to understand how NeRF's work as most follow ups follow a similar structure.

The goal is to optimize a volumetric representation of a scene that can be rendered from novel viewpoints. This representation is optimized from a set of images and associated camera poses.

```{admonition} Assumptions
If any of the following assumptions are broken, the reconstructions may fail completely or contain artifacts such as excess geometry.
* Camera poses are known
* Scene is static, objects do not move
* The scene appearance is constant (ie. exposure doesn't change)
* Dense input capture (Each point in the scene should be visible in multiple images)
```

### Field Representation

NeRFs are a volumetric representation encoded into a neural network. They are not 3D meshes and they are not voxels. For each point in space the NeRF represents a view dependent radiance. More concretely each point has a density which describes how transparent or opaque a point in space is. They also have a view dependent color that changes depending on the angle the point is viewed.

#### Positional Encoding

An extra trick is necessary to making the neural network expressive enough to represent fine details in the scene. The input coordinates $(x,y,z,\theta,\phi)$ need to be encoded to a higher dimensional space prior to being input into the network. You can learn more about encodings [here](../model_components/visualize_encoders.ipynb).

### Rendering

Now that we have a representation of space, we need some way to render new images of it. To accomplish this, we are going to _shoot_ a ray from the target pixel and evaluate points along that ray. We then rely on classic volumetric rendering techniques [[Kajiya, 1984]](https://dl.acm.org/doi/abs/10.1145/964965.808594) to composite the points into a predicted color. This compositing is similar to what happens in tools like Photoshop when you layer multiple objects of varying opacity on top of each other. The only difference is that NeRF takes into account the differences in spacing between points.

#### Sampling

How we sample the rays in space is an important design decision.

## Benchmarks

##### Blender Synthetic

| Implementation                                                                    |    Mic    | Ficus     |   Chair   | Hotdog    | Materials | Drums     | Ship      | Lego      | Average   |
| --------------------------------------------------------------------------------- | :-------: | --------- | :-------: | --------- | --------- | --------- | --------- | --------- | --------- |
| nerfactory                                                                        |   33.76   | **31.98** | **34.35** | 36.57     | **31.00** | **25.11** | 29.87     | **34.46** | **32.14** |
| [TF NeRF](https://github.com/bmild/nerf)                                          |   32.91   | 30.13     |   33.00   | 36.18     | 29.62     | 25.01     | 28.65     | 32.54     | 31.04     |
| [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) | **34.53** | 30.43     |   34.08   | **36.92** | 29.91     | 25.03     | **29.36** | 33.28     | 31.69     |
