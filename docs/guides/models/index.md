# Models

We provide a set of pre implemented nerfactory models. One of the goals of nerfactory is to modularize the various NeRF techniques as much as possible. As a result, many of the techniques from these pre-implemented models can be mixed.

## Running a model

It's easy!

```bash
python scripts/train.py --config-name MODEL_CONFIG
```

## Guides

In addition to their implementations, we have provided guides that walk through each of these method.

```{toctree}
    :maxdepth: 1
    NeRF<nerf.md>
    Mip-NeRF<mipnerf.md>
    Mip-NeRF 360<mipnerf_360.md>
    Instant-NGP<instant_ngp.md>
    NeRF-W<nerf_w.md>
    Semantic NeRF<semantic_nerf.md>
```
