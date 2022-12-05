# Mip-NeRF

<h4>A Multiscale Representation for Anti-Aliasing Neural Radiance Fields</h4>

```{button-link} https://jonbarron.info/mipnerf/
:color: primary
:outline:
Paper Website
```

### Running Model

The model requires requires a fair amount of GPU memory. You can decrease the batch size using the `--pipeline.datamanager.train-num-rays-per-batch` argument to solve CUDA out of memory error (Tested with 4096 rays on 24GB GPU).

```bash
ns-train mipnerf \
  --data data/nerfstudio/poster \
  --pipeline.datamanager.train-num-rays-per-batch 4096 \
  nerfstudio-data
```

## Overview

```{image} imgs/mipnerf/models_mipnerf_pipeline-light.png
:align: center
:class: only-light
```

```{image} imgs/mipnerf/models_mipnerf_pipeline-dark.png
:align: center
:class: only-dark
```

The primary modification in MipNeRF is in the encoding for the field representation. With the modification the same _mip-NeRF_ field can be use for the coarse and fine steps of the rendering hierarchy.

```{image} imgs/mipnerf/models_mipnerf_field-light.png
:align: center
:class: only-light
:width: 400
```

```{image} imgs/mipnerf/models_mipnerf_field-dark.png
:align: center
:class: only-dark
:width: 400
```

In the field, the Positional Encoding (PE) is replaced with an Integrated Positional Encoding (IPE) that takes into account the size of the sample.
