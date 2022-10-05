# Mip-NeRF

<h4>A Multiscale Representation for Anti-Aliasing Neural Radiance Fields</h4>

```{button-link} https://jonbarron.info/mipnerf/
:color: primary
:outline:
Paper Website
```

### Running Model

```bash
ns-train mipnerf
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
