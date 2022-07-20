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

### Overview

### Field Representation

#### Positional Encoding

### Rendering

#### Hierarchical Sampling

## Benchmarks

##### Blender Synthetic
| Implementation                                                                    |    Mic    | Ficus     |   Chair   | Hotdog    | Materials | Drums     | Ship      | Lego      | Average   |
|-----------------------------------------------------------------------------------|:---------:|-----------|:---------:|-----------|-----------|-----------|-----------|-----------|-----------|
| pyRad                                                                             |     34.28 | **30.63** | **36.16** |     36.49 |     28.36 | **25.44** |     28.69 | **33.67** | **31.71** |
| [TF NeRF](https://github.com/bmild/nerf)                                          |     32.91 |     30.13 |     33.00 |     36.18 |     29.62 |     25.01 |     28.65 |     32.54 |     31.04 |
| [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) | **34.53** |     30.43 |     34.08 | **36.92** | **29.91** |     25.03 | **29.36** |     33.28 |     31.69 |