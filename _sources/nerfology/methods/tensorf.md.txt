# TensoRF

<h4>Tensorial Radiance Fields</h4>

```{button-link} https://apchenstu.github.io/TensoRF/
:color: primary
:outline:
Paper Website
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://apchenstu.github.io/TensoRF/video/train_process.mp4" type="video/mp4">
</video>

### Running Model

```bash
ns-train tensorf
```

## Overview

```{image} imgs/tensorf/models_tensorf_pipeline.png
:align: center
```

TensoRF models the radiance field of a scene as a 4D tensor, which represents a 3D voxel grid with per-voxel multi-channel features. TensoRF factorizes the 4D scene tensor into multiple compact low-rank tensor components using CP or VM modes. CP decomposition factorizes tensors into rank-one components with compact vectors. Vector-Matrix (VM) decomposition factorizes tensors into compact vector and matrix factors.

```{image} imgs/tensorf/models_tensorf_factorization.png
:align: center
```

TensoRF with CP(left) and VM(right) decompositions results in a significantly reduced memory footprint compared to previous and concurrent works that directly optimize per-voxel features, such as [Plenoxels](https://alexyu.net/plenoxels/) and [PlenOctrees](https://alexyu.net/plenoctrees/). In experiments, TensoRF with CP decomposition achieves fast reconstruction with improved rendering quality and a smaller model size compared to NeRF. Furthermore, TensoRF with VM decomposition enhances rendering quality even further, while reducing reconstruction time and maintaining a compact model size.
