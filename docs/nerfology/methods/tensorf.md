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

TensoRF models the radiance field of a scene as a 4D tensor, which represents a 3D voxel grid with per-voxel multi-channel features. TensoRF factorizes the 4D scene tensor into multiple compact low-rank tensor components (CP and VM mode). CP decomposition factorizes tensors into rank-one components with compact vectors. vector-matrix (VM) decomposition factorizes tensors into compact vector and matrix factors.

```{image} imgs/tensorf/models_tensorf_factorization.png
:align: center
```

TensoRF with CP and VM decompositions lead to a significantly lower memory footprint in comparison to previous and concurrent works that directly optimize per-voxel features (such as Plenoxels and PlenOctrees). Experimentally, TensoRF with CP decomposition achieves fast reconstruction with better rendering quality and even a smaller model size compared to NeRF. Moreover, TensoRF with VM decomposition further boosts rendering quality, while reducing the reconstruction time and retaining a compact model size.
