# Feature Splatting

<h4>Feature Splatting</h4>

```{button-link} https://feature-splatting.github.io/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/vuer-ai/feature-splatting/
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://feature-splatting.github.io/resources/basic_ns_demo_feature_only.mp4" type="video/mp4">
</video>

**Feature Splatting distills SAM-enhanced CLIP features into 3DGS for segmentation and editing**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/vuer-ai/feature-splatting
```

## Running Feature Splatting

Details for running Feature Splatting (built with Nerfstudio!) can be found [here](https://github.com/vuer-ai/feature-splatting).
Once installed, run:

```bash
ns-train feature-splatting --help
```

Currently, we provide the following variants:

| Method              | Description                                                     | Memory | Quality |
| -----------         | -----------------------------------------------                 | ------ | ------- |
| `feature-splatting` | Feature Splatting with MaskCLIP ViT-L/14@336px and MobileSAMv2  | ~8 GB  | Good    |

Note that the reference features used in this version are different from the version used in the paper in two ways

- The SAM-enhanced CLIP features are computed using MobileSAMv2, which is much faster than original SAM but slightly less accurate.
- The CLIP features are computed only on the image-level.

## Method

Feature splatting distills CLIP features into 3DGS by view-independent rasterization, which allows open-vocabulary 2D segmentation and open-vocabulary 3D segmentation of Gaussians directly in the 3D space. This implementation supports simple editing applications by directly manipulating Gaussians.

### Reference feature computation and joint supervision

Feature splatting computes high-quality SAM-enhanced CLIP features as reference features. Compared to coarse CLIP features (such as those used in LERF), Feature splatting performs an object-level masked average pooling of the features to refine the boundary of objects. While the original ECCV'24 paper uses SAM for part-level masks, this implementation uses MobileSAMv2 for much faster reference features computation, which we hope would encourage downstream applications that require real-time performance.

In addition to SAM-enhanced features, we also found that using DINOv2 features as a joint supervision helps regularize the internal structure of objects, which is similar to findings in existing work.

### Scene Editing

Thanks to the explicit representation of 3DGS, grouped Gaussians can be easily manipulated. While the original ECCV'24 paper proposes a series of editing primitives, to avoid introducing excessive dependencies or hacks, we support a subset of editing primitives in this implementation:

Rigid operations
- Floor estimation (for intuitive rotation and gravity estimation)
- Translation
- Transparent (highlights segmented object and turns background Gaussians transparent)
- Rotation (yaw only w.r.t. estimated ground)

Non-rigid operations
- Sand-like melting (based on Taichi MPM method)

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://feature-splatting.github.io/resources/ns_editing_compressed.mp4" type="video/mp4">
</video>

If you find our work helpful for your research, please consider citing

```none
@inproceedings{qiu-2024-featuresplatting,
    title={Language-Driven Physics-Based Scene Synthesis and Editing via Feature Splatting},
    author={Ri-Zhao Qiu and Ge Yang and Weijia Zeng and Xiaolong Wang},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2024}
}
```
