# OpenNeRF
## OpenSet 3D Neural Scene Segmentation with Pixel-wise Features and Rendered Novel Views

```{button-link} https://opennerf.github.io/
:color: primary
:outline:
Paper Website
```

## Installation

Detailed installation instructions can be found at our [Project Page](https://github.com/opennerf/opennerf?tab=readme-ov-file#installation).

## Running OpenNeRF

Once installed, run:

```bash
ns-train opennerf --help
```

## Method

<img src=https://opennerf.github.io/static/images/teaser.png />

### Overview

Large visual-language models (VLMs), like CLIP, enable open-set image segmentation to segment arbitrary concepts from an image in a zero-shot manner. This goes beyond the traditional closed-set assumption, i.e., where models can only segment classes from a pre-defined training set. More recently, first works on open-set segmentation in 3D scenes have appeared in the literature. These methods are heavily influenced by closed-set 3D convolutional approaches that process point clouds or polygon meshes. However, these 3D scene representations do not align well with the image-based nature of the visual-language models. Indeed, point cloud and 3D meshes typically have a lower resolution than images and the reconstructed 3D scene geometry might not project well to the underlying 2D image sequences used to compute pixel-aligned CLIP features. To address these challenges, we propose OpenNeRF which naturally operates on posed images and directly encodes the VLM features within the NeRF. This is similar in spirit to LERF, however our work shows that using pixel-wise VLM features (instead of global CLIP features) results in an overall less complex architecture without the need for additional DINO regularization. Our OpenNeRF further leverages NeRF's ability to render novel views and extract open-set VLM features from areas that are not well observed in the initial posed images. For 3D point cloud segmentation on Replica, OpenNeRF outperforms recent open-vocabulary methods such as LERF and OpenScene by at least +4.9 mIoU.

## Zero-Shot 3D Semantic Segmentation on Replica

<img src="https://opennerf.github.io/static/visualizations/final_office0_.gif" />

<img src="https://opennerf.github.io/static/visualizations/final_office1_.gif" />

<img src="https://opennerf.github.io/static/visualizations/final_office2_.gif" />

## BibTeX
```
@inproceedings{engelmann2024opennerf,
  title={{OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views}},
  author={Engelmann, Francis and Manhardt, Fabian and Niemeyer, Michael and Tateno, Keisuke and Pollefeys, Marc and Tombari, Federico},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
