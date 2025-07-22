# LiveScene

<h4>Language Embedding Interactive Radiance Fields for Physical Scene Rendering and Control</h4>

```{button-link} https://tavish9.github.io/livescene//
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/Tavish9/livescene/
:color: primary
:outline:
Code
```

<video id="demo" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://tavish9.github.io/livescene//static/video/demo.mp4" type="video/mp4">
</video>

**The first scene-level language-embedded interactive radiance field, which efficiently reconstructs and controls complex physical scenes, enabling manipulation of multiple articulated objects and language-based interaction.**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/Tavish9/livescene
```

## Running LiveScene

Details for running LiveScene (built with Nerfstudio!) can be found [here](https://github.com/Tavish9/livescene).
Once installed, run:

```bash
ns-train livescene --help
```

There is only one default configuration provided. However, you can run it for different datasets.

The default configurations provided is:

| Method      | Description                                     | Memory | Quality |
| ----------- | ----------------------------------------------- | ------ | ------- |
| `livescene` | LiveScene with OpenCLIP ViT-B/16, used in paper | ~8 GB  | Good    |

There are two new dataparser provider for LiveScene:

| Method           | Description                     | Scene type        |
| ---------------- | ------------------------------- | ----------------- |
| `livescene-sim`  | OmniSim dataset for LiveScene   | Synthetic dataset |
| `livescene-real` | InterReal dataset for LiveScene | Real dataset      |

## Method

LiveScene proposes an efficient factorization that decomposes the interactive scene into multiple local deformable fields to separately reconstruct individual interactive objects, achieving the first accurate and independent control on multiple interactive objects in a complex scene. Moreover, LiveScene introduces an interaction-aware language embedding method that generates varying language embeddings to localize individual interactive objects under different interactive states, enabling arbitrary control of interactive objects using natural language.

### Overview

Given a camera view and control variable $\boldsymbol{\kappa}$ of one specific interactive object, a series 3D points are sampled in a local deformable field that models the interactive motions of this specific interactive object, and then the interactive object with novel interactive motion state is generated via volume-rendering. Moreover, an interaction-aware language embedding is utilized to localize and control individual interactive objects using natural language.

<img id="livescene_pipeline" src="https://tavish9.github.io/livescene//static/image/pipeline.png" style="background-color:white;" width="100%">

### Multi-scale Interaction Space Factorization

LiveScene maintains mutiple local deformable fields $\left \{\mathcal{R}_1, \mathcal{R}\_2, \cdots \mathcal{R}_\alpha \right \}$ for each interactive object in the 4D space, and project high-dimensional interaction features into a compact multi-scale 4D space. In training, LiveScene denotes a feature repulsion loss and to amplify the feature differences between distinct deformable scenes, which relieve the boundary ray sampling and feature storage conflicts.

<img id="livescene_factorization" src="https://tavish9.github.io/livescene//static/image/decompose.png" style="background-color:white;" width="100%">

### Interaction-Aware Language Embedding

LiveScene Leverages the proposed multi-scale interaction space factorization to efficiently store language features in lightweight planes by indexing the maximum probability sampling instead of 3D fields in LERF. For any sampling point $\mathbf{p}$, it retrieves local language feature group, and perform bilinear interpolation to obtain a language embedding that adapts to interactive variable changes from surrounding clip features.

<img id="livescene_language" src="https://tavish9.github.io/livescene//static/image/embeds.png" style="background-color:white;" width="100%">

## Dataset

To our knowledge, existing view synthetic datasets for interactive scene rendering are primarily limited to a few interactive objects, making it impractical to scale up to real scenarios involving multi-object interactions. To bridge this gap, we construct two scene-level, high-quality annotated datasets to advance research progress in reconstructing and understanding interactive scenes: OminiSim and InterReal, containing 28 subsets and 70 interactive objects with 2 million samples, providing rgbd images, camera trajectories, interactive object masks, prompt captions, and corresponding object state quantities at each time step.

<video id="dataset" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://tavish9.github.io/livescene//static/video/livescene_dataset.mp4" type="video/mp4">
</video>

## Interaction

For more interaction with viewer, please see [here](https://github.com/Tavish9/livescene?tab=readme-ov-file#3-interact-with-viewer).

## BibTeX

If you find our work helpful for your research, please consider citing

```none
@misc{livescene2024,
    title={LiveScene: Language Embedding Interactive Radiance Fields for Physical Scene Rendering and Control},
    author={Delin Qu, Qizhi Chen, Pingrui Zhang, Xianqiang Gao, Bin Zhao, Zhigang Wang, Dong Wang†, Xuelong Li†},
    year={2024},
    eprint={2406.16038},
    archivePrefix={arXiv},
  }
```
