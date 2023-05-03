# LERF

<h4>📎 Language Embedded Radiance Fields 🚜</h4>

```{button-link} https://www.lerf.io/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/kerrj/lerf
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://www.lerf.io/data/teaser.mp4" type="video/mp4">
</video>

**Grounding CLIP vectors volumetrically inside NeRF allows flexible natural language queries in 3D**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/kerrj/lerf
```

## Running LERF

Details for running LERF (built with Nerfstudio!) can be found [here](https://github.com/kerrj/lerf). Once installed, run:

```bash
ns-train lerf --help
```

Three variants of LERF are provided:

| Method      | Description                                     | Memory | Quality |
| ----------- | ----------------------------------------------- | ------ | ------- |
| `lerf-big`  | LERF with OpenCLIP ViT-L/14                     | ~22 GB | Best    |
| `lerf`      | Model with OpenCLIP ViT-B/16, used in paper     | ~15 GB | Good    |
| `lerf-lite` | LERF with smaller network and less LERF samples | ~8 GB  | Ok      |

`lerf-lite` should work on a single NVIDIA 2080.
`lerf-big` is experimental, and needs further tuning.

## Method

LERF enables pixel-aligned queries of the distilled 3D CLIP embeddings without relying on region proposals, masks, or fine-tuning, supporting long-tail open-vocabulary queries hierarchically across the volume.

### Multi-scale supervision

To supervise language embeddings, we pre-compute an image pyramid of CLIP features for each training view. Then, each sampled ray during optimization is supervised by interpolating the CLIP embedding within this pyramid.

<img id="lerf_multiscale" src="https://www.lerf.io/data/clip_features.png" style="background-color:white;" width="100%">

### LERF Optimization

LERF optimizes a dense, multi-scale language 3D field by volume rendering CLIP embeddings along training rays, supervising these embeddings with multi-scale CLIP features across multi-view training images.

Inspired by Distilled Feature Fields (DFF), we use DINO features to regularize CLIP features. This leads to qualitative improvements in object boundaries, as CLIP embeddings in 3D can be sensitive to floaters and regions with few views.

After optimization, LERF can extract 3D relevancy maps for language queries interactively in real-time.

<img id="lerf_render" src="https://www.lerf.io/data/nerf_render.png" style="background-color:white;" width="100%">

### Visualizing relevancy for text queries

Set the "Output Render" type to `relevancy_0`, and enter the text query in the "LERF Positives" textbox (see image). The output render will show the 3D relevancy map for the query. View the [project page](https://lerf.io) for more examples and details about the relevancy map normalization.

<center>
<img id="lerf_viewer" src="https://www.lerf.io/data/lerf_screen.png" width="40%">
</center>

## Results

For results, view the [project page](https://lerf.io)!

Datasets used in the original paper can be found [here](https://drive.google.com/drive/folders/1LUzwEvBCE19PNYcwfmrG-9FLpZLbi4on?usp=sharing).

```none
@article{lerf2023,
 author = {Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
 title = {LERF: Language Embedded Radiance Fields},
 journal = {arXiv preprint arXiv:2303.09553},
 year = {2023},
}
```
