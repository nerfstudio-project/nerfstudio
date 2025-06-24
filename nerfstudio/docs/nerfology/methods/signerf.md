# SIGNeRF

<h3> Scene Integrated Generation for Neural Radiance Fields</h3>

```{button-link} https://signerf.jdihlmann.com/
:color: primary
:outline:
Website & Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://signerf.jdihlmann.com/videos/SIGNeRF_small.mp4" type="video/mp4">
</video>

**Generatively edits NeRF scenes in a controlled and fast manner.**

SIGNeRF allows for generative 3D scene editing. We present a novel approach to combine [NeRFs](https://www.matthewtancik.com/nerf) as scene representation with the image diffusion model [StableDiffusion](https://github.com/Stability-AI/stablediffusion) to allow fast and controlled 3D generation.

## Installation

Install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/cgtuebingen/SIGNeRF
```

SIGNeRF requires to use [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). For detailed installation information follow our [installation guide](https://github.com/cgtuebingen/SIGNeRF?tab=readme-ov-file#installation).

## Running SIGNeRF

Details for running SIGNeRF can be found [here](https://github.com/cgtuebingen/SIGNeRF). Once installed, run:

```bash
ns-train signerf --help
```

Two variants of SIGNeRF are provided:

| Method             | Description                | Time  | Quality |
| ------------------ | -------------------------- | ----- | ------- |
| `signerf`          | Full model, used in paper  | 40min | Best    |
| `signerf_nerfacto` | Faster model with Nerfacto | 20min | Good    |

For more information on hardware requirements and training times, please refer to the [training section](https://github.com/cgtuebingen/SIGNeRF?tab=readme-ov-file#training-1).

## Interface

<img src="https://github.com/cgtuebingen/SIGNeRF/raw/main/images/interface.png" width="100%" alt="SIGNeRF Interface" />

SIGNeRF fully integrates into the Nerfstudio [viser](https://viser.studio) interface. It allows for easy editing of NeRF scenes with a few simple clicks. The user can select the editing method, the region to edit, and the object to insert. The reference sheet can be previewed and the NeRF is fine-tuned on the edited images. If you are interested in how we fully edit the Nerfstudio interface please find our code [here](https://github.com/cgtuebingen/SIGNeRF).

## Method

### Overview

SIGNeRF is a novel approach for fast and controllable NeRF scene editing and scene-integrated object generation. We introduce a new generative update strategy that ensures 3D consistency across the edited images, without requiring iterative optimization. We find that depth-conditioned diffusion models inherently possess the capability to generate 3D consistent views by requesting a grid of images instead of single views. Based on these insights, we introduce a multi-view reference sheet of modified images. Our method updates an image collection consistently based on the reference sheet and refines the original NeRF with the newly generated image set in one go. By exploiting the depth conditioning mechanism of the image diffusion model, we gain fine control over the spatial location of the edit and enforce shape guidance by a selected region or an external mesh.

For an in-depth visual explanation and our results please watch our [videos](https://www.youtube.com/playlist?list=PL5y23CB9WmildtW3QyMEi3arXg06zB4ex) or read our [paper](https://arxiv.org/abs/2401.01647).

## Pipeline

<video id="pipeline" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://signerf.jdihlmann.com/videos/pipeline_compressed.mp4" type="video/mp4">
</video>

We leverage the strengths of [ControlNet](https://github.com/lllyasviel/ControlNet), a depth condition image diffusion model, to edit an existing NeRF scene. We do so with a few simple steps in a single forward pass:

0. We start with an original NeRF scene and select an editing method / region
1. For object generation, we place a mesh object into the scene
2. And control the precise location and shape of the edit
3. We position reference cameras in the scene
4. Render the corresponding color, depth, and mask images, and arrange them into image grids
5. These grids are used to generate the reference sheet with conditioned image diffusion
6. Generate new edited images consistent with the reference sheet by leveraging an inpainting mask.

- Repeat step (6) for all cameras

7. Finally, the NeRF is fine-tuned on the edited images

### Reference Sheet Generation

<img src="https://arxiv.org/html/2401.01647v2/x4.png" width="100%" alt="Reference Sheet Generation" />

SIGNeRF uses a novel technique called reference sheet generation. We observe that the image
diffusion model [ControlNet](https://github.com/lllyasviel/ControlNet) can already generate multiview consistent images of a scene without the need for iterative refinement like [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/). While generating individual views sequentially introduces too much variation to integrate them into a consistent 3D model, arranging them in a grid of images that are processed by ControlNet in one pass significantly
improves the multi-view consistency. Based on the depth maps rendered from the original NeRF
scene we employ a depth-conditioned inpainting variant of ControlNet to generate such a reference sheet of the edited scene. A mask specifies the scene region where the generation should occur. This step gives a lot of control to the user. Different appearances can be produced by generating reference sheets with different seeds or prompts. The one sheet finally selected will directly determine the look of the final 3D scene.

If you want to learn more about the method, please read our [paper](https://arxiv.org/abs/2401.01647) or read the breakdown of the method in the [Radiance Fields article](https://radiancefields.com/controlnet-nerfs-signerf/).

## Related Work

SIGNeRF was inspired by the work of [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/) which is also available in the [Nerfology](https://docs.nerf.studio/nerfology/methods/in2n.html) documentation.

## Authors

SIGNeRF was developed by [Jan-Niklas Dihlmann](https://jdihlmann.com) and [Andreas Engelhardt](https://aengelhardt.com/).
