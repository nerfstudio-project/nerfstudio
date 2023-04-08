# Instruct-NeRF2NeRF

<h4>Editing 3D Scenes with Instructions</h4>

```{button-link} https://instruct-nerf2nerf.github.io/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/ayaanzhaque/instruct-nerf2nerf
:color: primary
:outline:
Code
```

```{image} imgs/in2n/teaser.png
:align: center
```

<h5>TL;DR: Instruct-NeRF2NeRF enables instruction-based editing of NeRFs via a 2D diffusion model</h5>

### Running the Method

Details for running Instruct-NeRF2NeRF (built with Nerfstudio!) can be found [here](https://github.com/ayaanzhaque/instruct-nerf2nerf). Once installed, run 

```bash
ns-train in2n --help
```
 
## Method
 
### Overview
 
We propose a method for editing NeRF scenes with text-instructions. Given a NeRF of a scene and the collection of images used to reconstruct it, our method uses an image-conditioned diffusion model ([InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix)) to iteratively edit the input images while optimizing the underlying scene, resulting in an optimized 3D scene that respects the edit instruction. We demonstrate that our proposed method is able to edit large-scale, real-world scenes, and is able to accomplish more realistic, targeted edits than prior work.
 
## Pipeline
 
```{image} imgs/in2n/pipeline_figure.png
:align: center
```

In this section, we will walk through each component of the Instruct-NeRF2NeRF method.
 
### How it Works

Our method gradually updates a reconstructed NeRF scene by iteratively updating the dataset images while training the NeRF:

1. An image is rendered from the scene at a training viewpoint.
2. It is edited by InstructPix2Pix given a global text instruction.
3. The training dataset image is replaced with the edited image.
4. The NeRF continues training as usual.

### Editing Images with InstructPix2Pix
 
InstructPix2Pix is an image-editing diffusion model which can be prompted using text instructions. More details on InstructPix2Pix can be found [here](https://www.timothybrooks.com/instruct-pix2pix).

Traditionally, at inference time, InstructPix2Pix takes as input random noise and is conditioned on an image (the image to edit) and a text instruction. The strength of the edit can be controlled using the image and text classifier-free guidance scales.

To update a dataset image a given viewpoint, we first take the original, unedited training image as our image conditioning and use the global text instruction as our text conditioning. To construct the main input to the diffusion model, we input a noised version of the current render from the given viewpoint. The noise is sampled from a normal distribution and scaled based on a randomly chosen timestep. Then InstructPix2Pix slowly denoises the rendered image by predicting the noised version of the image at previous timesteps until the image is fully denoised. This will produce an edited version of the input image.

This process mixes the information of the diffusion model, which attempts to edit the image, the current 3D structure of the NeRF, and view-consistent information from the unedited, ground-truth images. By combining this set of information, we are able to edit our underlying NeRF while maintaining 3D consistency.

The code snippet for how an image is edited in our pipeline can be found [here](https://github.com/ayaanzhaque/instruct-nerf2nerf/blob/main/in2n/ip2p.py).

### Iterative Dataset Update

When NeRF training starts, our dataset consists of the original, unedited images used to train the original scene. We save these images separately to use as conditioning for InstructPix2Pix. At each optimization iteration, we perform some number of NeRF optimization steps, and then update some number of images (often just one image). The images are randomly ordered prior to training and then at each step, the images are chosen in order to edit. Once an image has been edited, we replace it in the dataset. Importantly, at each NeRF step, we sample rays across the entire dataset, meaning there is a mixed source of supervision between edited images and unedited images. This allows for a gradual optimization that balances maintaining the underlying structure of the NeRF as well as the actual edit we want to perform.

At early iterations of this process, the edited images may be inconistent with one another, as InstructPix2Pix often doesn't perform consistent edits across viewpoints. However, over time, since images are edited using the current render of the NeRF, the edits begin to converge towards a globally consistent depiction of the underlying scene. Here is an example of how our underlying dataset evolves and becomes more consistent.

```{image} imgs/in2n/dataset_update.png
:align: center
```

The traditional method for supervising NeRFs using diffusion models is to use a Score Distillation Sampling (SDS) loss, as proposed in [DreamFusion](https://dreamfusion3d.github.io/). Our method can be viewed as a variant of SDS, as instead of updating a discrete set of images at each step, our loss is a mix of rays from various viewpoints which are edited to varying degrees. We find that this leads to higher quality performance and more stable optimization.

## Results

For results, view our [project page](https://instruct-nerf2nerf.github.io/)!

## Code

As outlined on [this page](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html), we host our code in an external repository which imports Nerfstudio as a dependency. For more information on how you can run Instruct-NeRF2NeRF yourself, visit the README of our [code repository](https://github.com/ayaanzhaque/instruct-nerf2nerf).