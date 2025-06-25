# Instruct-GS2GS

<h4>Editing Gaussian Splatting Scenes with Instructions</h4>

```{button-link} https://instruct-gs2gs.github.io/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/cvachha/instruct-gs2gs
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-gs2gs.github.io/data/videos/face.mp4" type="video/mp4">
</video>

**Instruct-GS2GS enables instruction-based editing of 3D Gaussian Splatting scenes via a 2D diffusion model**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/cvachha/instruct-gs2gs
cd instruct-gs2gs
pip install --upgrade pip setuptools
pip install -e .
```

## Running Instruct-GS2GS

Details for running Instruct-GS2GS (built with Nerfstudio!) can be found [here](https://github.com/cvachha/instruct-gs2gs). Once installed, run:

```bash
ns-train igs2gs --help
```

| Method       | Description                  | Memory |
| ------------ | ---------------------------- | ------ |
| `igs2gs`       | Full model, used in paper    | ~15GB  |

Datasets need to be processed with COLMAP for Gaussian Splatting support.

Once you have trained your GS scene for 20k iterations, the checkpoints will be saved to the `outputs` directory. Copy the path to the `nerfstudio_models` folder. (Note: We noticed that training for 20k iterations rather than 30k seemed to run more reliably)

To start training for editing the GS, run the following command:

```bash
ns-train igs2gs --data {PROCESSED_DATA_DIR} --load-dir {outputs/.../nerfstudio_models} --pipeline.prompt {"prompt"} --pipeline.guidance-scale 12.5 --pipeline.image-guidance-scale 1.5
```

The `{PROCESSED_DATA_DIR}` must be the same path as used in training the original GS. Using the CLI commands, you can choose the prompt and the guidance scales used for InstructPix2Pix.

## Method

### Overview

Instruct-GS2GS is a method for editing 3D Gaussian Splatting (3DGS) scenes with text instructions in a method based on [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/). Given a 3DGS scene of a scene and the collection of images used to reconstruct it, this method uses an image-conditioned diffusion model ([InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix)) to iteratively edit the input images while optimizing the underlying scene, resulting in an optimized 3D scene that respects the edit instruction. The paper demonstrates that our proposed method is able to edit large-scale, real-world scenes, and is able to accomplish  realistic and targeted edits.


## Pipeline

<video id="pipeline" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-gs2gs.github.io/data/videos/pipeline.mp4" type="video/mp4">
</video>

This section will walk through each component of the Instruct-GS2GS method.

### How it Works

Instruct-GS2GS gradually updates a reconstructed Gaussian Splatting scene by iteratively updating the dataset images while training the 3DGS:

1. Images are rendered from the scene at all training viewpoints.
2. They get edited by InstructPix2Pix given a global text instruction.
3. The training dataset images are replaced with the edited images.
4. The 3DGS continues training as usual for 2.5k iterations.

### Editing Images with InstructPix2Pix

To update a dataset image from a given viewpoint, Instruct-GS2GS takes the original, unedited training image as image conditioning and uses the global text instruction as text conditioning. This process mixes the information of the diffusion model, which attempts to edit the image, the current 3D structure of the 3DGS, and view-consistent information from the unedited, ground-truth images. By combining this set of information, the edit is respected while maintaining 3D consistency.

The code snippet for how an image is edited in the pipeline can be found [here](https://github.com/cvachha/instruct-gs2gs/blob/main/igs2gs/ip2p.py).

### Iterative Dataset Update and Implementation

The method takes in a dataset of camera poses and training images, a trained 3DGS scene, and a user-specified text-prompt instruction, e.g. “make him a marble statue”. Instruct-GS2GS constructs the edited GS scene guided by the text-prompt by applying a 2D text and image conditioned diffusion model, in this case Instruct-Pix2Pix, to all training images over the course of training. It performs these edits using an iterative udpate scheme in which all training dataset images are updated using a diffusion model individually, for sequential iterations spanning the size of the training images, every 2.5k training iterations. This process allows the GS to have a holistic edit and maintain 3D consistency.

The process is similar to Instruct-NeRF2NeRF where for a given training camera view, it sets the original training image as the conditioning image, the noisy image input as the GS rendered from the camera combined with some randomly selected noise, and receives an edited image respecting the text conditioning. With this method, it is able to propagate the edited changes to the GS scene. The method is able to maintain grounded edits by conditioning Instruct-Pix2Pix on the original unedited training image.

This method uses Nerfstudio’s gsplat library for our underlying gaussian splatting model. We adapt similar parameters for the diffusion model from Instruct-NeRF2NeRF. Among these are the values that define the amount of noise (and therefore the amount signal retained from the original images). We vary the classifier-free guidance scales per edit and scene, using a range of values. We edit the entire dataset and then train the scene for 2.5k iterations. For GS training, we use L1 and LPIPS losses. We train our method for a maximum of 27.5k iterations (starting with a GS scene trained for 20k iterations). However, in practice we stop training once the edit has converged. In many cases, the optimal training length is a subjective decision — a user may prefer more subtle or more extreme edits that are best found at different stages of training.


## Results

For results, view the [project page](https://instruct-gs2gs.github.io/)!

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-gs2gs.github.io/data/videos/campanile_all.mp4" type="video/mp4">
</video>