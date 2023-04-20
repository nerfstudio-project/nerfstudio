# Nerfbusters

<h4>Erasing Artifacts from Casually Captured NeRFs</h4>

```{button-link} https://ethanweber.me/Nerfbusters
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/ethanweber/Nerfbusters
:color: primary
:outline:
Code and Data
```

**TLDR: We present a method that uses a 3D diffusion prior to clean NeRFs and an evaluation procedure for in-the-wild NeRFs**

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://ethanweber.me/nerfbusters/media/car-fly-swipe.mp4" type="video/mp4">
</video>

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/ethanweber/nerfbusters
nerfbusters-setup
ns-train nerfbusters --help
```

For more details, see the [installation instructions](https://github.com/ethanweber/nerfbusters).
 
### Running the Method

Please checkout the readme for the [Nerfbusters repository](https://github.com/ethanweber/nerfbusters)

## Abstract
 
Casually captured Neural Radiance Fields (NeRFs) suffer from artifacts such as floaters or flawed geometry when rendered outside the path of the training views. However, common practice is still to evaluate on every 8th frame, which does not measure rendering quality away from training views, hindering progress in volume rendering. We propose a new dataset and evaluation procedure, where two camera trajectories are recorded of the scene, one used for training, and the other for evaluation. We find that existing hand-crafted regularizers do not remove floaters nor improve scene geometry in this more challenging in-the-wild setting. To this end, we propose a learned, local 3D diffusion prior and a novel density score distillation sampling loss. We show that this learned prior removes floaters and improves scene geometry for casual captures.

![teaser](https://ethanweber.me/nerfbusters/media/teaser.png)

Rendering NeRFs away from the training views results in artifacts, such as floaters or bad geometry. These artifacts are common for in in-the-wild captures (left) but are rarely seen in NeRF papers. The reason is that evaluation views are often part of the same path as the training views. We propose a new dataset and a more realistic evaluation procedure, where each scene is captured by two paths, one for training and one for evaluation (middle). We find that current regularizers are insufficient to improve the geometry and remove floaters in this more realistic evaluation setting. To this end, we propose Nerfbusters that learns a local 3D diffusion prior that improves scene geometry and reduces floaters. 

### Model Overview

We learn a local 3D prior with a diffusion model that regularizes the 3D geometry of NeRFs. We use importance sampling to query a cube with NeRF densities. We binarize these densities and perform one single denoising step using a pre-trained 3D diffusion model. With these denoised densities, we compute a density score distillation sampling (DSDS) that penalizes NeRF densities where the diffusion model predicts empty voxels and pushes the NeRF densities above the target w where the diffusion model predicts occupied voxels.

![Overview](https://ethanweber.me/nerfbusters/media/method.png)


### Visibiltiy Loss

Our visibility loss enables stepping behind or outside the training camera frustums. We accomplish this by supervising densities to be low when not seen by at least one training view. Other solutions would be to store an occupancy grid or compute ray-frustum intersection tests during rendering. Our solution is easy to implement and applicable to any NeRF.

![Visibility](https://ethanweber.me/nerfbusters/media/visibility_loss.png)

### Results and dataset preview

For results and a dataset preview, view the [project page](https://ethanweber.me/nerfbusters)!