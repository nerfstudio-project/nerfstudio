# From NeRFs to Gaussian Splats, and Back

This is the implementation of [From NeRFs to Gaussian Splats, and Back](https://arxiv.org/abs/2405.09717); An efficient procedure to convert back and forth between NeRF and GS, and thereby get the best of both approaches. New dataset in the paper can be downloaded from [this google drive link](https://drive.google.com/drive/folders/1xvbONL4EVgHxaHMsV101455l_jNgyaUM?usp=sharing). The official code can be found [here](https://github.com/grasp-lyrl/NeRFtoGSandBack).

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://github.com/grasp-lyrl/NeRFtoGSandBack/blob/main/assets/Giannini_demo_compressed.mp4" type="video/mp4">
</video>

## Installation
This repository follows the nerfstudio method [template](https://github.com/nerfstudio-project/nerfstudio-method-template/tree/main)

### 0. Install Nerfstudio dependencies
Please follow the Nerfstudio [installation guide](https://docs.nerf.studio/quickstart/installation.html)  to create an environment and install dependencies.

### 1. Install the repository
Clone and navigate into this repository. Run the following commands:

`pip install -e nerfsh`

and

`pip install -e nerfgs`.

Finally, run `ns-install-cli`.

### 2. Check installation
Run `ns-train --help`. You should be able to find two methods, `nerfsh` and `nerfgs`, in the list of methods.

## Downloading data
You could download the Giannini-Hall and aspen datasets from [this google drive link](https://drive.google.com/drive/folders/19TV6kdVGcmg3cGZ1bNIUnBBMD-iQjRbG). Our new dataset (Wissahickon and Locust-Walk) can be downloaded from [this google drive link](https://drive.google.com/drive/folders/1xvbONL4EVgHxaHMsV101455l_jNgyaUM?usp=sharing).

## NeRFGS: NeRFs to Gaussian Splats
### Training NeRF-SH
Run the following command for training. Replace `DATA_PATH` with the data directory location.

`ns-train nerfsh --data DATA_PATH --pipeline.model.camera-optimizer.mode off `

To train on Wissahickon or Locust-Walk dataset, you need to add `nerfstudio-data --eval-mode filename` to properly split training and validation data, i.e.,

`ns-train nerfsh --data DATA_PATH --pipeline.model.camera-optimizer.mode off nerfstudio-data --eval-mode filename`


### NeRFGS: Converting NeRF-SH to Guassian splats
Replace `CONFIG_LOCATION` with the location of config file saved after training.

`ns-export-nerfsh --load-config CONFIG_LOCATION --output-dir exports/nerfgs/ --num-points 2000000 --remove-outliers True --normal-method open3d --use_bounding_box False`

### Visualize converted Gaussian splats
Replace `DATA_PATH` with the data directory location. You also need to add `nerfstudio-data --eval-mode filename` if train on Wissahickon or Locust-Walk.

`ns-train nerfgs --data DATA_PATH --max-num-iterations 1 --pipeline.model.ply-file-path exports/nerfgs/nerfgs.ply`

### Fintuned NeRFGS 
We reduces the learning rate for finetuning. You also need to add `nerfstudio-data --eval-mode filename` if train on Wissahickon or Locust-Walk.

`ns-train nerfgs --data DATA_PATH --pipeline.model.ply-file-path exports/nerfgs/nerfgs.ply --max-num-iterations 5000 --pipeline.model.sh-degree-interval 0 --pipeline.model.warmup-length 100 --optimizers.xyz.optimizer.lr 0.00001 --optimizers.xyz.scheduler.lr-pre-warmup 0.0000001 --optimizers.xyz.scheduler.lr-final 0.0000001 --optimizers.features-dc.optimizer.lr 0.01 --optimizers.features-rest.optimizer.lr 0.001 --optimizers.opacity.optimizer.lr 0.05 --optimizers.scaling.optimizer.lr 0.01 --optimizers.rotation.optimizer.lr 0.0000000001 --optimizers.camera-opt.optimizer.lr 0.0000000001 --optimizers.camera-opt.scheduler.lr-pre-warmup 0.0000000001 --optimizers.camera-opt.scheduler.lr-final 0.0000000001`

## GSNeRF: Gaussian Splats to NeRFs

### Scene modification
Coming soon

### Rendering new training images
In the new dataset, training images are rendered from splats. Replace `CONFIG_LOCATION` with the location of config file saved after training.

`ns-nerfgs-render --load-config CONFIG_LOCATION --render-output-path exports/splatting_data --export-nerf-gs-data`

### GSNeRF: Training on new training images
`ns-train nerfsh --data exports/splatting_data --pipeline.model.camera-optimizer.mode off nerfstudio-data --eval-mode filename`

## Extending the method
The conversion from NeRF to GS has inefficiency as mentioned at the discussion section of the paper. We welcome your efforts to reduce the inefficiency! The code for conversion is mainly in `nerfsh/nerfsh/nerfsh_exporter.py`.

## Method
<p align="center">
  <img src="https://github.com/grasp-lyrl/NeRFtoGSandBack/blob/main/assets/overview.jpg">
</p>

### NeRF-SH
The NeRF-SH field structure is shown above in the overview figure. NeRF-SH is modified from Nerfacto to predict spherical harmonics (degree 3 by default) for each rgb channel. The volumetric rendering process remains unchanges: at each point along a ray, we predict the spherical harmonics and calculate color based on the view direction. 

### NeRFGS
Given a trained NeRF-SH, we extract pointcloud based on rendered depth, following the pointcloud export pipeline in Nerfstudio. In addition to exporting the location of each point, we export spherical harmonic coefficients and density predicted by NeRF-SH. We exclude rays with low opacity or corresponding to the sky. We initialize each Gaussian as isotropic where the scale depends on the sparsity of points in the neighborhood. Specifically, the scale of each Gaussian is half of the average distance between each point and its three nearest neighbors. To avoid large Gaussians, the scale is clipped between 0 and 0.8-th quantile of the scales in the scene. The exported Gaussian splats already captures the geometric and photometric properties of the scene. To obtain fine-grained splats and remove outliers during exportation, we finetune the splats using training views.  

### GSNeRF
After editing the Gaussian splats, rendered training views from edited Gaussian splats can be used to create a new dataset. The new dataset can be used to train/update other models (especially implicit representations that are difficult to edited directly). 