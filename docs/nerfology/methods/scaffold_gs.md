# Scaffold-GS + GSDF

<h4>Unofficial implementations of "Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering" and "GSDF: 3DGS Meets SDF for Improved Neural Rendering and Reconstruction"</h4>

```{button-link} https://city-super.github.io/scaffold-gs/
:color: primary
:outline:
Scaffold-GS Website
```
```{button-link} https://city-super.github.io/GSDF/
:color: primary
:outline:
GSDF Website
```
```{button-link} https://github.com/brian-xu/scaffold-gs-nerfstudio
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://www.brian-xu.com/content/scaffold_gs_teaser.mp4" type="video/mp4">
</video>


### Installation
Ensure that nerfstudio has been installed according to the instructions. Then run the following command:
```
pip install git+https://github.com/brian-xu/scaffold-gs-nerfstudio
```
You must also install the correct torch_scatter for your environment (https://pytorch-geometric.com/whl/torch-2.1.2%2Bcu118.html).


### Running Model

This repository creates two new Nerfstudio methods, named "scaffold-gs" and "gsdf". To train with them, run the following commands:

```bash
ns-train scaffold-gs --data [PATH]
ns-train gsdf --data [PATH]
```

GSDF also supports mesh export. Follow the guide for [exporting geometry](../../quickstart/export_geometry.md). 

## Overview
Scaffold-GS replaces the Gaussian kernel described in [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) with neural Gaussians that are bound to anchor points. During rasterization, the attributes (opacity, color, scale, rotation) of these neural Gaussians are calculated with respect to the viewing direction and distance. This view-adaptive rendering improves results in challenging conditions, such as texture-less areas, insufficient observations, fine-scale details, view-dependent light effects and multi-scale observations.

GSDF is a dual-branch model that uses 3D Gaussian Splatting to supervise the training of a neural Signed Distance Field. The gaussian splatting branch provides depth maps to supervise SDF training. Once sufficiently initialized, the SDF can be used to guide the splitting and pruning of gaussians, ensuring geometric consistency and reducing floaters. 