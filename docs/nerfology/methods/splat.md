# Gaussian Splatting
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) was proposed in SIGGRAPH 2023 from INRIA, and is a completely 
different method of representing radiance fields by explicitly storing a collection of 3D volumetric gaussians. These can be "splatted", or projected, onto a 2D image
provided a camera pose, and rasterized to obtain per-pixel colors. Because rasterization is very fast on GPUs, this method can render much faster than neural representations
of radiance fields.

### Installation
[] talk about gsplat installation, if pip doesn't work use git

### Data
todo:
[] talk about colmap initialization
[] talk about how it currently assumes a colmap dataset type for loading 3d points
[] what about other data sources? polycam etc, not yet supported but in the works


### Running the Method

### Details
todo:
brief summary, link to original paper, link to gsplat, link to vickie's gradient derivation

### Exporting splats
todo:
discuss different web viewers, talk about export script