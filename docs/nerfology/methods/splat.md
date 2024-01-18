# Gaussian Splatting
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) was proposed in SIGGRAPH 2023 from INRIA, and is a completely different method of representing radiance fields by explicitly storing a collection of 3D volumetric gaussians. These can be "splatted", or projected, onto a 2D image provided a camera pose, and rasterized to obtain per-pixel colors. Because rasterization is very fast on GPUs, this method can render much faster than neural representations of radiance fields.

### Installation
Nerfstudio uses [gsplat](https://github.com/nerfstudio-project/gsplat) as its gaussian rasterization backend, an in-house re-implementation which is designed to be more developer friendly. This can be installed with `pip install gsplat`. The associated CUDA code will be compiled the first time gaussian splatting is executed. Some users with PyTorch 2.0 have experienced issues with this, which can be resolved by either installing gsplat from source, or upgrading torch to 2.1.

### Data
Gaussian Splatting works much better if you initialize it from pre-existing geometry, such as SfM points from COLMAP. COLMAP datasets or datasets from `ns-process-data` will automatically save these points and initialize gaussians on them. Other datasets currently do not support initialization, and will initialize gaussians randomly. Initializing from other data inputs (i.e. depth from phone app scanners) may be supported in the future.

Because gaussian splatting trains on *full images* instead of bundles of rays, there is a new datamanager in `full_images_datamanager.py` which undistorts input images, caches them, and provides single images at each train step.


### Running the Method
To run gaussian splatting, run `ns-train gaussian-splatting --data <data>`. Just like NeRF methods, the splat can be interactively viewed in the web-viewer, loaded from a checkpoint, rendered, and exported.

### Details
For more details on the method, see the [original paper](https://arxiv.org/abs/2308.04079). Additionally, for a detailed derivation of the gradients used in the gsplat library, see [here](https://arxiv.org/abs/2312.02121).

### Exporting splats
Gaussian splats can be exported as a `.ply` file which are ingestable by a variety of online web viewers. You can do this via the viewer, or `ns-export gaussian-splat --load-config <config> --output-dir exports/splat`. Currently splats can only be exported from trained splats, not from nerfacto.

Nerfstudio gaussian splat exports currently supports multiple third-party splat viewers:
- [Polycam Viewer](https://poly.cam/tools/gaussian-splatting)
- [Playcanvas SuperSplat](https://playcanvas.com/super-splat)
- [WebGL Viewer by antimatter15](https://antimatter15.com/splat/) 
- [Spline](https://spline.design/) 
- [Three.js Viewer by mkkellogg](https://github.com/mkkellogg/GaussianSplats3D)

### FAQ
- Can I export a mesh or pointcloud?
Currently these export options are not supported, but may become in the future. Contributions are always welcome!
- Can I render fisheye, equirectangular, orthographic images?
Currently, no. Gaussian splatting assumes a perspective camera for its rasterization pipeline. Implementing other camera models is of interest but not currently planned.
