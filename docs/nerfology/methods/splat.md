# Splatfacto
<h4>Nerfstudio's Gaussian Splatting Implementation</h4>
<iframe width="560" height="315" src="https://www.youtube.com/embed/0yueTFx-MdQ?si=GxiYnFAeYVVl-soJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

```{button-link} https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
:color: primary
:outline:
Paper Website
```

[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) was proposed in SIGGRAPH 2023 from INRIA, and is a completely different method of representing radiance fields by explicitly storing a collection of 3D volumetric gaussians. These can be "splatted", or projected, onto a 2D image provided a camera pose, and rasterized to obtain per-pixel colors. Because rasterization is very fast on GPUs, this method can render much faster than neural representations of radiance fields.

To avoid confusion with the original paper, we refer to nerfstudio's implementation as "Splatfacto", which will drift away from the original as more features are added. Just as Nerfacto is a blend of various different methods, Splatfacto will be a blend of different gaussian splatting methodologies.

### Installation

```{button-link} https://docs.gsplat.studio/
:color: primary
:outline:
GSplat 
```

Nerfstudio uses [gsplat](https://github.com/nerfstudio-project/gsplat) as its gaussian rasterization backend, an in-house re-implementation which is designed to be more developer friendly. This can be installed with `pip install gsplat`. The associated CUDA code will be compiled the first time gsplat is executed. Some users with PyTorch 2.0 have experienced issues with this, which can be resolved by either installing gsplat from source, or upgrading torch to 2.1.

### Data
Gaussian splatting works much better if you initialize it from pre-existing geometry, such as SfM points from COLMAP. COLMAP datasets or datasets from `ns-process-data` will automatically save these points and initialize gaussians on them. Other datasets currently do not support initialization, and will initialize gaussians randomly. Initializing from other data inputs (i.e. depth from phone app scanners) may be supported in the future.

Because the method trains on *full images* instead of bundles of rays, there is a new datamanager in `full_images_datamanager.py` which undistorts input images, caches them, and provides single images at each train step.


### Running the Method
To run splatfacto, run `ns-train splatfacto --data <data>`. Just like NeRF methods, the splat can be interactively viewed in the web-viewer, loaded from a checkpoint, rendered, and exported.

We provide a few additional variants:

| Method           | Description                    | Memory | Speed   |
| ---------------- | ------------------------------ | ------ | ------- |
| `splatfacto`     | Default Model                  | ~6GB   | Fast    |
| `splatfacto-big` | More Gaussians, Higher Quality | ~12GB  | Slower  |


A full evalaution of Nerfstudio's implementation of Gaussian Splatting against the original Inria method can be found [here](https://docs.gsplat.studio/tests/eval.html).

#### Quality and Regularization
The default settings provided maintain a balance between speed, quality, and splat file size, but if you care more about quality than training speed or size, you can decrease the alpha cull threshold 
(threshold to delete translucent gaussians) and disable culling after 15k steps like so: `ns-train splatfacto --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.continue_cull_post_densification=False --data <data>`

A common artifact in splatting is long, spikey gaussians. [PhysGaussian](https://xpandora.github.io/PhysGaussian/) proposes a scale regularizer that encourages gaussians to be more evenly shaped. To enable this, set the `pipeline.model.use_scale_regularization` flag to `True`.

### Details
For more details on the method, see the [original paper](https://arxiv.org/abs/2308.04079). Additionally, for a detailed derivation of the gradients used in the gsplat library, see [here](https://arxiv.org/abs/2312.02121).

### Exporting splats
Gaussian splats can be exported as a `.ply` file which are ingestable by a variety of online web viewers. You can do this via the viewer, or `ns-export gaussian-splat --load-config <config> --output-dir exports/splat`. Currently splats can only be exported from trained splats, not from nerfacto.

Nerfstudio's splat export currently supports multiple third-party splat viewers:
- [Polycam Viewer](https://poly.cam/tools/gaussian-splatting)
- [Playcanvas SuperSplat](https://playcanvas.com/super-splat)
- [WebGL Viewer by antimatter15](https://antimatter15.com/splat/) 
- [Spline](https://spline.design/) 
- [Three.js Viewer by mkkellogg](https://github.com/mkkellogg/GaussianSplats3D)

### FAQ
- Can I export a mesh or pointcloud?

Currently these export options are not supported, but may be in the future. Contributions are always welcome!
- Can I render fisheye, equirectangular, orthographic images?

Currently, no. Gaussian rasterization assumes a perspective camera for its rasterization pipeline. Implementing other camera models is of interest but not currently planned.
