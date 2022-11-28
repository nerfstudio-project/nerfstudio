# Export geometry

Here we document how to export point clouds and meshes from nerfstudio. The main command you'll be working with is `ns-export`. Our point clouds are exported as `.ply` files and the textured meshes are exported as `.obj` files.

## Install extra dependencies

If you want to texture the mesh with `--unwrap-method xatlas`, then you need to install the following package shown below. We use PyTorch to rasterize the mesh and interpolate vertices and normals in the texture image.

## Exporting a mesh

> Poisson surface reconstruction

    We default to using Poisson surface reconstruction implemented in Open3D, followed by higher resolution texturing by using a texture image and querying the NeRF. See the steps below to train your own model and export as a mesh.

    1. Train nerfacto with network settings that predict normals.

    ```python
    ns-train nerfacto --pipeline.model.predict-normals True
    ```

    2. Export a mesh with the Poisson algorithm.

    ```python
    ns-export poisson --load-config CONFIG.yml --output-dir OUTPUT_DIR
    ```

> TSDF Fusion

    Run the folowing command to see non-default export commands. For example, `ns-export tsdf` will run TSDF Fusion on depth and color images from NeRF, followed by texturing. This will work for models that don't have predicted normals.

    ```python
    ns-export tsdf --load-config CONFIG.yml --output-dir OUTPUT_DIR
    ```

## Exporting a point cloud

## Other exporting methods

Run the folowing command to see non-default export commands.

```python
ns-export --help
```

<hr>

## Texturing an existing mesh with NeRF

Say you want to simplify and/or smooth a mesh offline, and then you want to texture it with NeRF. You can do that with the following command. It will work for any mesh filetypes that trimesh can support, for example a `.ply`.

```python
python scripts/texture.py --load-config CONFIG.yml --output-dir OUTPUT_DIR --input-mesh-filename FILENAME
```
