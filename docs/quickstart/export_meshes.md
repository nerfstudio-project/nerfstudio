# Export meshes

Here we document how to export point clouds and meshes from nerfstudio. The main command you'll be working with is `ns-export`.

## Exporting a mesh

We default to using Poisson surface reconstruction implemented in Open3D, followed by higher resolution texturing by using a texture image and querying the NeRF. See the steps below to train your own model and export as a mesh.

1. Train nerfacto with network settings that predict normals.

```python
ns-train nerfacto --pipeline.model.predict-normals True
```

2. Export a mesh with the Poisson algorithm.

```python
ns-export poisson --load-config CONFIG.yml --output-dir exports/mesh
```

## Exporting a point cloud

## Non-default exporting methods

Run the folowing command to see non-default export commands. For example, `ns-export tsdf` will run TSDF Fusion on depth and color images from NeRF, followed by texturing.

```python
ns-export --help
```

## Texturing an existing mesh with NeRF

We don't support this in the pip package, but if you install the repo locally, you can run the following command to texture an existing mesh. It will work for any mesh filetypes that trimesh can support, for example a `.ply`.

```python
python scripts/texture.py --load-config CONFIG.yml --output-dir OUTPUT-DIR --input-ply-filename FILENAME
```
