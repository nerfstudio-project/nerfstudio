# Export geometry

Here we document how to export point clouds and meshes from nerfstudio. The main command you'll be working with is `ns-export`. Our point clouds are exported as `.ply` files and the textured meshes are exported as `.obj` files.

## Exporting a mesh

### 1. TSDF Fusion

TSDF (truncated signed distance function) Fusion is a meshing algorithm that uses depth maps to extract a surface as a mesh. This method works for all models.

```python
ns-export tsdf --load-config CONFIG.yml --output-dir OUTPUT_DIR
```

### 2. Poisson surface reconstruction

Poisson surface reconstruction gives the highest quality meshes. See the steps below to use Poisson surface reconstruction in our repo.

> **Note:**
> This will only work with a Model that computes or predicts normals, e.g., nerfacto.


1. Train nerfacto with network settings that predict normals.

```bash
ns-train nerfacto --pipeline.model.predict-normals True
```

2. Export a mesh with the Poisson meshing algorithm.

```bash
ns-export poisson --load-config CONFIG.yml --output-dir OUTPUT_DIR
```

## Exporting a point cloud

```bash
ns-export pointcloud --help
```

## Other exporting methods

Run the following command to see other export methods that may exist.

```python
ns-export --help
```

## Texturing an existing mesh with NeRF

Say you want to simplify and/or smooth a mesh offline, and then you want to texture it with NeRF. You can do that with the following command. It will work for any mesh filetypes that [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/) can support, for example a `.ply`.

```python
python scripts/texture.py --load-config CONFIG.yml --input-mesh-filename FILENAME --output-dir OUTPUT_DIR
```

## Dependencies

Our dependencies are shipped with the pip package in the pyproject.toml file. These are the following:

- [xatlas-python](https://github.com/mworchel/xatlas-python) for unwrapping meshes to a UV map
- [pymeshlab](https://pymeshlab.readthedocs.io/en/latest/) for reducing the number of faces in a mesh
