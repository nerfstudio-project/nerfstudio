# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module implements the Marching Cubes algorithm for extracting
isosurfaces
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh
from jaxtyping import Bool, Float
from skimage import measure
from torch import Tensor

avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")


def create_point_pyramid(points: Float[Tensor, "3 height width depth"]) -> List[Float[Tensor, "3 height width depth"]]:
    """
    Create a point pyramid for multi-resolution evaluation.

    Args:
        points: A torch tensor containing 3D points.

    Returns:
        A list of torch tensors representing points at different resolutions.
    """
    points_pyramid = [points]
    for _ in range(3):
        points = avg_pool_3d(points[None])[0]
        points_pyramid.append(points)
    points_pyramid = points_pyramid[::-1]
    return points_pyramid


def evaluate_sdf(sdf: Callable[[Tensor], Tensor], points: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch"]:
    """
    Evaluate a signed distance function (SDF) for a batch of points.

    Args:
        sdf: A callable function that takes a tensor of size (N, 3) containing
            3D points and returns a tensor of size (N,) with the SDF values.
        points: A torch tensor containing 3D points.

    Returns:
        A torch tensor with the SDF values evaluated at the given points.
    """
    z: List[Tensor] = []
    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts))
    return torch.cat(z, dim=0)


def evaluate_multiresolution_sdf(
    evaluate: Callable,
    points_pyramid: List[Float[Tensor, "3 height width depth"]],
    coarse_mask: Union[Bool[Tensor, "1 1 height width depth"], None],
    x_max: float,
    x_min: float,
    crop_n: int,
) -> Float[Tensor, "batch"]:
    """
    Evaluate SDF values using a multi-resolution approach with a given point pyramid.

    Args:
        evaluate: A callable function to evaluate SDF values at given points.
        points_pyramid: A list of torch tensors representing points at different resolutions.
        coarse_mask: A binary mask tensor indicating regions of the bounding box where the SDF
            is expected to have a zero-crossing.
        x_max: The maximum x-coordinate of the bounding box.
        x_min: The minimum x-coordinate of the bounding box.
        crop_n: The resolution of the grid used to sample the SDF.

    Returns:
        A torch tensor with the SDF values evaluated at the given points.
    """
    mask = None
    pts_sdf: Optional[Tensor] = None
    threshold = 2 * (x_max - x_min) / crop_n * 8
    for pid, pts in enumerate(points_pyramid):
        coarse_n = pts.shape[-1]
        pts = pts.reshape(3, -1).permute(1, 0).contiguous()

        if mask is None:
            # Only evaluate SDF
            if coarse_mask is not None:
                pts_sdf = torch.ones_like(pts[:, 1])
                valid_mask = torch.nn.functional.grid_sample(coarse_mask, pts[None, None, None])[0, 0, 0, 0] > 0
                if valid_mask.any():
                    pts_sdf[valid_mask] = evaluate(pts[valid_mask].contiguous())
            else:
                pts_sdf = evaluate(pts)
        else:
            mask = mask.reshape(-1)
            pts_to_eval = pts[mask]

            if pts_to_eval.shape[0] > 0:
                pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                assert pts_sdf is not None
                pts_sdf[mask] = pts_sdf_eval

        if pid < 3:
            # Update mask
            assert pts_sdf is not None
            mask = torch.abs(pts_sdf) < threshold
            mask = mask.reshape(coarse_n, coarse_n, coarse_n)[None, None]
            mask = upsample(mask.float()).bool()

            pts_sdf = pts_sdf.reshape(coarse_n, coarse_n, coarse_n)[None, None]
            pts_sdf = upsample(pts_sdf)
            assert pts_sdf is not None
            pts_sdf = pts_sdf.reshape(-1)

        threshold /= 2.0

    assert pts_sdf is not None
    return pts_sdf


@torch.no_grad()
def generate_mesh_with_multires_marching_cubes(
    geometry_callable_field: Callable,
    resolution: int = 512,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    isosurface_threshold: float = 0.0,
    coarse_mask: Union[None, Bool[Tensor, "height width depth"]] = None,
) -> trimesh.Trimesh:
    """
    Computes the isosurface of a signed distance function (SDF) defined by the
    callable `sdf` in a given bounding box with a specified resolution. The SDF
    is sampled at a set of points within a regular grid, and the marching cubes
    algorithm is used to generate a mesh that approximates the isosurface at a
    specified isovalue `level`.

    Args:
        sdf: A callable function that takes as input a tensor of size
            (N, 3) containing 3D points, and returns a tensor of size (N,) containing
            the signed distance function evaluated at those points.
        output_path: The output directory where the resulting mesh will be saved.
        resolution: The resolution of the grid used to sample the SDF.
        bounding_box_min: The minimum coordinates of the bounding box in which the SDF
            will be evaluated.
        bounding_box_max: The maximum coordinates of the bounding box in which the SDF
            will be evaluated.
        isosurface_threshold: The isovalue at which to approximate the isosurface.
        coarse_mask: A binary mask tensor of size ("height", "width", "depth") that indicates the regions
            of the bounding box where the SDF is expected to have a zero-crossing. If
            provided, the algorithm first evaluates the SDF at the coarse voxels where
            the mask is True, and then refines the evaluation within these voxels using
            a multi-scale approach. If None, evaluates the SDF at all points in the
            bounding box.
    Returns:
        A torch tensor with the SDF values evaluated at the given points.
    """
    # Check if resolution is divisible by 512
    assert (
        resolution % 512 == 0
    ), f"""resolution must be divisible by 512, got {resolution}.
       This is important because the algorithm uses a multi-resolution approach
       to evaluate the SDF where the mimimum resolution is 512."""
    # Prepare coarse mask if provided
    if coarse_mask is not None:
        coarse_mask = coarse_mask.permute(2, 1, 0)[None, None].cuda().float()

    # Initialize variables
    crop_n = 512
    N = resolution // crop_n
    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # Initialize meshes list
    meshes = []

    # Iterate over the grid
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Calculate grid cell boundaries
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                # Create point grid
                x = np.linspace(x_min, x_max, crop_n)
                y = np.linspace(y_min, y_max, crop_n)
                z = np.linspace(z_min, z_max, crop_n)
                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                # Function to evaluate SDF for a batch of points
                def evaluate(points: torch.Tensor) -> torch.Tensor:
                    return evaluate_sdf(geometry_callable_field, points)

                # Construct point pyramids
                points = points.reshape(crop_n, crop_n, crop_n, 3).permute(3, 0, 1, 2)
                if coarse_mask is not None:
                    points_tmp = points.permute(1, 2, 3, 0)[None].cuda()
                    current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
                    current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                else:
                    current_mask = None

                # Create point pyramid for multi-resolution evaluation
                points_pyramid = create_point_pyramid(points)

                # Evaluate SDF using multi-resolution approach and mask
                pts_sdf = evaluate_multiresolution_sdf(evaluate, points_pyramid, coarse_mask, x_max, x_min, crop_n)

                z = pts_sdf.detach().cpu().numpy()

                # Skip if no surface found
                if current_mask is not None:
                    valid_z = z.reshape(crop_n, crop_n, crop_n)[current_mask]
                    if valid_z.shape[0] <= 0 or (
                        np.min(valid_z) > isosurface_threshold or np.max(valid_z) < isosurface_threshold
                    ):
                        continue

                if not (np.min(z) > isosurface_threshold or np.max(z) < isosurface_threshold):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(  # type: ignore
                        volume=z.reshape(crop_n, crop_n, crop_n),
                        level=isosurface_threshold,
                        spacing=(
                            (x_max - x_min) / (crop_n - 1),
                            (y_max - y_min) / (crop_n - 1),
                            (z_max - z_min) / (crop_n - 1),
                        ),
                        mask=current_mask,
                    )
                    verts = verts + np.array([x_min, y_min, z_min])

                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)

    combined_mesh: trimesh.Trimesh = trimesh.util.concatenate(meshes)  # type: ignore
    return combined_mesh
