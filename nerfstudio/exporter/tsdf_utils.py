# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
TSDF utils.
"""

# pylint: disable=no-member

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pymeshlab
import torch
import torch.nn.functional as F
from rich.console import Console
from skimage import measure
from torchtyping import TensorType

from nerfstudio.exporter.exporter_utils import Mesh, render_trajectory
from nerfstudio.pipelines.base_pipeline import Pipeline

CONSOLE = Console(width=120)


@dataclass
class TSDF:
    """
    Class for creating TSDFs.
    """

    voxel_coords: TensorType[3, "xdim", "ydim", "zdim"]
    """Coordinates of each voxel in the TSDF."""
    values: TensorType["xdim", "ydim", "zdim"]
    """TSDF values for each voxel."""
    weights: TensorType["xdim", "ydim", "zdim"]
    """TSDF weights for each voxel."""
    colors: TensorType["xdim", "ydim", "zdim", 3]
    """TSDF colors for each voxel."""
    voxel_size: TensorType[3]
    """Size of each voxel in the TSDF. [x, y, z] size."""
    origin: TensorType[3]
    """Origin of the TSDF [xmin, ymin, zmin]."""
    truncation_margin: float = 5.0
    """Margin for truncation."""

    def to(self, device: str):
        """Move the tensors to the specified device.

        Args:
            device: The device to move the tensors to. E.g., "cuda:0" or "cpu".
        """
        self.voxel_coords = self.voxel_coords.to(device)
        self.values = self.values.to(device)
        self.weights = self.weights.to(device)
        self.colors = self.colors.to(device)
        self.voxel_size = self.voxel_size.to(device)
        self.origin = self.origin.to(device)
        return self

    @property
    def device(self):
        """Returns the device that voxel_coords is on."""
        return self.voxel_coords.device

    @property
    def truncation(self):
        """Returns the truncation distance."""
        # TODO: clean this up
        truncation = self.voxel_size[0] * self.truncation_margin
        return truncation

    @staticmethod
    def from_aabb(aabb: TensorType[2, 3], volume_dims: TensorType[3]):
        """Returns an instance of TSDF from an axis-aligned bounding box and volume dimensions.

        Args:
            aabb: The axis-aligned bounding box with shape [[xmin, ymin, zmin], [xmax, ymax, zmax]].
            volume_dims: The volume dimensions with shape [xdim, ydim, zdim].
        """

        origin = aabb[0]
        voxel_size = (aabb[1] - aabb[0]) / volume_dims

        # create the voxel coordinates
        xdim = torch.arange(volume_dims[0])
        ydim = torch.arange(volume_dims[1])
        zdim = torch.arange(volume_dims[2])
        grid = torch.stack(torch.meshgrid([xdim, ydim, zdim], indexing="ij"), dim=0)
        voxel_coords = origin.view(3, 1, 1, 1) + grid * voxel_size.view(3, 1, 1, 1)

        # initialize the values and weights
        values = -torch.ones(volume_dims.tolist())
        weights = torch.zeros(volume_dims.tolist())
        colors = torch.zeros(volume_dims.tolist() + [3])

        # TODO: move to device

        return TSDF(voxel_coords, values, weights, colors, voxel_size, origin)

    def get_mesh(self) -> Mesh:
        """Extracts a mesh using marching cubes."""

        device = self.values.device

        # run marching cubes on CPU
        tsdf_values_np = self.values.clamp(-1, 1).cpu().numpy()
        vertices, faces, normals, _ = measure.marching_cubes(tsdf_values_np, level=0, allow_degenerate=False)

        vertices_indices = np.round(vertices).astype(int)
        colors = self.colors[vertices_indices[:, 0], vertices_indices[:, 1], vertices_indices[:, 2]]

        # move back to original device
        vertices = torch.from_numpy(vertices.copy()).to(device)
        faces = torch.from_numpy(faces.copy()).to(device)
        normals = torch.from_numpy(normals.copy()).to(device)

        # move vertices back to world space
        vertices = self.origin.view(1, 3) + vertices * self.voxel_size.view(1, 3)

        return Mesh(vertices=vertices, faces=faces, normals=normals, colors=colors)

    @classmethod
    def export_mesh(cls, mesh: Mesh, filename: str):
        """Exports the mesh to a file.
        We use pymeshlab to export the mesh as a PLY file.

        Args:
            mesh: The mesh to export.
            filename: The filename to export the mesh to.
        """
        vertex_matrix = mesh.vertices.cpu().numpy().astype("float64")
        face_matrix = mesh.faces.cpu().numpy().astype("int32")
        v_normals_matrix = mesh.normals.cpu().numpy().astype("float64")
        v_color_matrix = mesh.colors.cpu().numpy().astype("float64")
        # colors need an alpha channel
        v_color_matrix = np.concatenate([v_color_matrix, np.ones((v_color_matrix.shape[0], 1))], axis=-1)

        # create a new Mesh
        m = pymeshlab.Mesh(
            vertex_matrix=vertex_matrix,
            face_matrix=face_matrix,
            v_normals_matrix=v_normals_matrix,
            v_color_matrix=v_color_matrix,
        )
        # create a new MeshSet
        ms = pymeshlab.MeshSet()
        # add the mesh to the MeshSet
        ms.add_mesh(m, "mesh")
        # save the current mesh
        ms.save_current_mesh(filename)

    def integrate_tsdf(
        self,
        c2w: TensorType["batch", 4, 4],
        K: TensorType["batch", 3, 3],
        depth_images: TensorType["batch", 1, "height", "width"],
        color_images: Optional[TensorType["batch", 3, "height", "width"]] = None,
        mask_images: Optional[TensorType["batch", 1, "height", "width"]] = None,
    ):
        """Integrates a batch of depth images into the TSDF.

        Args:
            c2w: The camera extrinsics.
            K: The camera intrinsics.
            depth_images: The depth images to integrate.
            color_images: The color images to integrate.
            mask_images: The mask images to integrate.
        """

        if mask_images is not None:
            raise NotImplementedError("Mask images are not supported yet.")

        batch_size = c2w.shape[0]
        shape = self.voxel_coords.shape[1:]

        # Project voxel_coords into image space...

        image_size = torch.tensor(
            [depth_images.shape[-1], depth_images.shape[-2]], device=self.device
        )  # [width, height]

        # make voxel_coords homogeneous
        voxel_world_coords = self.voxel_coords.view(3, -1)
        voxel_world_coords = torch.cat(
            [voxel_world_coords, torch.ones(1, voxel_world_coords.shape[1], device=self.device)], dim=0
        )
        voxel_world_coords = voxel_world_coords.unsqueeze(0)  # [1, 4, N]
        voxel_world_coords = voxel_world_coords.expand(batch_size, *voxel_world_coords.shape[1:])  # [batch, 4, N]

        voxel_cam_coords = torch.bmm(torch.inverse(c2w), voxel_world_coords)  # [batch, 4, N]

        # flip the z axis
        voxel_cam_coords[:, 2, :] = -voxel_cam_coords[:, 2, :]
        # flip the y axis
        voxel_cam_coords[:, 1, :] = -voxel_cam_coords[:, 1, :]

        # we need the distance of the point to the camera, not the z coordinate
        voxel_depth = torch.sqrt(torch.sum(voxel_cam_coords[:, :3, :] ** 2, dim=-2, keepdim=True))  # [batch, 1, N]

        voxel_cam_coords_z = voxel_cam_coords[:, 2:3, :]
        voxel_cam_points = torch.bmm(K, voxel_cam_coords[:, 0:3, :] / voxel_cam_coords_z)  # [batch, 3, N]
        voxel_pixel_coords = voxel_cam_points[:, :2, :]  # [batch, 2, N]

        # Sample the depth images with grid sample...

        grid = voxel_pixel_coords.permute(0, 2, 1)  # [batch, N, 2]
        # normalize grid to [-1, 1]
        grid = 2.0 * grid / image_size.view(1, 1, 2) - 1.0  # [batch, N, 2]
        grid = grid[:, None]  # [batch, 1, N, 2]
        # depth
        sampled_depth = F.grid_sample(
            input=depth_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        )  # [batch, N, 1]
        sampled_depth = sampled_depth.squeeze(2)  # [batch, 1, N]
        # colors
        if color_images is not None:
            sampled_colors = F.grid_sample(
                input=color_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
            )  # [batch, N, 3]
            sampled_colors = sampled_colors.squeeze(2)  # [batch, 3, N]

        dist = sampled_depth - voxel_depth  # [batch, 1, N]
        tsdf_values = torch.clamp(dist / self.truncation, min=-1.0, max=1.0)  # [batch, 1, N]
        valid_points = (voxel_depth > 0) & (sampled_depth > 0) & (dist > -self.truncation)  # [batch, 1, N]

        # Sequentially update the TSDF...

        for i in range(batch_size):

            valid_points_i = valid_points[i]
            valid_points_i_shape = valid_points_i.view(*shape)  # [xdim, ydim, zdim]

            # the old values
            old_tsdf_values_i = self.values[valid_points_i_shape]
            old_weights_i = self.weights[valid_points_i_shape]

            # the new values
            # TODO: let the new weight be configurable
            new_tsdf_values_i = tsdf_values[i][valid_points_i]
            new_weights_i = 1.0

            total_weights = old_weights_i + new_weights_i

            self.values[valid_points_i_shape] = (
                old_tsdf_values_i * old_weights_i + new_tsdf_values_i * new_weights_i
            ) / total_weights
            self.weights[valid_points_i_shape] = torch.clamp(total_weights, max=1.0)

            if color_images is not None:
                old_colors_i = self.colors[valid_points_i_shape]  # [M, 3]
                new_colors_i = sampled_colors[i][:, valid_points_i.squeeze(0)].permute(1, 0)  # [M, 3]
                self.colors[valid_points_i_shape] = (
                    old_colors_i * old_weights_i[:, None] + new_colors_i * new_weights_i
                ) / total_weights[:, None]


def export_tsdf_mesh(
    pipeline: Pipeline,
    output_dir: Path,
    downscale_factor: int = 2,
    depth_output_name: str = "depth",
    rgb_output_name: str = "rgb",
    resolution: Union[int, List[int]] = field(default_factory=lambda: [256, 256, 256]),
    batch_size: int = 10,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Export a TSDF mesh from a pipeline.

    Args:
        pipeline: The pipeline to export the mesh from.
        output_dir: The directory to save the mesh to.
        downscale_factor: Downscale factor for the images.
        depth_output_name: Name of the depth output.
        rgb_output_name: Name of the RGB output.
        resolution: Resolution of the TSDF volume or [x, y, z] resolutions individually.
        batch_size: How many depth images to integrate per batch.
        use_bounding_box: Whether to use a bounding box for the TSDF volume.
        bounding_box_min: Minimum coordinates of the bounding box.
        bounding_box_max: Maximum coordinates of the bounding box.
    """

    device = pipeline.device

    dataparser_outputs = pipeline.datamanager.train_dataset._dataparser_outputs  # pylint: disable=protected-access

    # initialize the TSDF volume
    if not use_bounding_box:
        aabb = dataparser_outputs.scene_box.aabb
    else:
        aabb = torch.tensor([bounding_box_min, bounding_box_max])
    if isinstance(resolution, int):
        volume_dims = torch.tensor([resolution] * 3)
    elif isinstance(resolution, List):
        volume_dims = torch.tensor(resolution)
    else:
        raise ValueError("Resolution must be an int or a list.")
    tsdf = TSDF.from_aabb(aabb, volume_dims=volume_dims)
    # move TSDF to device
    tsdf.to(device)

    cameras = dataparser_outputs.cameras
    # we turn off distortion when populating the TSDF
    color_images, depth_images = render_trajectory(
        pipeline,
        cameras,
        rgb_output_name=rgb_output_name,
        depth_output_name=depth_output_name,
        rendered_resolution_scaling_factor=1.0 / downscale_factor,
        disable_distortion=True,
    )

    # camera extrinsics and intrinsics
    c2w: TensorType["N", 3, 4] = cameras.camera_to_worlds.to(device)
    # make c2w homogeneous
    c2w = torch.cat([c2w, torch.zeros(c2w.shape[0], 1, 4, device=device)], dim=1)
    c2w[:, 3, 3] = 1
    K: TensorType["N", 3, 3] = cameras.get_intrinsics_matrices().to(device)
    color_images = torch.tensor(np.array(color_images), device=device).permute(0, 3, 1, 2)  # shape (N, 3, H, W)
    depth_images = torch.tensor(np.array(depth_images), device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)

    CONSOLE.print("Integrating the TSDF")
    for i in range(0, len(c2w), batch_size):
        tsdf.integrate_tsdf(
            c2w[i : i + batch_size],
            K[i : i + batch_size],
            depth_images[i : i + batch_size],
            color_images=color_images[i : i + batch_size],
        )

    CONSOLE.print("Computing Mesh")
    mesh = tsdf.get_mesh()
    CONSOLE.print("Saving TSDF Mesh")
    tsdf.export_mesh(mesh, filename=str(output_dir / "tsdf_mesh.ply"))
