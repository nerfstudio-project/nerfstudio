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

"""Some utilities for creating TSDFs."""

from dataclasses import dataclass
from typing import Optional

import torch
import trimesh
from skimage import measure
from torchtyping import TensorType

import torch.nn.functional as F


@dataclass
class Mesh:
    vertices: TensorType["n", 3]
    """Vertices of the mesh."""
    faces: TensorType["m", 3]
    """Faces of the mesh."""
    normals: TensorType["n", 3]
    """Normals of the mesh."""


@dataclass
class TSDF:
    """Class for creating TSDFs."""

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

        # move back to original device
        vertices = torch.from_numpy(vertices.copy()).to(device)
        faces = torch.from_numpy(faces.copy()).to(device)
        normals = torch.from_numpy(normals.copy()).to(device)

        # move vertices back to world space
        vertices = self.origin.view(1, 3) + vertices * self.voxel_size.view(1, 3)

        return Mesh(
            vertices=vertices,
            faces=faces,
            normals=normals,
        )

    def export_mesh(self, mesh: Mesh, filename: str):
        """Exports the mesh to a file.
        We use trimesh to export the mesh as a PLY file.

        Args:
            mesh: The mesh to export.
            filename: The filename to export the mesh to.
        """
        assert filename.endswith(".ply"), "Only .ply files are supported."
        mesh_trimesh = trimesh.Trimesh(
            vertices=mesh.vertices.cpu().numpy(), faces=mesh.faces.cpu().numpy(), normals=mesh.normals.cpu().numpy()
        )
        mesh_trimesh.export(filename)

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

        batch_size = c2w.shape[0]

        # Project voxel_coords into image space...

        image_size = torch.tensor([depth_images.shape[2], depth_images.shape[1]], device=self.device)  # [width, height]

        # make voxel_coords homogeneous
        voxel_world_coords = self.voxel_coords.view(3, -1)
        voxel_world_coords = torch.cat(
            [voxel_world_coords, torch.ones(1, voxel_world_coords.shape[1], device=self.device)], dim=0
        )
        voxel_world_coords = voxel_world_coords.unsqueeze(0)  # [1, 4, N]
        voxel_world_coords = voxel_world_coords.expand(batch_size, *voxel_world_coords.shape[1:])  # [batch, 4, N]

        voxel_cam_coords = torch.matmul(torch.inverse(c2w), voxel_world_coords)  # [batch, 4, N]
        voxel_cam_points = torch.matmul(K, voxel_cam_coords[:, :3, :])  # [batch, 3, N]
        voxel_depth = voxel_cam_points[:, 2:3, :]  # [batch, 1, N]
        voxel_pixel_coords = voxel_cam_points[:, :2, :] / voxel_depth  # [batch, 2, N]

        # Sample the depth images with grid sample...
        grid = voxel_pixel_coords.permute(0, 2, 1)  # [batch, N, 2]
        # normalize grid to [-1, 1]
        grid = 2.0 * grid / image_size.view(1, 1, 2) - 1.0  # [batch, N, 2]
        grid = grid[:, None]  # [batch, 1, N, 2]
        sampled_depth = F.grid_sample(
            input=depth_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        )  # [batch, N, 1]
        sampled_depth = sampled_depth.squeeze(1)  # [batch, 1, N]

        # calculate the truncation
        # TODO: clean this up
        truncation = self.voxel_size[0] * 5.0

        dist = sampled_depth - voxel_depth  # [batch, 1, N]
        tsdf_values = torch.clamp(dist / truncation, min=-1.0, max=1.0)  # [batch, 1, N]
        valid_points = (voxel_depth > 0) & (sampled_depth > 0) & (dist > truncation)  # [batch, 1, N]

        # Sequentially update the TSDF...
        for i in range(batch_size):
            # TODO: finish this
            print(i)
