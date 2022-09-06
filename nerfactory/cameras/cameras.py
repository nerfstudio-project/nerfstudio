# Copyright 2022 The Plenoptix Team. All rights reserved.
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
Camera Models
"""
import base64
import math
from enum import Enum, auto
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize
from torchtyping import TensorType

from nerfactory.cameras.rays import RayBundle


class CameraType(Enum):
    """Supported camera types."""

    PERSPECTIVE = auto()
    FISHEYE = auto()


class Cameras:
    """Dataset inputs for the image dataset and the ray generator.

    Note: currently only supports cameras with the same principal points and types.

    Args:
        camera_to_worlds: Tensor of per-image c2w matrices, in [R | t] format.
        fx: Focal length x.
        fy: Focal length y.
        cx: Principal point x.
        cy: Principal point y.
        distortion_params: OpenCV 6 radial distortion coefficients.
        camera_type: Type of camera model.
    """

    def __init__(
        self,
        camera_to_worlds: TensorType["num_cameras", 3, 4],
        fx: Union[TensorType["num_cameras"], float],
        fy: Union[TensorType["num_cameras"], float],
        cx: float,
        cy: float,
        width: Optional[int] = None,
        height: Optional[int] = None,
        distortion_params: Optional[TensorType["num_cameras", 6]] = None,
        camera_type: CameraType = CameraType.PERSPECTIVE,
    ):
        self._num_cameras = camera_to_worlds.shape[0]
        self.camera_to_worlds = camera_to_worlds
        if not isinstance(fx, torch.Tensor):
            fx = torch.Tensor([fx])
        if not isinstance(fy, torch.Tensor):
            fy = torch.Tensor([fy])
        self.fx = fx.broadcast_to((self._num_cameras)).to(self.device)
        self.fy = fy.broadcast_to((self._num_cameras)).to(self.device)
        self.cx = cx
        self.cy = cy
        if distortion_params is not None:
            self.distortion_params = distortion_params.broadcast_to((self._num_cameras, 6))
        else:
            self.distortion_params = None
        self._image_heights = int(self.cy * 2) if height is None else height
        self._image_widths = int(self.cx * 2) if width is None else width
        self.camera_type = camera_type

    @property
    def device(self):
        """Returns the device that the camera is on."""
        return self.camera_to_worlds.device

    @property
    def size(self) -> int:
        """Returns the number of cameras."""
        return self._num_cameras

    @property
    def image_height(self) -> int:
        """Returns the height of the images."""
        return self._image_heights

    @property
    def image_width(self) -> int:
        """Returns the height of the images."""
        return self._image_widths

    def to(self, device: Union[torch.device, str]) -> "Cameras":
        """
        Args:
            Device to move the camera to.

        Returns:
            Cameras: Cameras on the specified device.
        """
        distortion_params = self.distortion_params.to(device) if self.distortion_params is not None else None
        return Cameras(
            camera_to_worlds=self.camera_to_worlds.to(device),
            fx=self.fx.to(device),
            fy=self.fy.to(device),
            cx=self.cx,
            cy=self.cy,
            width=self.image_width,
            height=self.image_height,
            distortion_params=distortion_params,
            camera_type=self.camera_type,
        )

    def get_image_coords(self, pixel_offset: float = 0.5) -> TensorType["height", "width", 2]:
        """
        Args:
            pixel_offset (float): Offset for each pixel. Defaults to center of pixel (0.5)

        Returns:
            TensorType["image_height", "image_width", 2]: Grid of image coordinates.
        """
        image_height = self.image_height
        image_width = self.image_width
        image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
        image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        return image_coords

    def generate_rays(
        self,
        camera_indices: Union[TensorType["num_rays":...], int],
        coords: Optional[TensorType["num_rays":..., 2]] = None,
        camera_to_world_delta: Optional[TensorType["num_rays":..., 3, 4]] = None,
        distortion_params_delta: Optional[TensorType["num_rays":..., 6]] = None,
    ) -> RayBundle:
        """Generates rays for the given camera indices.

        TODO - add support for intrinsics delta.
        TODO - add support for distortions.

        Args:
            camera_indices: Indices of the cameras to generate rays for.
            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered.
            camera_to_world_delta: Optional delta for the camera to world matrices.
            distortion_params_delta: Optional delta for the distortion parameters.

        Returns:
            Rays for the given camera indices and coords.
        """

        if coords is None:
            coords = self.get_image_coords().to(self.device)

        assert coords is not None
        y = coords[..., 0]  # (..., 1)
        x = coords[..., 1]  # (..., 1)
        fx, fy = self.fx[camera_indices], self.fy[camera_indices]
        cx, cy = self.cx, self.cy

        coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)
        coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)

        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)

        distortion_params = None
        if self.distortion_params is not None:
            if distortion_params_delta is not None:
                distortion_params = self.distortion_params[camera_indices] + distortion_params_delta
        elif distortion_params_delta is not None:
            distortion_params = distortion_params_delta

        if distortion_params is not None:
            raise NotImplementedError("Camera distortion not implemented.")

        if self.camera_type == CameraType.PERSPECTIVE:
            directions_stack = torch.stack(
                [coord_stack[..., 0], coord_stack[..., 1], -torch.ones_like(coord_stack[..., 1])], dim=-1
            )
        elif self.camera_type == CameraType.FISHEYE:
            theta = torch.sqrt(torch.sum(coord_stack**2, dim=-1))
            theta = torch.clip(theta, 0.0, math.pi)

            sin_theta = torch.sin(theta)
            directions_stack = torch.stack(
                [
                    coord_stack[..., 0] * sin_theta / theta,
                    coord_stack[..., 1] * sin_theta / theta,
                    -torch.cos(theta),
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Camera type {self.camera_type} not supported.")

        c2w = self.camera_to_worlds[camera_indices]
        if camera_to_world_delta is not None:
            c2w = c2w + camera_to_world_delta
        rotation = c2w[..., :3, :3]  # (..., 3, 3)
        directions_stack = torch.sum(
            directions_stack[..., None, :] * rotation, dim=-1
        )  # (..., 1, 3) * (..., 3, 3) -> (..., 3)

        directions_stack = normalize(directions_stack, dim=-1)

        origins = c2w[..., :3, 3]  # (..., 3)
        directions = directions_stack[0]

        dx = torch.sqrt(torch.sum((directions - directions_stack[1]) ** 2, dim=-1))
        dy = torch.sqrt(torch.sum((directions - directions_stack[2]) ** 2, dim=-1))
        pixel_area = dx * dy

        return RayBundle(origins=origins, directions=directions, pixel_area=pixel_area[..., None])

    def to_json(
        self,
        camera_idx: int,
        image: Optional[TensorType["height", "width", 2]] = None,
        resize_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Convert a camera to a json dictionary.

        Args:
            camera_idx (int): Index of the camera to convert.
            image: An image in range [0, 1] that is encoded to a base64 string. Defaults to None.
            resize_shape: Shape to resize the image to. Defaults to None.

        Returns:
            A JSON representation of the camera
        """
        json_ = {
            "type": "PinholeCamera",
            "cx": self.cx,
            "cy": self.cy,
            "fx": self.fx[camera_idx].tolist(),
            "fy": self.fy[camera_idx].tolist(),
            "camera_to_world": self.camera_to_worlds[camera_idx].tolist(),
            "camera_index": camera_idx,
        }
        if image is not None:
            image_uint8 = (image * 255).detach().cpu().numpy().astype(np.uint8)
            if resize_shape:
                image_uint8 = cv2.resize(image_uint8, resize_shape)
            data = cv2.imencode(".png", image_uint8)[1].tobytes()
            json_["image"] = str("data:image/png;base64," + base64.b64encode(data).decode("ascii"))
        return json_

    def get_intrinsics_matrices(self) -> TensorType["num_cameras", 3, 3]:
        """Returns the intrinsic matrices for each camera.

        Returns:
            Pinhole camera intrinsics matrices
        """
        K = torch.zeros((self.size, 3, 3), dtype=torch.float32)
        K[:, 0, 0] = self.fx
        K[:, 1, 1] = self.fy
        K[:, 0, 2] = self.cx
        K[:, 1, 2] = self.cy
        K[:, 2, 2] = 1.0
        return K

    def rescale_output_resolution(self, scaling_factor: float) -> None:
        """Rescale the output resolution of the cameras.

        Args:
            scaling_factor: Scaling factor to apply to the output resolution.
        """
        self.fx = self.fx * scaling_factor
        self.fy = self.fy * scaling_factor
        self.cx = self.cx * scaling_factor
        self.cy = self.cy * scaling_factor
        self._image_heights = int(self._image_heights * scaling_factor)
        self._image_widths = int(self._image_widths * scaling_factor)
