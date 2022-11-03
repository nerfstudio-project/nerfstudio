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
Camera Models
"""
import base64
import math
from enum import Enum, auto
from typing import Dict, List, Optional, Union

import cv2
import torch
import torchvision
from torch.nn.functional import normalize
from torchtyping import TensorType

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.rays import RayBundle


class CameraType(Enum):
    """Supported camera types."""

    PERSPECTIVE = auto()
    FISHEYE = auto()


CAMERA_MODEL_TO_TYPE = {
    "SIMPLE_PINHOLE": CameraType.PERSPECTIVE,
    "PINHOLE": CameraType.PERSPECTIVE,
    "SIMPLE_RADIAL": CameraType.PERSPECTIVE,
    "RADIAL": CameraType.PERSPECTIVE,
    "OPENCV": CameraType.PERSPECTIVE,
    "OPENCV_FISHEYE": CameraType.FISHEYE,
}


class Cameras:
    """Dataparser outputs for the image dataset and the ray generator.

    Note: currently only supports cameras with the same principal points and types. The reason we type
    the focal lengths, principal points, and image sizes as tensors is to allow for batched cameras
    down the line in cases where your batches of camera data don't come from the same cameras.

    Args:
        camera_to_worlds: Tensor of per-image c2w matrices, in [R | t] format.
        fx: Focal length x. If a single value is provided, it is broadcasted to all cameras.
        fy: Focal length y. If a single value is provided, it is broadcasted to all cameras.
        cx: Principal point x. If a single value is provided, it is broadcasted to all cameras.
        cy: Principal point y. If a single value is provided, it is broadcasted to all cameras.
        width: Image width. If a single value is provided, it is broadcasted to all cameras.
        height: Image height. If a single value is provided, it is broadcasted to all cameras.
        distortion_params: OpenCV 6 radial distortion coefficients.
        camera_type: Type of camera model. If a single value is provided, it is broadcasted to
            all cameras. This will be an int corresponding to the CameraType enum.
    """

    def __init__(
        self,
        camera_to_worlds: TensorType["num_cameras", 3, 4],
        fx: Union[TensorType["num_cameras"], float],
        fy: Union[TensorType["num_cameras"], float],
        cx: Union[TensorType["num_cameras"], float],
        cy: Union[TensorType["num_cameras"], float],
        width: Optional[Union[TensorType["num_cameras"], int]] = None,
        height: Optional[Union[TensorType["num_cameras"], int]] = None,
        distortion_params: Optional[TensorType["num_cameras", 6]] = None,
        camera_type: Optional[
            Union[TensorType["num_cameras"], int, List[CameraType], CameraType]
        ] = CameraType.PERSPECTIVE,
        times: Optional[TensorType["num_cameras"]] = None,
    ):
        self._num_cameras = camera_to_worlds.shape[0]
        self.camera_to_worlds = camera_to_worlds  # This comes first since it determines @property self.device

        # fx fy calculation
        if not isinstance(fx, torch.Tensor):
            fx = torch.Tensor([fx])
        if not isinstance(fy, torch.Tensor):
            fy = torch.Tensor([fy])
        self.fx = fx.to(self.device).broadcast_to((self._num_cameras))
        self.fy = fy.to(self.device).broadcast_to((self._num_cameras))

        # cx cy calculation
        if not isinstance(cx, torch.Tensor):
            cx = torch.Tensor([cx])
        else:
            assert torch.all(
                cx == (cx[0] if cx.ndim > 0 else cx.item())
            ), "Batched cameras of different types will be allowed in the future."
        if not isinstance(cy, torch.Tensor):
            cy = torch.Tensor([cy])
        else:
            assert torch.all(
                cy == (cy[0] if cy.ndim > 0 else cy.item())
            ), "Batched cameras of different types will be allowed in the future."
        self.cx = cx.to(self.device).broadcast_to((self._num_cameras))
        self.cy = cy.to(self.device).broadcast_to((self._num_cameras))

        # Distortion Params Calculation:
        if distortion_params is not None:
            self.distortion_params = distortion_params.broadcast_to((self._num_cameras, 6))
        else:
            self.distortion_params = None

        self._image_heights = self._init_get_height_width(height, cy)

        self._image_widths = self._init_get_height_width(width, cx)

        self.camera_type = self._init_get_camera_type(camera_type)

        self.times = times

    def _init_get_camera_type(
        self, camera_type: Union[TensorType["num_cameras"], int, List[CameraType], CameraType]
    ) -> TensorType["num_cameras"]:
        """
        Parses the __init__() argument camera_type

        Camera Type Calculation:
        If CameraType, convert to int and then to tensor, then broadcast to all cameras
        If List of CameraTypes, convert to ints and then to tensor, then broadcast to all cameras
        If int, first go to tensor and then broadcast to all cameras
        If tensor, broadcast to all cameras

        Args:
            camera_type: camera_type argument from __init__()
        """
        if isinstance(camera_type, CameraType):
            camera_type = torch.tensor(camera_type.value, device=self.device).broadcast_to((self._num_cameras))
        elif isinstance(camera_type, List):
            camera_type = torch.tensor([c.value for c in camera_type], device=self.device).broadcast_to(
                (self._num_cameras)
            )
        elif isinstance(camera_type, int):
            camera_type = torch.tensor(camera_type, device=self.device).broadcast_to((self._num_cameras))
        elif isinstance(camera_type, torch.Tensor):
            for cam_type in camera_type:
                assert camera_type[0] == cam_type, "Batched cameras of different types will be allowed in the future."
            camera_type = camera_type.to(self.device).broadcast_to((self._num_cameras))
        else:
            raise ValueError(
                'Invalid camera_type. Must be CameraType, List[CameraType], int, or torch.Tensor["num_cameras"]. \
                    Received: '
                + str(type(camera_type))
            )
        return camera_type

    def _init_get_height_width(
        self, h_w: Union[TensorType["num_cameras"], int, None], c_x_y: TensorType["num_cameras"]
    ) -> TensorType["num_cameras"]:
        """
        Parses the __init__() argument for height or width

        Height/Width Calculation:
        If int, first go to tensor and then broadcast to all cameras
        If tensor, broadcast to all cameras
        If none, use cx or cy * 2
        Else raise error

        Args:
            h_w: height or width argument from __init__()
            c_x_y: cx or cy for when h_w == None
        """
        if isinstance(h_w, int):
            h_w = torch.Tensor([h_w]).to(torch.int64).to(self.device)
            h_w = h_w.broadcast_to((self._num_cameras))
        elif isinstance(h_w, torch.Tensor):
            h_w = h_w.to(torch.int64).to(self.device).broadcast_to((self._num_cameras))
            assert torch.all(h_w == h_w[0]), "Batched cameras of different types will be allowed in the future."
        elif h_w is None:
            h_w = torch.Tensor((c_x_y * 2).to(torch.int64).to(self.device)).broadcast_to((self._num_cameras))
        else:
            raise ValueError("Height must be an int, tensor, or None, received: " + str(type(h_w)))
        return h_w

    @property
    def device(self):
        """Returns the device that the camera is on."""
        return self.camera_to_worlds.device

    @property
    def size(self) -> int:
        """Returns the number of cameras."""
        return self._num_cameras

    @property
    def image_height(self) -> TensorType["num_cameras"]:
        """Returns the height of the images."""
        return self._image_heights

    @property
    def image_width(self) -> TensorType["num_cameras"]:
        """Returns the height of the images."""
        return self._image_widths

    def to(self, device: Union[torch.device, str]) -> "Cameras":
        """
        Args:
            device: Device to move the camera onto

        Returns:
            Cameras on the specified device.
        """
        distortion_params = self.distortion_params.to(device) if self.distortion_params is not None else None
        return Cameras(
            camera_to_worlds=self.camera_to_worlds.to(device),
            fx=self.fx.to(device),
            fy=self.fy.to(device),
            cx=self.cx.to(device),
            cy=self.cy.to(device),
            width=self.image_width.to(device),
            height=self.image_height.to(device),
            distortion_params=distortion_params.to(device) if distortion_params is not None else None,
            camera_type=self.camera_type.to(device),
            times=self.times.to(device) if self.times is not None else None,
        )

    def get_image_coords(self, pixel_offset: float = 0.5) -> TensorType["height", "width", 2]:
        """This gets the image coordinates of one of the cameras in this object

        Down the line we may support jagged images, allowing this to return multiple image coordinates of
        different sizes, but for the time being since all cameras are constrained to be the same height and
        width, this will return the same image coordinates for all cameras.

        Args:
            pixel_offset: Offset for each pixel. Defaults to center of pixel (0.5)

        Returns:
            Grid of image coordinates.
        """
        image_height = self.image_height[0]
        image_width = self.image_width[0]
        image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
        image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        return image_coords

    def generate_rays(
        self,
        camera_indices: Union[TensorType["num_rays":...], int],
        coords: Optional[TensorType["num_rays":..., 2]] = None,
        camera_opt_to_camera: Optional[TensorType["num_rays":..., 3, 4]] = None,
        distortion_params_delta: Optional[TensorType["num_rays":..., 6]] = None,
    ) -> RayBundle:
        """Generates rays for the given camera indices.

        TODO - add support for intrinsics delta.
        TODO - add support for distortions.

        Args:
            camera_indices: Indices of the cameras to generate rays for.
            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered.
            camera_opt_to_camera: Optional transform for the camera to world matrices.
            distortion_params_delta: Optional delta for the distortion parameters.

        Returns:
            Rays for the given camera indices and coords.
        """

        if isinstance(camera_indices, torch.Tensor):
            camera_indices = camera_indices.to(self.device)

        if coords is None:
            coords = self.get_image_coords().to(self.device)

        assert coords is not None
        y = coords[..., 0]  # (..., 1)
        x = coords[..., 1]  # (..., 1)
        fx, fy = self.fx[camera_indices], self.fy[camera_indices]
        cx, cy = self.cx[camera_indices], self.cy[camera_indices]

        coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)
        coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)

        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)

        distortion_params = None
        if self.distortion_params is not None:
            distortion_params = self.distortion_params[camera_indices]
            if distortion_params_delta is not None:
                distortion_params = distortion_params + distortion_params_delta
        elif distortion_params_delta is not None:
            distortion_params = distortion_params_delta

        if distortion_params is not None:
            coord_stack = camera_utils.radial_and_tangential_undistort(coord_stack, distortion_params)

        if self.camera_type[0] == CameraType.PERSPECTIVE.value:
            directions_stack = torch.stack(
                [coord_stack[..., 0], coord_stack[..., 1], -torch.ones_like(coord_stack[..., 1])], dim=-1
            )
        elif self.camera_type[0] == CameraType.FISHEYE.value:
            theta = torch.sqrt(torch.sum(coord_stack**2, dim=-1))
            theta = torch.clip(theta, 0.0, math.pi)

            sin_theta = torch.sin(theta)
            directions_stack = torch.stack(
                [coord_stack[..., 0] * sin_theta / theta, coord_stack[..., 1] * sin_theta / theta, -torch.cos(theta)],
                dim=-1,
            )
        else:
            raise ValueError(f"Camera type {CameraType(self.camera_type[0])} not supported.")

        c2w = self.camera_to_worlds[camera_indices]
        if camera_opt_to_camera is not None:
            c2w = pose_utils.multiply(c2w, camera_opt_to_camera)
        rotation = c2w[..., :3, :3]  # (..., 3, 3)
        directions_stack = torch.sum(
            directions_stack[..., None, :] * rotation, dim=-1
        )  # (..., 1, 3) * (..., 3, 3) -> (..., 3)

        directions_stack = normalize(directions_stack, dim=-1)

        origins = c2w[..., :3, 3]  # (..., 3)
        directions = directions_stack[0]

        dx = torch.sqrt(torch.sum((directions - directions_stack[1]) ** 2, dim=-1))
        dy = torch.sqrt(torch.sum((directions - directions_stack[2]) ** 2, dim=-1))
        pixel_area = (dx * dy)[..., None]

        if not isinstance(camera_indices, torch.Tensor):
            ray_bundle_camera_indices = torch.Tensor([camera_indices]).broadcast_to(pixel_area.shape).to(self.device)
        else:
            ray_bundle_camera_indices = camera_indices.view(pixel_area.shape)

        return RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=ray_bundle_camera_indices
        )

    def to_json(
        self, camera_idx: int, image: Optional[TensorType["height", "width", 2]] = None, max_size: Optional[int] = None
    ) -> Dict:
        """Convert a camera to a json dictionary.

        Args:
            camera_idx: Index of the camera to convert.
            image: An image in range [0, 1] that is encoded to a base64 string.
            max_size: Max size to resize the image to if present.

        Returns:
            A JSON representation of the camera
        """
        json_ = {
            "type": "PinholeCamera",
            "cx": self.cx[camera_idx].item(),
            "cy": self.cy[camera_idx].item(),
            "fx": self.fx[camera_idx].tolist(),
            "fy": self.fy[camera_idx].tolist(),
            "camera_to_world": self.camera_to_worlds[camera_idx].tolist(),
            "camera_index": camera_idx,
            "times": self.times[camera_idx] if self.times is not None else None,
        }
        if image is not None:
            image_uint8 = (image * 255).detach().type(torch.uint8)
            if max_size is not None:
                image_uint8 = image_uint8.permute(2, 0, 1)
                image_uint8 = torchvision.transforms.functional.resize(image_uint8, max_size)  # type: ignore
                image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            data = cv2.imencode(".jpg", image_uint8)[1].tobytes()
            json_["image"] = str("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
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

    def rescale_output_resolution(self, scaling_factor: Union[TensorType["num_cameras"], float]) -> None:
        """Rescale the output resolution of the cameras.

        Args:
            scaling_factor: Scaling factor to apply to the output resolution.
        """
        if isinstance(scaling_factor, float):
            scaling_factor = torch.tensor([scaling_factor]).to(self.device).broadcast_to((self.size))

        self.fx = self.fx * scaling_factor
        self.fy = self.fy * scaling_factor
        self.cx = self.cx * scaling_factor
        self.cy = self.cy * scaling_factor
        self._image_heights = (self._image_heights * scaling_factor).to(torch.int64)
        self._image_widths = (self._image_widths * scaling_factor).to(torch.int64)

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            return Cameras(
                self.camera_to_worlds[indices],
                self.fx[indices],
                self.fy[indices],
                self.cx[indices],
                self.cy[indices],
                height=self._image_heights[indices],
                width=self._image_widths[indices],
                distortion_params=self.distortion_params[indices] if self.distortion_params is not None else None,
                camera_type=self.camera_type[indices],
                times=self.times[indices] if self.times is not None else None,
            )
        if isinstance(indices, (int, slice)):
            indices = (indices,)
        return Cameras(
            self.camera_to_worlds[indices + (slice(None),)],
            self.fx[indices + (slice(None),)],
            self.fy[indices + (slice(None),)],
            self.cx[indices + (slice(None),)],
            self.cy[indices + (slice(None),)],
            height=self._image_heights[indices + (slice(None),)],
            width=self._image_widths[indices + (slice(None),)],
            distortion_params=self.distortion_params[indices + (slice(None),)]
            if self.distortion_params is not None
            else None,
            camera_type=self.camera_type[indices + (slice(None),)],
            times=self.times[indices + (slice(None),)] if self.times is not None else None,
        )
