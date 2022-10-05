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
from dataclasses import dataclass
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
from nerfstudio.utils.tensor_dataclass import TensorDataclass


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


@dataclass(init=False)
class Cameras(TensorDataclass):
    """Dataparser outputs for the image dataset and the ray generator.

    Note: currently only supports cameras with the same principal points and types. The reason we type
    the focal lengths, principal points, and image sizes as tensors is to allow for batched cameras
    down the line in cases where your batches of camera data don't come from the same cameras.

    Args:
        camera_to_worlds: Camera to world matrices. Tensor of per-image c2w matrices, in [R | t] format,
            optionally flattened
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

    camera_to_worlds: TensorType["num_cameras":..., 3, 4]
    fx: TensorType["num_cameras":..., 1]
    fy: TensorType["num_cameras":..., 1]
    cx: TensorType["num_cameras":..., 1]
    cy: TensorType["num_cameras":..., 1]
    width: TensorType["num_cameras":..., 1]
    height: TensorType["num_cameras":..., 1]
    distortion_params: Union[TensorType["num_cameras":..., 6], None]
    camera_type: TensorType["num_cameras":..., 1]

    def __init__(
        self,
        camera_to_worlds: TensorType["batch_c2ws":..., 3, 4],
        fx: Union[TensorType["batch_fxs":..., 1], TensorType["batch_fxs":...], float],
        fy: Union[TensorType["batch_fys":..., 1], TensorType["batch_fys":...], float],
        cx: Union[TensorType["batch_cxs":..., 1], TensorType["batch_cxs":...], float],
        cy: Union[TensorType["batch_cys":..., 1], TensorType["batch_cys":...], float],
        width: Optional[Union[TensorType["batch_ws":..., 1], TensorType["batch_ws":...], int]] = None,
        height: Optional[Union[TensorType["batch_hs":..., 1], TensorType["batch_hs":...], int]] = None,
        distortion_params: Optional[TensorType["batch_dist_params":..., 6]] = None,
        camera_type: Optional[
            Union[
                TensorType["batch_cam_types":..., 1],
                TensorType["batch_cam_types":...],
                int,
                List[CameraType],
                CameraType,
            ]
        ] = CameraType.PERSPECTIVE,
    ):
        """Initializes the Cameras object.

        Note on Input Tensor Dimensions: All of these tensors have items of dimensions TensorType[3, 4]
        (in the case of the c2w matrices), TensorType[6] (in the case of distortion params), or
        TensorType[1] (in the case of the rest of the elements). The dimensions before that are
        considered the batch dimension of that tensor (batch_c2ws, batch_fxs, etc.). We will broadcast
        all the tensors to be the same batch dimension. This means you can use any combination of the
        input types in the function signature and it won't break. Your batch size for all tensors
        must be broadcastable to the same size, and the resulting number of batch dimensions will be
        the batch dimension with the largest number of dimensions.


        Args:
            camera_to_worlds: Camera to world matrices. Tensor of per-image c2w matrices, in [R | t] format,
                optionally flattened
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

        # This will notify the tensordataclass that we have a field with more than 1 dimension
        self._field_custom_dimensions = {"camera_to_worlds": 2}

        self.camera_to_worlds = camera_to_worlds

        # fx fy calculation
        if isinstance(fx, float):
            fx = torch.Tensor([fx], device=self.device)
        elif isinstance(fx, torch.Tensor):
            if fx.ndim == 0 or fx.shape[-1] != 1:
                fx = fx.unsqueeze(-1)
            fx = fx.to(self.device)
        else:
            raise ValueError(f"fx must be a float or tensor, got {type(fx)}")
        self.fx = fx  # @dataclass's post_init will take care of broadcasting

        if isinstance(fy, float):
            fy = torch.Tensor([fy], device=self.device)
        elif isinstance(fy, torch.Tensor):
            if fy.ndim == 0 or fy.shape[-1] != 1:
                fy = fy.unsqueeze(-1)
            fy = fy.to(self.device)
        else:
            raise ValueError(f"fy must be a float or tensor, got {type(fy)}")
        self.fy = fy  # @dataclass's post_init will take care of broadcasting

        # cx cy calculation
        if isinstance(cx, float):
            cx = torch.Tensor([cx], device=self.device)
        elif isinstance(cx, torch.Tensor):
            if cx.ndim == 0 or cx.shape[-1] != 1:
                cx = cx.unsqueeze(-1)
            assert torch.all(cx == cx.ravel()[0]), "Batched cameras of different cx will be allowed in the future."
            cx = cx.to(self.device)
        else:
            raise ValueError(f"cx must be a float or tensor, got {type(cx)}")
        self.cx = cx  # @dataclass's post_init will take care of broadcasting

        if isinstance(cy, float):
            cy = torch.Tensor([cy], device=self.device)
        elif isinstance(cy, torch.Tensor):
            if cy.ndim == 0 or cy.shape[-1] != 1:
                cy = cy.unsqueeze(-1)
            assert torch.all(cy == cy.ravel()[0]), "Batched cameras of different cy will be allowed in the future."
            cy = cy.to(self.device)
        else:
            raise ValueError(f"cy must be a float or tensor, got {type(cy)}")
        self.cy = cy  # @dataclass's post_init will take care of broadcasting

        # Distortion Params Calculation:
        self.distortion_params = distortion_params  # @dataclass's post_init will take care of broadcasting

        # @dataclass's post_init will take care of broadcasting
        self.height = self._init_get_height_width(height, cy)
        self.width = self._init_get_height_width(width, cx)
        self.camera_type = self._init_get_camera_type(camera_type)

        self.__post_init__()

    def _init_get_camera_type(
        self, camera_type: Union[TensorType["num_cameras":...], int, List[CameraType], CameraType]
    ) -> TensorType["num_cameras":...]:
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
            camera_type = torch.tensor([camera_type.value], device=self.device)
        elif isinstance(camera_type, List) and isinstance(camera_type[0], CameraType):
            camera_type = torch.tensor([[c.value] for c in camera_type], device=self.device)
        elif isinstance(camera_type, int):
            camera_type = torch.tensor([camera_type], device=self.device)
        elif isinstance(camera_type, torch.Tensor):
            assert not torch.is_floating_point(
                camera_type
            ), f"camera_type tensor must be of type int, not: {camera_type.dtype}"
            camera_type = camera_type.to(self.device)
            if camera_type.ndim == 0 or camera_type.shape[-1] != 1:
                camera_type = camera_type.unsqueeze(-1)
            assert torch.all(
                camera_type.ravel()[0] == camera_type
            ), "Batched cameras of different camera_types will be allowed in the future."
        else:
            raise ValueError(
                'Invalid camera_type. Must be CameraType, List[CameraType], int, or torch.Tensor["num_cameras"]. \
                    Received: '
                + str(type(camera_type))
            )
        return camera_type

    def _init_get_height_width(
        self,
        h_w: Union[TensorType["batch_hs":..., 1], TensorType["batch_hs":...], int, None],
        c_x_y: TensorType["num_cameras":...],
    ) -> TensorType["num_cameras":...]:
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
        elif isinstance(h_w, torch.Tensor):
            assert not torch.is_floating_point(h_w), f"height and width tensor must be of type int, not: {h_w.dtype}"
            h_w = h_w.to(torch.int64).to(self.device)
            if h_w.ndim == 0 or h_w.shape[-1] != 1:
                h_w = h_w.unsqueeze(-1)
            assert torch.all(h_w == h_w.ravel()[0]), "Batched cameras of different h, w will be allowed in the future."
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
    def image_height(self) -> TensorType["num_cameras"]:
        """Returns the height of the images."""
        return self.height

    @property
    def image_width(self) -> TensorType["num_cameras"]:
        """Returns the height of the images."""
        return self.width

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
        image_height = self.image_height.ravel()[0]
        image_width = self.image_width.ravel()[0]
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
            camera_indices: Camera indices of the flattened cameras object to generate rays for.
            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered.
            camera_opt_to_camera: Optional transform for the camera to world matrices.
            distortion_params_delta: Optional delta for the distortion parameters.

        Returns:
            Rays for the given camera indices and coords.
        """

        cameras = self.flatten()

        # print("INDICES", camera_indices)

        if isinstance(camera_indices, torch.Tensor):
            camera_indices = camera_indices.to(cameras.device)

        if coords is None:
            coords = cameras.get_image_coords().to(cameras.device)

        # print("COORDS", coords.shape)

        assert coords is not None
        y = coords[..., 0]  # (...,)
        x = coords[..., 1]  # (...,)
        fx, fy = cameras.fx[camera_indices].squeeze(-1), cameras.fy[camera_indices].squeeze(-1)
        cx, cy = cameras.cx[camera_indices].squeeze(-1), cameras.cy[camera_indices].squeeze(-1)

        # print("RAY GENERATOR SHAPES")
        # print(self.shape, cameras.shape)
        # print(x.shape, cx.shape, fx.shape, y.shape, cy.shape, fy.shape)

        coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)
        coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)

        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)

        distortion_params = None
        if cameras.distortion_params is not None:
            distortion_params = cameras.distortion_params[camera_indices]
            if distortion_params_delta is not None:
                distortion_params = distortion_params + distortion_params_delta
        elif distortion_params_delta is not None:
            distortion_params = distortion_params_delta

        if distortion_params is not None:
            coord_stack = camera_utils.radial_and_tangential_undistort(coord_stack, distortion_params)

        if cameras.camera_type.ravel()[0] == CameraType.PERSPECTIVE.value:
            directions_stack = torch.stack(
                [coord_stack[..., 0], coord_stack[..., 1], -torch.ones_like(coord_stack[..., 1])], dim=-1
            )
        elif cameras.camera_type.ravel()[0] == CameraType.FISHEYE.value:
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
            raise ValueError(f"Camera type {CameraType(cameras.camera_type.ravel()[0])} not supported.")

        c2w = cameras.camera_to_worlds[camera_indices]
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
            ray_bundle_camera_indices = torch.Tensor([camera_indices]).broadcast_to(pixel_area.shape).to(cameras.device)
        else:
            ray_bundle_camera_indices = camera_indices.view(pixel_area.shape)

        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=ray_bundle_camera_indices,
        )

    def to_json(
        self,
        camera_idx: int,
        image: Optional[TensorType["height", "width", 2]] = None,
        max_size: Optional[int] = None,
    ) -> Dict:
        """Convert a camera to a json dictionary.

        Args:
            camera_idx: Index of the camera to convert.
            image: An image in range [0, 1] that is encoded to a base64 string.
            max_size: Max size to resize the image to if present.

        Returns:
            A JSON representation of the camera
        """
        flattened = self.flatten()
        json_ = {
            "type": "PinholeCamera",
            "cx": flattened[camera_idx].cx.item(),
            "cy": flattened[camera_idx].cy.item(),
            "fx": flattened[camera_idx].fx.tolist(),
            "fy": flattened[camera_idx].fy.tolist(),
            "camera_to_world": self.camera_to_worlds.ravel()[camera_idx].tolist(),
            "camera_index": camera_idx,
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
        K = torch.zeros((*self.shape, 3, 3), dtype=torch.float32)
        K[..., 0, 0] = self.fx.squeeze(-1)
        K[..., 1, 1] = self.fy.squeeze(-1)
        K[..., 0, 2] = self.cx.squeeze(-1)
        K[..., 1, 2] = self.cy.squeeze(-1)
        K[..., 2, 2] = 1.0
        return K

    def rescale_output_resolution(
        self, scaling_factor: Union[TensorType["num_cameras":...], TensorType["num_cameras":..., 1], float, int]
    ) -> None:
        """Rescale the output resolution of the cameras.

        Args:
            scaling_factor: Scaling factor to apply to the output resolution.
        """
        if isinstance(scaling_factor, (float, int)):
            scaling_factor = torch.tensor([scaling_factor]).to(self.device).broadcast_to((self.cx.shape))
        elif isinstance(scaling_factor, torch.Tensor) and scaling_factor.shape == self.shape:
            scaling_factor = scaling_factor.unsqueeze(-1)
        elif isinstance(scaling_factor, torch.Tensor) and scaling_factor.shape == (*self.shape, 1):
            pass
        else:
            raise ValueError(
                f"Scaling factor must be a float, int, or a tensor of shape {self.shape} or {(*self.shape, 1)}."
            )

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
            distortion_params=self.distortion_params[indices + (slice(None),)] if self.distortion_params else None,
            camera_type=self.camera_type[indices + (slice(None),)],
        )
