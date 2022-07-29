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
from abc import abstractmethod
from typing import Dict, Optional, Type

import cv2
import torch
from torch.nn.functional import normalize
from torchtyping import TensorType

from nerfactory.cameras.rays import RayBundle


class Camera:
    """Base Camera. Intended to be subclassed.

    Args:
        camera_to_world: Camera to world transformation
        camera_index: Camera index
    """

    def __init__(
        self, camera_to_world: TensorType[3, 4] = torch.eye(4)[:3], camera_index: Optional[int] = None
    ) -> None:
        self.camera_to_world = camera_to_world
        self.camera_index = camera_index

    @property
    def device(self):
        """Returns the device that the camera is on."""
        return self.camera_to_world.device

    @abstractmethod
    def get_num_intrinsics_params(self) -> int:
        """
        Returns:
            number of optimizable intrinsic parameters
        """
        return

    @abstractmethod
    def get_intrinsics(self) -> torch.Tensor:
        """
        Returns:
            Intrinsics matrix
        """
        return

    def get_camera_to_world(self) -> TensorType[3, 4]:
        """
        Returns:
            Camera to world transformation
        """
        return self.camera_to_world

    def get_camera_to_world_h(self) -> TensorType[4, 4]:
        """
        Returns:
            Camera to world transformation with homogeneous coordinates
        """
        c2w = self.camera_to_world
        ones = torch.tensor([0, 0, 0, 1], device=c2w.device)[None]
        c2w = torch.cat([c2w, ones], dim=0)
        return c2w

    @abstractmethod
    def get_image_height(self) -> int:
        """
        Returns:
            Image height
        """
        return

    @abstractmethod
    def get_image_width(self) -> int:
        """
        Returns:
            Image width
        """
        return

    @abstractmethod
    def rescale_output_resolution(self, scaling_factor: float) -> None:
        """Rescales the camera intrinsics for output resolution.

        Args:
            scaling_factor: Scaling factor

        Returns:
            None
        """
        return

    def get_image_coords(self, pixel_offset: float = 0.5) -> TensorType["image_height", "image_width", 2]:
        """
        Args:
            pixel_offset (float): Offset for each pixel. Defaults to center of pixel (0.5)

        Returns:
            TensorType["image_height", "image_width", 2]: Grid of image coordinates.
        """
        image_height = self.get_image_height()
        image_width = self.get_image_width()
        image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
        image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        return image_coords

    @classmethod
    @abstractmethod
    def generate_rays(
        cls,
        intrinsics: TensorType[..., "num_intrinsics_params"],
        camera_to_world: TensorType[..., 3, 4],
        coords: TensorType[..., 2],
    ) -> RayBundle:
        """
        Args:
            intrinsics (TensorType[..., "num_intrinsics_params"]): Camera intrinsics
            camera_to_world (TensorType[..., 3, 4]): Camera to world transformation matrix
            coords (TensorType[..., 2]): Image grid coordinates

        Returns:
            RayBundle: A bundle of rays for each grid coordinate.
        """
        return

    def get_camera_ray_bundle(self, device=None) -> RayBundle:
        """Generate rays for the camera.

        Returns:
            Rays: Camera rays of shape [height, width]
        """
        if device is None:
            device = self.camera_to_world.device
        height = self.get_image_height()
        width = self.get_image_width()
        intrinsics = (
            self.get_intrinsics().unsqueeze(0).repeat(height, width, 1).to(device)
        )  # (num_rays, num_intrinsics_params)
        camera_to_world = self.camera_to_world.unsqueeze(0).repeat(height, width, 1, 1).to(device)  # (num_rays, 3, 4)
        coords = self.get_image_coords().to(device)
        ray_bundle = self.generate_rays(intrinsics, camera_to_world, coords)
        if self.camera_index is not None:
            ray_bundle.set_camera_indices(camera_index=self.camera_index)
        return ray_bundle

    @abstractmethod
    def to_json(self, image: Optional[TensorType["image_height", "image_width", 2]] = None) -> Dict:
        """Converts the camera to a json dictionary.

        Args:
            image: An image that is encoded to a base64 string. Defaults to None.

        Returns:
            A JSON representation of the camera
        """
        raise NotImplementedError

    @staticmethod
    def from_json(json_: Dict) -> "Camera":  # pylint:disable=no-self-use
        """Converts a json dictionary to the camera object.

        Args:
            json_: The json dictionary.

        Returns:
            The camera object instantiated from the json dictionary.
        """
        raise NotImplementedError


class PinholeCamera(Camera):
    """Pinhole camera model.

    Args:
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point in x direction
        cy: Principal point in y direction
        camera_to_world: Camera to world transformation
        camera_index: Camera index. Defaults to None.
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        camera_to_world: TensorType[3, 4] = torch.eye(4)[:3],
        camera_index: Optional[int] = None,
    ):
        super().__init__(camera_to_world=camera_to_world, camera_index=camera_index)
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy

    def get_num_intrinsics_params(self) -> int:
        return 4

    def get_image_width(self) -> int:
        return int(self.cx * 2.0)

    def get_image_height(self) -> int:
        return int(self.cy * 2.0)

    def get_intrinsics_matrix(self) -> TensorType[3, 3]:
        """
        Returns:
            TensorType[3, 3]: Pinhole camera intrinsics matrix
        """
        K = torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0, 0, 1.0]],
            dtype=torch.float32,
        )
        return K

    def get_intrinsics(self) -> torch.Tensor:
        return torch.tensor([self.cx, self.cy, self.fx, self.fy])

    @classmethod
    def fx_index(cls) -> int:
        """Returns the index of the fx parameter in the intrinsics vector
        TODO(ethan): redo this in a better way.
        Ideally we can dynamically grab the focal length parameters depending on
        Simple vs. not Simple Pinhole Model.

        Returns:
            x focal length index
        """
        return 2

    @classmethod
    def fy_index(cls) -> int:
        """Returns the index of the fy parameter in the intrinsics vector

        Returns:
            y focal length index
        """
        return 3

    def rescale_output_resolution(self, scaling_factor: float) -> None:
        self.cx *= scaling_factor
        self.cy *= scaling_factor
        self.fx *= scaling_factor
        self.fy *= scaling_factor

    @classmethod
    def generate_rays(
        cls,
        intrinsics: TensorType[..., "num_intrinsics_params"],
        camera_to_world: TensorType[..., 3, 4],
        coords: TensorType[..., 2],
    ) -> RayBundle:

        cx = intrinsics[..., 0:1]
        cy = intrinsics[..., 1:2]
        fx = intrinsics[..., cls.fx_index() : cls.fx_index() + 1]
        fy = intrinsics[..., cls.fy_index() : cls.fy_index() + 1]
        y = coords[..., 0:1]  # (..., 1)
        x = coords[..., 1:2]  # (..., 1)
        original_directions = torch.cat([(x - cx) / fx, -(y - cy) / fy, -torch.ones_like(x)], -1)  # (..., 3)
        rotation = camera_to_world[..., :3, :3]  # (..., 3, 3)
        directions = torch.sum(
            original_directions[..., None, :] * rotation, dim=-1
        )  # (..., 1, 3) * (..., 3, 3) -> (..., 3)
        directions = normalize(directions, dim=-1)
        origins = camera_to_world[..., :3, 3]  # (..., 3)

        ## Calculate pixel area directly
        dirx_min = normalize(torch.cat([(x - cx - 0.5) / fx, -(y - cy) / fy, -torch.ones_like(x)], -1), dim=-1)
        dirx_max = normalize(torch.cat([(x - cx + 0.5) / fx, -(y - cy) / fy, -torch.ones_like(x)], -1), dim=-1)
        diry_min = normalize(torch.cat([(x - cx) / fx, -(y - cy - 0.5) / fy, -torch.ones_like(x)], -1), dim=-1)
        diry_max = normalize(torch.cat([(x - cx) / fx, -(y - cy + 0.5) / fy, -torch.ones_like(x)], -1), dim=-1)
        dx = torch.sqrt(torch.sum((dirx_max - dirx_min) ** 2, dim=-1))
        dy = torch.sqrt(torch.sum((diry_max - diry_min) ** 2, dim=-1))
        pixel_area = dx * dy

        return RayBundle(origins=origins, directions=directions, pixel_area=pixel_area[..., None])

    def to_json(self, image: Optional[TensorType["image_height", "image_width", 2]] = None) -> Dict:
        json_ = {
            "type": "PinholeCamera",
            "cx": self.cx,
            "cy": self.cy,
            "fx": self.fx,
            "fy": self.fy,
            "camera_to_world": self.camera_to_world.tolist(),
            "camera_index": self.camera_index,
        }
        if image:
            # move image to cpu if not already
            # TODO: move to numpy
            data = cv2.imencode(".png", image[:, :, ::-1])[1].tobytes()
            json_["image"] = str("data:image/png;base64," + base64.b64encode(data).decode("ascii"))
        return json_

    @staticmethod
    def from_json(json_: Dict) -> "PinholeCamera":  # pylint:disable=no-self-use
        raise NotImplementedError


class SimplePinholeCamera(PinholeCamera):
    """Simple Pinhole Camera model.

    Args:
        cx: Principal point x direction
        cy: Principal point y direction
        f: Focal length
        camera_to_world: Camera to world transformation matrix
        camera_index: Camera index. Defaults to None.
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        f: float,
        camera_to_world: TensorType[3, 4] = torch.eye(4)[:3],
        camera_index: Optional[int] = None,
    ):
        super().__init__(cx, cy, f, f, camera_to_world=camera_to_world, camera_index=camera_index)

    @classmethod
    def fx_index(cls):
        return 2

    @classmethod
    def fy_index(cls):
        return 2

    def to_json(self, image: Optional[TensorType["image_height", "image_width", 2]] = None) -> Dict:
        raise NotImplementedError

    @staticmethod
    def from_json(json_: Dict) -> "SimplePinholeCamera":  # pylint:disable=no-self-use
        raise NotImplementedError


class EquirectangularCamera(Camera):
    """Equirectangular (360 degree) camera model.

    Args:
        height: Height of the image
        width: Width of the image
        camera_to_world: Camera to world transformation matrix
        camera_index: Camera index. Defaults to None.
    """

    def __init__(
        self,
        height: int,
        width: int,
        camera_to_world: TensorType[3, 4] = torch.eye(4)[:3],
        camera_index: Optional[int] = None,
    ):
        super().__init__(camera_to_world=camera_to_world, camera_index=camera_index)
        self.height = height
        self.width = width

    def get_num_intrinsics_params(self):
        return 2

    def get_image_width(self):
        return self.width

    def get_image_height(self):
        return self.height

    def get_intrinsics(self) -> torch.Tensor:
        return torch.tensor([self.height, self.width])

    def rescale_output_resolution(self, scaling_factor: float) -> None:
        self.height *= scaling_factor
        self.width *= scaling_factor

    @classmethod
    def generate_rays(
        cls,
        intrinsics: TensorType[..., 2],
        camera_to_world: TensorType[..., 3, 4],
        coords: TensorType[..., 2],
    ) -> RayBundle:

        y = coords[..., 0:1]  # (num_rays, 1)
        x = coords[..., 1:2]  # (num_rays, 1)

        height = intrinsics[..., 0:1]
        width = intrinsics[..., 1:2]

        phi = y / height * torch.pi
        theta = -x / width * 2 * torch.pi

        d_phi_min = (y + 0.5) / height * torch.pi
        d_phi_max = (y - 0.5) / height * torch.pi
        d_theta_min = -(x - 0.5) / width * 2 * torch.pi
        d_theta_max = -(x + 0.5) / width * 2 * torch.pi

        s_phi = torch.sin(phi)
        c_phi = torch.cos(phi)
        s_theta = torch.sin(theta)
        c_theta = torch.cos(theta)

        directions = torch.cat([c_theta * s_phi, c_phi, s_theta * s_phi], dim=-1)
        origins = camera_to_world[..., :3, 3]  # (num_rays, 3)

        ## Calculate area directly
        dirx_min = normalize(
            torch.cat([torch.cos(d_theta_min) * s_phi, c_phi, torch.sin(d_theta_min) * s_phi], -1), dim=-1
        )
        dirx_max = normalize(
            torch.cat([torch.cos(d_theta_max) * s_phi, c_phi, torch.sin(d_theta_max) * s_phi], -1), dim=-1
        )
        diry_min = normalize(
            torch.cat([c_theta * torch.sin(d_phi_min), torch.cos(d_phi_min), s_theta * torch.sin(d_phi_min)], -1),
            dim=-1,
        )
        diry_max = normalize(
            torch.cat([c_theta * torch.sin(d_phi_max), torch.cos(d_phi_max), s_theta * torch.sin(d_phi_max)], -1),
            dim=-1,
        )
        dx = torch.sqrt(torch.sum((dirx_max - dirx_min) ** 2, dim=-1))
        dy = torch.sqrt(torch.sum((diry_max - diry_min) ** 2, dim=-1))
        pixel_area = dx * dy
        return RayBundle(origins=origins, directions=directions, pixel_area=pixel_area[..., None])

    def to_json(self, image: Optional[TensorType["image_height", "image_width", 2]] = None) -> Dict:
        raise NotImplementedError

    @staticmethod
    def from_json(json_: Dict) -> "SimplePinholeCamera":  # pylint:disable=no-self-use
        raise NotImplementedError


def get_intrinsics_from_intrinsics_matrix(intrinsics_matrix: TensorType[3, 3]):
    """
    Return a flat intrinsics tensor.
    """
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]
    if fx == fy:
        return torch.tensor([cx, cy, fx])
    return torch.tensor([cx, cy, fx, fy])


def get_camera_model(num_intrinsics_params: int) -> Type[Camera]:
    # TODO: we should specify the type in the dataloading pipeline instead of relying on number of intrinsics
    # to make the choice
    """Returns the camera model given the specified number of intrinsics parameters.

    Args:
        num_intrinsics_params (int): Number of intrinsic parametes.

    Returns:
        Camera: Camera model
    """
    if num_intrinsics_params == 2:
        return EquirectangularCamera
    if num_intrinsics_params == 3:
        return SimplePinholeCamera
    if num_intrinsics_params == 4:
        return PinholeCamera
    raise NotImplementedError


def get_camera(intrinsics: TensorType["num_intrinsics"], camera_to_world: TensorType[3, 4], camera_index=None):
    """Returns a transformed camera.

    Args:
        intrinsics (TensorType["num_intrinsics"]): Intrinsics tensor.
        camera_to_world (TensorType[3, 4]): Camera to world transformation.
        camera_index (int): Camera index.
    """
    assert len(intrinsics.shape) == 1, "The intrinsics object should be a flat tensor."
    num_intrinsics_params = len(intrinsics)
    camera_class = get_camera_model(num_intrinsics_params)
    camera = camera_class(*intrinsics.tolist(), camera_to_world=camera_to_world, camera_index=camera_index)
    return camera
