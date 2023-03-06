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
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import cv2
import torch
import torchvision
from torch.nn import Parameter
from torchtyping import TensorType

import nerfstudio.utils.math
import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.misc import strtobool
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name


class CameraType(Enum):
    """Supported camera types."""

    PERSPECTIVE = auto()
    FISHEYE = auto()
    EQUIRECTANGULAR = auto()


CAMERA_MODEL_TO_TYPE = {
    "SIMPLE_PINHOLE": CameraType.PERSPECTIVE,
    "PINHOLE": CameraType.PERSPECTIVE,
    "SIMPLE_RADIAL": CameraType.PERSPECTIVE,
    "RADIAL": CameraType.PERSPECTIVE,
    "OPENCV": CameraType.PERSPECTIVE,
    "OPENCV_FISHEYE": CameraType.FISHEYE,
    "EQUIRECTANGULAR": CameraType.EQUIRECTANGULAR,
}


@dataclass(init=False)
class Cameras(TensorDataclass):
    """Dataparser outputs for the image dataset and the ray generator.

    Note: currently only supports cameras with the same principal points and types. The reason we type
    the focal lengths, principal points, and image sizes as tensors is to allow for batched cameras
    down the line in cases where your batches of camera data don't come from the same cameras.

     If a single value is provided, it is broadcasted to all cameras.

    Args:
        camera_to_worlds: Camera to world matrices. Tensor of per-image c2w matrices, in [R | t] format
        fx: Focal length x
        fy: Focal length y
        cx: Principal point x
        cy: Principal point y
        width: Image width
        height: Image height
        distortion_params: OpenCV 6 radial distortion coefficients
        camera_type: Type of camera model. This will be an int corresponding to the CameraType enum.
        times: Timestamps for each camera
    """

    camera_to_worlds: TensorType["num_cameras":..., 3, 4]
    fx: TensorType["num_cameras":..., 1]
    fy: TensorType["num_cameras":..., 1]
    cx: TensorType["num_cameras":..., 1]
    cy: TensorType["num_cameras":..., 1]
    width: TensorType["num_cameras":..., 1]
    height: TensorType["num_cameras":..., 1]
    distortion_params: Optional[TensorType["num_cameras":..., 6]]
    camera_type: TensorType["num_cameras":..., 1]
    times: Optional[TensorType["num_cameras", 1]]

    def __init__(
        self,
        camera_to_worlds: TensorType["batch_c2ws":..., 3, 4],
        fx: Union[TensorType["batch_fxs":..., 1], float],
        fy: Union[TensorType["batch_fys":..., 1], float],
        cx: Union[TensorType["batch_cxs":..., 1], float],
        cy: Union[TensorType["batch_cys":..., 1], float],
        width: Optional[Union[TensorType["batch_ws":..., 1], int]] = None,
        height: Optional[Union[TensorType["batch_hs":..., 1], int]] = None,
        distortion_params: Optional[TensorType["batch_dist_params":..., 6]] = None,
        camera_type: Optional[
            Union[
                TensorType["batch_cam_types":..., 1],
                int,
                List[CameraType],
                CameraType,
            ]
        ] = CameraType.PERSPECTIVE,
        times: Optional[TensorType["num_cameras"]] = None,
    ) -> None:
        """Initializes the Cameras object.

        Note on Input Tensor Dimensions: All of these tensors have items of dimensions TensorType[3, 4]
        (in the case of the c2w matrices), TensorType[6] (in the case of distortion params), or
        TensorType[1] (in the case of the rest of the elements). The dimensions before that are
        considered the batch dimension of that tensor (batch_c2ws, batch_fxs, etc.). We will broadcast
        all the tensors to be the same batch dimension. This means you can use any combination of the
        input types in the function signature and it won't break. Your batch size for all tensors
        must be broadcastable to the same size, and the resulting number of batch dimensions will be
        the batch dimension with the largest number of dimensions.
        """

        # This will notify the tensordataclass that we have a field with more than 1 dimension
        self._field_custom_dimensions = {"camera_to_worlds": 2}

        self.camera_to_worlds = camera_to_worlds

        # fx fy calculation
        self.fx = self._init_get_fc_xy(fx, "fx")  # @dataclass's post_init will take care of broadcasting
        self.fy = self._init_get_fc_xy(fy, "fy")  # @dataclass's post_init will take care of broadcasting

        # cx cy calculation
        self.cx = self._init_get_fc_xy(cx, "cx")  # @dataclass's post_init will take care of broadcasting
        self.cy = self._init_get_fc_xy(cy, "cy")  # @dataclass's post_init will take care of broadcasting

        # Distortion Params Calculation:
        self.distortion_params = distortion_params  # @dataclass's post_init will take care of broadcasting

        # @dataclass's post_init will take care of broadcasting
        self.height = self._init_get_height_width(height, self.cy)
        self.width = self._init_get_height_width(width, self.cx)
        self.camera_type = self._init_get_camera_type(camera_type)
        self.times = self._init_get_times(times)

        self.__post_init__()  # This will do the dataclass post_init and broadcast all the tensors

        self._use_nerfacc = strtobool(os.environ.get("INTERSECT_WITH_NERFACC", "TRUE"))

    def _init_get_fc_xy(self, fc_xy: Union[float, torch.Tensor], name: str) -> torch.Tensor:
        """
        Parses the input focal length / principle point x or y and returns a tensor of the correct shape

        Only needs to make sure that we a 1 in the last dimension if it is a tensor. If it is a float, we
        just need to make it into a tensor and it will be broadcasted later in the __post_init__ function.

        Args:
            fc_xy: The focal length / principle point x or y
            name: The name of the variable. Used for error messages
        """
        if isinstance(fc_xy, float):
            fc_xy = torch.Tensor([fc_xy], device=self.device)
        elif isinstance(fc_xy, torch.Tensor):
            if fc_xy.ndim == 0 or fc_xy.shape[-1] != 1:
                fc_xy = fc_xy.unsqueeze(-1)
            fc_xy = fc_xy.to(self.device)
        else:
            raise ValueError(f"{name} must be a float or tensor, got {type(fc_xy)}")
        return fc_xy

    def _init_get_camera_type(
        self,
        camera_type: Union[
            TensorType["batch_cam_types":..., 1], TensorType["batch_cam_types":...], int, List[CameraType], CameraType
        ],
    ) -> TensorType["num_cameras":..., 1]:
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
            # assert torch.all(
            #     camera_type.view(-1)[0] == camera_type
            # ), "Batched cameras of different camera_types will be allowed in the future."
        else:
            raise ValueError(
                'Invalid camera_type. Must be CameraType, List[CameraType], int, or torch.Tensor["num_cameras"]. \
                    Received: '
                + str(type(camera_type))
            )
        return camera_type

    def _init_get_height_width(
        self,
        h_w: Union[TensorType["batch_hws":..., 1], TensorType["batch_hws":...], int, None],
        c_x_y: TensorType["batch_cxys":...],
    ) -> TensorType["num_cameras":..., 1]:
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
            h_w = torch.as_tensor([h_w]).to(torch.int64).to(self.device)
        elif isinstance(h_w, torch.Tensor):
            assert not torch.is_floating_point(h_w), f"height and width tensor must be of type int, not: {h_w.dtype}"
            h_w = h_w.to(torch.int64).to(self.device)
            if h_w.ndim == 0 or h_w.shape[-1] != 1:
                h_w = h_w.unsqueeze(-1)
        # assert torch.all(h_w == h_w.view(-1)[0]), "Batched cameras of different h, w will be allowed in the future."
        elif h_w is None:
            h_w = torch.as_tensor((c_x_y * 2)).to(torch.int64).to(self.device)
        else:
            raise ValueError("Height must be an int, tensor, or None, received: " + str(type(h_w)))
        return h_w

    def _init_get_times(self, times: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if times is None:
            times = None
        elif isinstance(times, torch.Tensor):
            if times.ndim == 0 or times.shape[-1] != 1:
                times = times.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"times must be None or a tensor, got {type(times)}")

        return times

    @property
    def device(self) -> TORCH_DEVICE:
        """Returns the device that the camera is on."""
        return self.camera_to_worlds.device

    @property
    def image_height(self) -> TensorType["num_cameras":..., 1]:
        """Returns the height of the images."""
        return self.height

    @property
    def image_width(self) -> TensorType["num_cameras":..., 1]:
        """Returns the height of the images."""
        return self.width

    @property
    def is_jagged(self) -> bool:
        """
        Returns whether or not the cameras are "jagged" (i.e. the height and widths are different, meaning that
        you cannot concatenate the image coordinate maps together)
        """
        h_jagged = not torch.all(self.height == self.height.view(-1)[0])
        w_jagged = not torch.all(self.width == self.width.view(-1)[0])
        return h_jagged or w_jagged

    def get_image_coords(
        self, pixel_offset: float = 0.5, index: Optional[Tuple] = None
    ) -> TensorType["height", "width", 2]:
        """This gets the image coordinates of one of the cameras in this object.

        If no index is specified, it will return the maximum possible sized height / width image coordinate map,
        by looking at the maximum height and width of all the cameras in this object.

        Args:
            pixel_offset: Offset for each pixel. Defaults to center of pixel (0.5)
            index: Tuple of indices into the batch dimensions of the camera. Defaults to None, which returns the 0th
                flattened camera

        Returns:
            Grid of image coordinates.
        """
        if index is None:
            image_height = torch.max(self.image_height.view(-1))
            image_width = torch.max(self.image_width.view(-1))
            image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
            image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        else:
            image_height = self.image_height[index].item()
            image_width = self.image_width[index].item()
            image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
            image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        return image_coords

    def generate_rays(  # pylint: disable=too-many-statements
        self,
        camera_indices: Union[TensorType["num_rays":..., "num_cameras_batch_dims"], int],
        coords: Optional[TensorType["num_rays":..., 2]] = None,
        camera_opt_to_camera: Optional[TensorType["num_rays":..., 3, 4]] = None,
        distortion_params_delta: Optional[TensorType["num_rays":..., 6]] = None,
        keep_shape: Optional[bool] = None,
        disable_distortion: bool = False,
        aabb_box: Optional[SceneBox] = None,
    ) -> RayBundle:
        """Generates rays for the given camera indices.

        This function will standardize the input arguments and then call the _generate_rays_from_coords function
        to generate the rays. Our goal is to parse the arguments and then get them into the right shape:
            - camera_indices: (num_rays:..., num_cameras_batch_dims)
            - coords: (num_rays:..., 2)
            - camera_opt_to_camera: (num_rays:..., 3, 4) or None
            - distortion_params_delta: (num_rays:..., 6) or None

        Read the docstring for _generate_rays_from_coords for more information on how we generate the rays
        after we have standardized the arguments.

        We are only concerned about different combinations of camera_indices and coords matrices, and the following
        are the 4 cases we have to deal with:
            1. isinstance(camera_indices, int) and coords == None
                - In this case we broadcast our camera_indices / coords shape (h, w, 1 / 2 respectively)
            2. isinstance(camera_indices, int) and coords != None
                - In this case, we broadcast camera_indices to the same batch dim as coords
            3. not isinstance(camera_indices, int) and coords == None
                - In this case, we will need to set coords so that it is of shape (h, w, num_rays, 2), and broadcast
                    all our other args to match the new definition of num_rays := (h, w) + num_rays
            4. not isinstance(camera_indices, int) and coords != None
                - In this case, we have nothing to do, only check that the arguments are of the correct shape

        There is one more edge case we need to be careful with: when we have "jagged cameras" (ie: different heights
        and widths for each camera). This isn't problematic when we specify coords, since coords is already a tensor.
        When coords == None (ie: when we render out the whole image associated with this camera), we run into problems
        since there's no way to stack each coordinate map as all coordinate maps are all different shapes. In this case,
        we will need to flatten each individual coordinate map and concatenate them, giving us only one batch dimension,
        regardless of the number of prepended extra batch dimensions in the camera_indices tensor.


        Args:
            camera_indices: Camera indices of the flattened cameras object to generate rays for.
            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered.
            camera_opt_to_camera: Optional transform for the camera to world matrices.
            distortion_params_delta: Optional delta for the distortion parameters.
            keep_shape: If None, then we default to the regular behavior of flattening if cameras is jagged, otherwise
                keeping dimensions. If False, we flatten at the end. If True, then we keep the shape of the
                camera_indices and coords tensors (if we can).
            disable_distortion: If True, disables distortion.
            aabb_box: if not None will calculate nears and fars of the ray according to aabb box intesection

        Returns:
            Rays for the given camera indices and coords.
        """
        # Check the argument types to make sure they're valid and all shaped correctly
        assert isinstance(camera_indices, (torch.Tensor, int)), "camera_indices must be a tensor or int"
        assert coords is None or isinstance(coords, torch.Tensor), "coords must be a tensor or None"
        assert camera_opt_to_camera is None or isinstance(camera_opt_to_camera, torch.Tensor)
        assert distortion_params_delta is None or isinstance(distortion_params_delta, torch.Tensor)
        if isinstance(camera_indices, torch.Tensor) and isinstance(coords, torch.Tensor):
            num_rays_shape = camera_indices.shape[:-1]
            errormsg = "Batch dims of inputs must match when inputs are all tensors"
            assert coords.shape[:-1] == num_rays_shape, errormsg
            assert camera_opt_to_camera is None or camera_opt_to_camera.shape[:-2] == num_rays_shape, errormsg
            assert distortion_params_delta is None or distortion_params_delta.shape[:-1] == num_rays_shape, errormsg

        # If zero dimensional, we need to unsqueeze to get a batch dimension and then squeeze later
        if not self.shape:
            cameras = self.reshape((1,))
            assert torch.all(
                torch.tensor(camera_indices == 0) if isinstance(camera_indices, int) else camera_indices == 0
            ), "Can only index into single camera with no batch dimensions if index is zero"
        else:
            cameras = self

        # If the camera indices are an int, then we need to make sure that the camera batch is 1D
        if isinstance(camera_indices, int):
            assert (
                len(cameras.shape) == 1
            ), "camera_indices must be a tensor if cameras are batched with more than 1 batch dimension"
            camera_indices = torch.tensor([camera_indices], device=cameras.device)

        assert camera_indices.shape[-1] == len(
            cameras.shape
        ), "camera_indices must have shape (num_rays:..., num_cameras_batch_dims)"

        # If keep_shape is True, then we need to make sure that the camera indices in question
        # are all the same height and width and can actually be batched while maintaining the image
        # shape
        if keep_shape is True:
            assert torch.all(cameras.height[camera_indices] == cameras.height[camera_indices[0]]) and torch.all(
                cameras.width[camera_indices] == cameras.width[camera_indices[0]]
            ), "Can only keep shape if all cameras have the same height and width"

        # If the cameras don't all have same height / width, if coords is not none, we will need to generate
        # a flat list of coords for each camera and then concatenate otherwise our rays will be jagged.
        # Camera indices, camera_opt, and distortion will also need to be broadcasted accordingly which is non-trivial
        if cameras.is_jagged and coords is None and (keep_shape is None or keep_shape is False):
            index_dim = camera_indices.shape[-1]
            camera_indices = camera_indices.reshape(-1, index_dim)
            _coords = [cameras.get_image_coords(index=tuple(index)).reshape(-1, 2) for index in camera_indices]
            camera_indices = torch.cat(
                [index.unsqueeze(0).repeat(coords.shape[0], 1) for index, coords in zip(camera_indices, _coords)],
            )
            coords = torch.cat(_coords, dim=0)
            assert coords.shape[0] == camera_indices.shape[0]
            # Need to get the coords of each indexed camera and flatten all coordinate maps and concatenate them

        # The case where we aren't jagged && keep_shape (since otherwise coords is already set) and coords
        # is None. In this case we append (h, w) to the num_rays dimensions for all tensors. In this case,
        # each image in camera_indices has to have the same shape since otherwise we would have error'd when
        # we checked keep_shape is valid or we aren't jagged.
        if coords is None:
            index_dim = camera_indices.shape[-1]
            index = camera_indices.reshape(-1, index_dim)[0]
            coords: torch.Tensor = cameras.get_image_coords(index=tuple(index))  # (h, w, 2)
            coords = coords.reshape(coords.shape[:2] + (1,) * len(camera_indices.shape[:-1]) + (2,))  # (h, w, 1..., 2)
            coords = coords.expand(coords.shape[:2] + camera_indices.shape[:-1] + (2,))  # (h, w, num_rays, 2)
            camera_opt_to_camera = (  # (h, w, num_rays, 3, 4) or None
                camera_opt_to_camera.broadcast_to(coords.shape[:-1] + (3, 4))
                if camera_opt_to_camera is not None
                else None
            )
            distortion_params_delta = (  # (h, w, num_rays, 6) or None
                distortion_params_delta.broadcast_to(coords.shape[:-1] + (6,))
                if distortion_params_delta is not None
                else None
            )

        # If camera indices was an int or coords was none, we need to broadcast our indices along batch dims
        camera_indices = camera_indices.broadcast_to(coords.shape[:-1] + (len(cameras.shape),)).to(torch.long)

        # Checking our tensors have been standardized
        assert isinstance(coords, torch.Tensor) and isinstance(camera_indices, torch.Tensor)
        assert camera_indices.shape[-1] == len(cameras.shape)
        assert camera_opt_to_camera is None or camera_opt_to_camera.shape[:-2] == coords.shape[:-1]
        assert distortion_params_delta is None or distortion_params_delta.shape[:-1] == coords.shape[:-1]

        # This will do the actual work of generating the rays now that we have standardized the inputs
        # raybundle.shape == (num_rays) when done
        # pylint: disable=protected-access
        raybundle = cameras._generate_rays_from_coords(
            camera_indices, coords, camera_opt_to_camera, distortion_params_delta, disable_distortion=disable_distortion
        )

        # If we have mandated that we don't keep the shape, then we flatten
        if keep_shape is False:
            raybundle = raybundle.flatten()

        if aabb_box:
            with torch.no_grad():
                tensor_aabb = Parameter(aabb_box.aabb.flatten(), requires_grad=False)

                rays_o = raybundle.origins.contiguous()
                rays_d = raybundle.directions.contiguous()

                tensor_aabb = tensor_aabb.to(rays_o.device)
                shape = rays_o.shape

                rays_o = rays_o.reshape((-1, 3))
                rays_d = rays_d.reshape((-1, 3))

                t_min, t_max = nerfstudio.utils.math.intersect_aabb(rays_o, rays_d, tensor_aabb)

                t_min = t_min.reshape([shape[0], shape[1], 1])
                t_max = t_max.reshape([shape[0], shape[1], 1])

                raybundle.nears = t_min
                raybundle.fars = t_max

        # TODO: We should have to squeeze the last dimension here if we started with zero batch dims, but never have to,
        # so there might be a rogue squeeze happening somewhere, and this may cause some unintended behaviour
        # that we haven't caught yet with tests
        return raybundle

    # pylint: disable=too-many-statements
    def _generate_rays_from_coords(
        self,
        camera_indices: TensorType["num_rays":..., "num_cameras_batch_dims"],
        coords: TensorType["num_rays":..., 2],
        camera_opt_to_camera: Optional[TensorType["num_rays":..., 3, 4]] = None,
        distortion_params_delta: Optional[TensorType["num_rays":..., 6]] = None,
        disable_distortion: bool = False,
    ) -> RayBundle:
        """Generates rays for the given camera indices and coords where self isn't jagged

        This is a fairly complex function, so let's break this down slowly.

        Shapes involved:
            - num_rays: This is your output raybundle shape. It dictates the number and shape of the rays generated
            - num_cameras_batch_dims: This is the number of dimensions of our camera

        Args:
            camera_indices: Camera indices of the flattened cameras object to generate rays for.
                The shape of this is such that indexing into camera_indices["num_rays":...] will return the
                index into each batch dimension of the camera in order to get the correct camera specified by
                "num_rays".

                Example:
                    >>> cameras = Cameras(...)
                    >>> cameras.shape
                        (2, 3, 4)

                    >>> camera_indices = torch.tensor([0, 0, 0]) # We need an axis of length 3 since cameras.ndim == 3
                    >>> camera_indices.shape
                        (3,)
                    >>> coords = torch.tensor([1,1])
                    >>> coords.shape
                        (2,)
                    >>> out_rays = cameras.generate_rays(camera_indices=camera_indices, coords = coords)
                        # This will generate a RayBundle with a single ray for the
                        # camera at cameras[0,0,0] at image coordinates (1,1), so out_rays.shape == ()
                    >>> out_rays.shape
                        ()

                    >>> camera_indices = torch.tensor([[0,0,0]])
                    >>> camera_indices.shape
                        (1, 3)
                    >>> coords = torch.tensor([[1,1]])
                    >>> coords.shape
                        (1, 2)
                    >>> out_rays = cameras.generate_rays(camera_indices=camera_indices, coords = coords)
                        # This will generate a RayBundle with a single ray for the
                        # camera at cameras[0,0,0] at point (1,1), so out_rays.shape == (1,)
                        # since we added an extra dimension in front of camera_indices
                    >>> out_rays.shape
                        (1,)

                If you want more examples, check tests/cameras/test_cameras and the function check_generate_rays_shape

                The bottom line is that for camera_indices: (num_rays:..., num_cameras_batch_dims), num_rays is the
                output shape and if you index into the output RayBundle with some indices [i:...], if you index into
                camera_indices with camera_indices[i:...] as well, you will get a 1D tensor containing the batch
                indices into the original cameras object corresponding to that ray (ie: you will get the camera
                from our batched cameras corresponding to the ray at RayBundle[i:...]).

            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered, meaning
                height and width get prepended to the num_rays dimensions. Indexing into coords with [i:...] will
                get you the image coordinates [x, y] of that specific ray located at output RayBundle[i:...].

            camera_opt_to_camera: Optional transform for the camera to world matrices.
                In terms of shape, it follows the same rules as coords, but indexing into it with [i:...] gets you
                the 2D camera to world transform matrix for the camera optimization at RayBundle[i:...].

            distortion_params_delta: Optional delta for the distortion parameters.
                In terms of shape, it follows the same rules as coords, but indexing into it with [i:...] gets you
                the 1D tensor with the 6 distortion parameters for the camera optimization at RayBundle[i:...].

            disable_distortion: If True, disables distortion.

        Returns:
            Rays for the given camera indices and coords. RayBundle.shape == num_rays
        """
        # Make sure we're on the right devices
        camera_indices = camera_indices.to(self.device)
        coords = coords.to(self.device)

        # Checking to make sure everything is of the right shape and type
        num_rays_shape = camera_indices.shape[:-1]
        assert camera_indices.shape == num_rays_shape + (self.ndim,)
        assert coords.shape == num_rays_shape + (2,)
        assert coords.shape[-1] == 2
        assert camera_opt_to_camera is None or camera_opt_to_camera.shape == num_rays_shape + (3, 4)
        assert distortion_params_delta is None or distortion_params_delta.shape == num_rays_shape + (6,)

        # Here, we've broken our indices down along the num_cameras_batch_dims dimension allowing us to index by all
        # of our output rays at each dimension of our cameras object
        true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]

        # Get all our focal lengths, principal points and make sure they are the right shapes
        y = coords[..., 0]  # (num_rays,) get rid of the last dimension
        x = coords[..., 1]  # (num_rays,) get rid of the last dimension
        fx, fy = self.fx[true_indices].squeeze(-1), self.fy[true_indices].squeeze(-1)  # (num_rays,)
        cx, cy = self.cx[true_indices].squeeze(-1), self.cy[true_indices].squeeze(-1)  # (num_rays,)
        assert (
            y.shape == num_rays_shape
            and x.shape == num_rays_shape
            and fx.shape == num_rays_shape
            and fy.shape == num_rays_shape
            and cx.shape == num_rays_shape
            and cy.shape == num_rays_shape
        ), (
            str(num_rays_shape)
            + str(y.shape)
            + str(x.shape)
            + str(fx.shape)
            + str(fy.shape)
            + str(cx.shape)
            + str(cy.shape)
        )

        # Get our image coordinates and image coordinates offset by 1 (offsets used for dx, dy calculations)
        # Also make sure the shapes are correct
        coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
        coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)  # (num_rays, 2)
        assert (
            coord.shape == num_rays_shape + (2,)
            and coord_x_offset.shape == num_rays_shape + (2,)
            and coord_y_offset.shape == num_rays_shape + (2,)
        )

        # Stack image coordinates and image coordinates offset by 1, check shapes too
        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)  # (3, num_rays, 2)
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Undistorts our images according to our distortion parameters
        if not disable_distortion:
            distortion_params = None
            if self.distortion_params is not None:
                distortion_params = self.distortion_params[true_indices]
                if distortion_params_delta is not None:
                    distortion_params = distortion_params + distortion_params_delta
            elif distortion_params_delta is not None:
                distortion_params = distortion_params_delta

            # Do not apply distortion for equirectangular images
            if distortion_params is not None:
                mask = (self.camera_type[true_indices] != CameraType.EQUIRECTANGULAR.value).squeeze(-1)  # (num_rays)
                coord_mask = torch.stack([mask, mask, mask], dim=0)
                if mask.any():
                    coord_stack[coord_mask, :] = camera_utils.radial_and_tangential_undistort(
                        coord_stack[coord_mask, :].reshape(3, -1, 2),
                        distortion_params[mask, :],
                    ).reshape(-1, 2)

        # Make sure after we have undistorted our images, the shapes are still correct
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Gets our directions for all our rays in camera coordinates and checks shapes at the end
        # Here, directions_stack is of shape (3, num_rays, 3)
        # directions_stack[0] is the direction for ray in camera coordinates
        # directions_stack[1] is the direction for ray in camera coordinates offset by 1 in x
        # directions_stack[2] is the direction for ray in camera coordinates offset by 1 in y
        cam_types = torch.unique(self.camera_type, sorted=False)
        directions_stack = torch.empty((3,) + num_rays_shape + (3,), device=self.device)
        if CameraType.PERSPECTIVE.value in cam_types:
            mask = (self.camera_type[true_indices] == CameraType.PERSPECTIVE.value).squeeze(-1)  # (num_rays)
            mask = torch.stack([mask, mask, mask], dim=0)
            directions_stack[..., 0][mask] = torch.masked_select(coord_stack[..., 0], mask).float()
            directions_stack[..., 1][mask] = torch.masked_select(coord_stack[..., 1], mask).float()
            directions_stack[..., 2][mask] = -1.0

        if CameraType.FISHEYE.value in cam_types:
            mask = (self.camera_type[true_indices] == CameraType.FISHEYE.value).squeeze(-1)  # (num_rays)
            mask = torch.stack([mask, mask, mask], dim=0)

            theta = torch.sqrt(torch.sum(coord_stack**2, dim=-1))
            theta = torch.clip(theta, 0.0, math.pi)

            sin_theta = torch.sin(theta)

            directions_stack[..., 0][mask] = torch.masked_select(coord_stack[..., 0] * sin_theta / theta, mask).float()
            directions_stack[..., 1][mask] = torch.masked_select(coord_stack[..., 1] * sin_theta / theta, mask).float()
            directions_stack[..., 2][mask] = -torch.masked_select(torch.cos(theta), mask).float()

        if CameraType.EQUIRECTANGULAR.value in cam_types:
            mask = (self.camera_type[true_indices] == CameraType.EQUIRECTANGULAR.value).squeeze(-1)  # (num_rays)
            mask = torch.stack([mask, mask, mask], dim=0)

            # For equirect, fx = fy = height = width/2
            # Then coord[..., 0] goes from -1 to 1 and coord[..., 1] goes from -1/2 to 1/2
            theta = -torch.pi * coord_stack[..., 0]  # minus sign for right-handed
            phi = torch.pi * (0.5 - coord_stack[..., 1])
            # use spherical in local camera coordinates (+y up, x=0 and z<0 is theta=0)
            directions_stack[..., 0][mask] = torch.masked_select(-torch.sin(theta) * torch.sin(phi), mask).float()
            directions_stack[..., 1][mask] = torch.masked_select(torch.cos(phi), mask).float()
            directions_stack[..., 2][mask] = torch.masked_select(-torch.cos(theta) * torch.sin(phi), mask).float()

        for value in cam_types:
            if value not in [CameraType.PERSPECTIVE.value, CameraType.FISHEYE.value, CameraType.EQUIRECTANGULAR.value]:
                raise ValueError(f"Camera type {value} not supported.")

        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

        c2w = self.camera_to_worlds[true_indices]
        assert c2w.shape == num_rays_shape + (3, 4)

        if camera_opt_to_camera is not None:
            c2w = pose_utils.multiply(c2w, camera_opt_to_camera)
        rotation = c2w[..., :3, :3]  # (..., 3, 3)
        assert rotation.shape == num_rays_shape + (3, 3)

        directions_stack = torch.sum(
            directions_stack[..., None, :] * rotation, dim=-1
        )  # (..., 1, 3) * (..., 3, 3) -> (..., 3)
        directions_stack, directions_norm = camera_utils.normalize_with_norm(directions_stack, -1)
        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

        origins = c2w[..., :3, 3]  # (..., 3)
        assert origins.shape == num_rays_shape + (3,)

        directions = directions_stack[0]
        assert directions.shape == num_rays_shape + (3,)

        # norms of the vector going between adjacent coords, giving us dx and dy per output ray
        dx = torch.sqrt(torch.sum((directions - directions_stack[1]) ** 2, dim=-1))  # ("num_rays":...,)
        dy = torch.sqrt(torch.sum((directions - directions_stack[2]) ** 2, dim=-1))  # ("num_rays":...,)
        assert dx.shape == num_rays_shape and dy.shape == num_rays_shape

        pixel_area = (dx * dy)[..., None]  # ("num_rays":..., 1)
        assert pixel_area.shape == num_rays_shape + (1,)

        times = self.times[camera_indices, 0] if self.times is not None else None

        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            times=times,
            metadata={"directions_norm": directions_norm[0].detach()},
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
        flattened = self.flatten()
        json_ = {
            "type": "PinholeCamera",
            "cx": flattened[camera_idx].cx.item(),
            "cy": flattened[camera_idx].cy.item(),
            "fx": flattened[camera_idx].fx.item(),
            "fy": flattened[camera_idx].fy.item(),
            "camera_to_world": self.camera_to_worlds[camera_idx].tolist(),
            "camera_index": camera_idx,
            "times": flattened[camera_idx].times.item() if self.times is not None else None,
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

    def get_intrinsics_matrices(self) -> TensorType["num_cameras":..., 3, 3]:
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
        self.height = (self.height * scaling_factor).to(torch.int64)
        self.width = (self.width * scaling_factor).to(torch.int64)
