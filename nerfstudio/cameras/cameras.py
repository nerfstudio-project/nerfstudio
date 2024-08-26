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
Camera Models
"""

import base64
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor
from torch.nn import Parameter

import nerfstudio.utils.math
import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[torch.device, str]


class CameraType(Enum):
    """Supported camera types."""

    PERSPECTIVE = auto()
    FISHEYE = auto()
    EQUIRECTANGULAR = auto()
    OMNIDIRECTIONALSTEREO_L = auto()
    OMNIDIRECTIONALSTEREO_R = auto()
    VR180_L = auto()
    VR180_R = auto()
    ORTHOPHOTO = auto()
    FISHEYE624 = auto()


CAMERA_MODEL_TO_TYPE = {
    "SIMPLE_PINHOLE": CameraType.PERSPECTIVE,
    "PINHOLE": CameraType.PERSPECTIVE,
    "SIMPLE_RADIAL": CameraType.PERSPECTIVE,
    "RADIAL": CameraType.PERSPECTIVE,
    "OPENCV": CameraType.PERSPECTIVE,
    "OPENCV_FISHEYE": CameraType.FISHEYE,
    "EQUIRECTANGULAR": CameraType.EQUIRECTANGULAR,
    "OMNIDIRECTIONALSTEREO_L": CameraType.OMNIDIRECTIONALSTEREO_L,
    "OMNIDIRECTIONALSTEREO_R": CameraType.OMNIDIRECTIONALSTEREO_R,
    "VR180_L": CameraType.VR180_L,
    "VR180_R": CameraType.VR180_R,
    "ORTHOPHOTO": CameraType.ORTHOPHOTO,
    "FISHEYE624": CameraType.FISHEYE624,
}


@dataclass(init=False)
class Cameras(TensorDataclass):
    """Dataparser outputs for the image dataset and the ray generator.

    If a single value is provided, it is broadcasted to all cameras.

    Args:
        camera_to_worlds: Camera to world matrices. Tensor of per-image c2w matrices, in [R | t] format
        fx: Focal length x
        fy: Focal length y
        cx: Principal point x
        cy: Principal point y
        width: Image width
        height: Image height
        distortion_params: distortion coefficients (OpenCV 6 radial or 6-2-4 radial, tangential, thin-prism for Fisheye624)
        camera_type: Type of camera model. This will be an int corresponding to the CameraType enum.
        times: Timestamps for each camera
        metadata: Additional metadata or data needed for interpolation, will mimic shape of the cameras
            and will be broadcasted to the rays generated from any derivative RaySamples we create with this
    """

    camera_to_worlds: Float[Tensor, "*num_cameras 3 4"]
    fx: Float[Tensor, "*num_cameras 1"]
    fy: Float[Tensor, "*num_cameras 1"]
    cx: Float[Tensor, "*num_cameras 1"]
    cy: Float[Tensor, "*num_cameras 1"]
    width: Shaped[Tensor, "*num_cameras 1"]
    height: Shaped[Tensor, "*num_cameras 1"]
    distortion_params: Optional[Float[Tensor, "*num_cameras 6"]]
    camera_type: Int[Tensor, "*num_cameras 1"]
    times: Optional[Float[Tensor, "num_cameras 1"]]
    metadata: Optional[Dict]

    def __init__(
        self,
        camera_to_worlds: Float[Tensor, "*batch_c2ws 3 4"],
        fx: Union[Float[Tensor, "*batch_fxs 1"], float],
        fy: Union[Float[Tensor, "*batch_fys 1"], float],
        cx: Union[Float[Tensor, "*batch_cxs 1"], float],
        cy: Union[Float[Tensor, "*batch_cys 1"], float],
        width: Optional[Union[Shaped[Tensor, "*batch_ws 1"], int]] = None,
        height: Optional[Union[Shaped[Tensor, "*batch_hs 1"], int]] = None,
        distortion_params: Optional[Float[Tensor, "*batch_dist_params 6"]] = None,
        camera_type: Union[
            Int[Tensor, "*batch_cam_types 1"],
            int,
            List[CameraType],
            CameraType,
        ] = CameraType.PERSPECTIVE,
        times: Optional[Float[Tensor, "num_cameras"]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Initializes the Cameras object.

        Note on Input Tensor Dimensions: All of these tensors have items of dimensions Shaped[Tensor, "3 4"]
        (in the case of the c2w matrices), Shaped[Tensor, "6"] (in the case of distortion params), or
        Shaped[Tensor, "1"] (in the case of the rest of the elements). The dimensions before that are
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

        self.metadata = metadata

        self.__post_init__()  # This will do the dataclass post_init and broadcast all the tensors

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
            fc_xy = torch.tensor([fc_xy], device=self.device)
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
            Int[Tensor, "*batch_cam_types 1"], Int[Tensor, "*batch_cam_types"], int, List[CameraType], CameraType
        ],
    ) -> Int[Tensor, "*num_cameras 1"]:
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
        h_w: Union[Shaped[Tensor, "*batch_hws 1"], Shaped[Tensor, "*batch_hws"], int, None],
        c_x_y: Shaped[Tensor, "*batch_cxys"],
    ) -> Shaped[Tensor, "*num_cameras 1"]:
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
    def image_height(self) -> Shaped[Tensor, "*num_cameras 1"]:
        """Returns the height of the images."""
        return self.height

    @property
    def image_width(self) -> Shaped[Tensor, "*num_cameras 1"]:
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
    ) -> Float[Tensor, "height width 2"]:
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
            image_height = torch.max(self.image_height.view(-1)).item()
            image_width = torch.max(self.image_width.view(-1)).item()
            image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
            image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        else:
            image_height = self.image_height[index].item()
            image_width = self.image_width[index].item()
            image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
            image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        return image_coords

    def generate_rays(
        self,
        camera_indices: Union[Int[Tensor, "*num_rays num_cameras_batch_dims"], int],
        coords: Optional[Float[Tensor, "*num_rays 2"]] = None,
        camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
        distortion_params_delta: Optional[Float[Tensor, "*num_rays 6"]] = None,
        keep_shape: Optional[bool] = None,
        disable_distortion: bool = False,
        aabb_box: Optional[SceneBox] = None,
        obb_box: Optional[OrientedBox] = None,
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
            aabb_box: if not None will calculate nears and fars of the ray according to aabb box intersection

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
            coords = cameras.get_image_coords(index=tuple(index))  # (h, w, 2)
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

        raybundle = cameras._generate_rays_from_coords(
            camera_indices, coords, camera_opt_to_camera, distortion_params_delta, disable_distortion=disable_distortion
        )

        # If we have mandated that we don't keep the shape, then we flatten
        if keep_shape is False:
            raybundle = raybundle.flatten()

        if aabb_box is not None or obb_box is not None:
            with torch.no_grad():
                rays_o = raybundle.origins.contiguous()
                rays_d = raybundle.directions.contiguous()

                shape = rays_o.shape

                rays_o = rays_o.reshape((-1, 3))
                rays_d = rays_d.reshape((-1, 3))

                if aabb_box is not None:
                    tensor_aabb = Parameter(aabb_box.aabb.flatten(), requires_grad=False)
                    tensor_aabb = tensor_aabb.to(rays_o.device)
                    t_min, t_max = nerfstudio.utils.math.intersect_aabb(rays_o, rays_d, tensor_aabb)
                elif obb_box is not None:
                    t_min, t_max = nerfstudio.utils.math.intersect_obb(rays_o, rays_d, obb_box)
                else:
                    assert False

                t_min = t_min.reshape([shape[0], shape[1], 1])
                t_max = t_max.reshape([shape[0], shape[1], 1])

                raybundle.nears = t_min
                raybundle.fars = t_max

        # TODO: We should have to squeeze the last dimension here if we started with zero batch dims, but never have to,
        # so there might be a rogue squeeze happening somewhere, and this may cause some unintended behaviour
        # that we haven't caught yet with tests
        return raybundle

    def _generate_rays_from_coords(
        self,
        camera_indices: Int[Tensor, "*num_rays num_cameras_batch_dims"],
        coords: Float[Tensor, "*num_rays 2"],
        camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
        distortion_params_delta: Optional[Float[Tensor, "*num_rays 6"]] = None,
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
        coord = torch.stack([(x - cx) / fx, (y - cy) / fy], -1)  # (num_rays, 2)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, (y - cy) / fy], -1)  # (num_rays, 2)
        coord_y_offset = torch.stack([(x - cx) / fx, (y - cy + 1) / fy], -1)  # (num_rays, 2)
        assert (
            coord.shape == num_rays_shape + (2,)
            and coord_x_offset.shape == num_rays_shape + (2,)
            and coord_y_offset.shape == num_rays_shape + (2,)
        )

        # Stack image coordinates and image coordinates offset by 1, check shapes too
        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)  # (3, num_rays, 2)
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Undistorts our images according to our distortion parameters
        distortion_params = None
        if not disable_distortion:
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
                if mask.any() and (distortion_params != 0).any():
                    coord_stack[coord_mask, :] = camera_utils.radial_and_tangential_undistort(
                        coord_stack[coord_mask, :].reshape(3, -1, 2),
                        distortion_params[mask, :],
                    ).reshape(-1, 2)

        # Switch from OpenCV to OpenGL
        coord_stack[..., 1] *= -1

        # Make sure after we have undistorted our images, the shapes are still correct
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Gets our directions for all our rays in camera coordinates and checks shapes at the end
        # Here, directions_stack is of shape (3, num_rays, 3)
        # directions_stack[0] is the direction for ray in camera coordinates
        # directions_stack[1] is the direction for ray in camera coordinates offset by 1 in x
        # directions_stack[2] is the direction for ray in camera coordinates offset by 1 in y
        cam_types = torch.unique(self.camera_type, sorted=False)
        directions_stack = torch.empty((3,) + num_rays_shape + (3,), device=self.device)

        c2w = self.camera_to_worlds[true_indices]
        assert c2w.shape == num_rays_shape + (3, 4)

        def _compute_rays_for_omnidirectional_stereo(
            eye: Literal["left", "right"],
        ) -> Tuple[Float[Tensor, "num_rays_shape 3"], Float[Tensor, "3 num_rays_shape 3"]]:
            """Compute the rays for an omnidirectional stereo camera

            Args:
                eye: Which eye to compute rays for.

            Returns:
                A tuple containing the origins and the directions of the rays.
            """
            # Directions calculated similarly to equirectangular
            ods_cam_type = (
                CameraType.OMNIDIRECTIONALSTEREO_R.value if eye == "right" else CameraType.OMNIDIRECTIONALSTEREO_L.value
            )
            mask = (self.camera_type[true_indices] == ods_cam_type).squeeze(-1)
            mask = torch.stack([mask, mask, mask], dim=0)
            theta = -torch.pi * coord_stack[..., 0]
            phi = torch.pi * (0.5 - coord_stack[..., 1])

            directions_stack[..., 0][mask] = torch.masked_select(-torch.sin(theta) * torch.sin(phi), mask).float()
            directions_stack[..., 1][mask] = torch.masked_select(torch.cos(phi), mask).float()
            directions_stack[..., 2][mask] = torch.masked_select(-torch.cos(theta) * torch.sin(phi), mask).float()

            vr_ipd = 0.064  # IPD in meters (note: scale of NeRF must be true to life and can be adjusted with the Blender add-on)
            isRightEye = 1 if eye == "right" else -1

            # find ODS camera position
            c2w = self.camera_to_worlds[true_indices]
            assert c2w.shape == num_rays_shape + (3, 4)
            transposedC2W = c2w[0][0].t()
            ods_cam_position = transposedC2W[3].repeat(c2w.shape[1], 1)

            rotation = c2w[..., :3, :3]

            ods_theta = -torch.pi * ((x - cx) / fx)[0]

            # local axes of ODS camera
            ods_x_axis = torch.tensor([1, 0, 0], device=c2w.device)
            ods_z_axis = torch.tensor([0, 0, -1], device=c2w.device)

            # circle of ODS ray origins
            ods_origins_circle = (
                isRightEye * (vr_ipd / 2.0) * (ods_x_axis.repeat(c2w.shape[1], 1)) * (torch.cos(ods_theta))[:, None]
                + isRightEye * (vr_ipd / 2.0) * (ods_z_axis.repeat(c2w.shape[1], 1)) * (torch.sin(ods_theta))[:, None]
            )

            # rotate origins to match the camera rotation
            for i in range(ods_origins_circle.shape[0]):
                ods_origins_circle[i] = rotation[0][0] @ ods_origins_circle[i] + ods_cam_position[0]
            ods_origins_circle = ods_origins_circle.unsqueeze(0).repeat(c2w.shape[0], 1, 1)

            # assign final camera origins
            c2w[..., :3, 3] = ods_origins_circle

            return ods_origins_circle, directions_stack

        def _compute_rays_for_vr180(
            eye: Literal["left", "right"],
        ) -> Tuple[Float[Tensor, "num_rays_shape 3"], Float[Tensor, "3 num_rays_shape 3"]]:
            """Compute the rays for a VR180 camera

            Args:
                eye: Which eye to compute rays for.

            Returns:
                A tuple containing the origins and the directions of the rays.
            """
            # Directions calculated similarly to equirectangular
            vr180_cam_type = CameraType.VR180_R.value if eye == "right" else CameraType.VR180_L.value
            mask = (self.camera_type[true_indices] == vr180_cam_type).squeeze(-1)
            mask = torch.stack([mask, mask, mask], dim=0)

            # adjusting theta range to +/-90 deg
            theta = -torch.pi * ((x - cx) / (fx * 2))[0]
            phi = torch.pi * (0.5 - coord_stack[..., 1])

            directions_stack[..., 0][mask] = torch.masked_select(-torch.sin(theta) * torch.sin(phi), mask).float()
            directions_stack[..., 1][mask] = torch.masked_select(torch.cos(phi), mask).float()
            directions_stack[..., 2][mask] = torch.masked_select(-torch.cos(theta) * torch.sin(phi), mask).float()

            vr_ipd = 0.064  # IPD in meters (note: scale of NeRF must be true to life and can be adjusted with the Blender add-on)
            isRightEye = 1 if eye == "right" else -1

            # find VR180 camera position
            c2w = self.camera_to_worlds[true_indices]
            assert c2w.shape == num_rays_shape + (3, 4)
            transposedC2W = c2w[0][0].t()
            vr180_cam_position = transposedC2W[3].repeat(c2w.shape[1], 1)

            rotation = c2w[..., :3, :3]

            # interocular axis of the VR180 camera
            vr180_x_axis = torch.tensor([1, 0, 0], device=c2w.device)

            # VR180 ray origins of horizontal offset
            vr180_origins = isRightEye * (vr_ipd / 2.0) * (vr180_x_axis.repeat(c2w.shape[1], 1))

            # rotate origins to match the camera rotation
            for i in range(vr180_origins.shape[0]):
                vr180_origins[i] = rotation[0][0] @ vr180_origins[i] + vr180_cam_position[0]

            vr180_origins = vr180_origins.unsqueeze(0).repeat(c2w.shape[0], 1, 1)

            # assign final camera origins
            c2w[..., :3, 3] = vr180_origins

            return vr180_origins, directions_stack

        for cam in cam_types:
            if CameraType.PERSPECTIVE.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.PERSPECTIVE.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)
                directions_stack[..., 0][mask] = torch.masked_select(coord_stack[..., 0], mask).float()
                directions_stack[..., 1][mask] = torch.masked_select(coord_stack[..., 1], mask).float()
                directions_stack[..., 2][mask] = -1.0

            elif CameraType.FISHEYE.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.FISHEYE.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)

                theta = torch.sqrt(torch.sum(coord_stack**2, dim=-1))
                theta = torch.clip(theta, 0.0, math.pi)

                sin_theta = torch.sin(theta)

                directions_stack[..., 0][mask] = torch.masked_select(
                    coord_stack[..., 0] * sin_theta / theta, mask
                ).float()
                directions_stack[..., 1][mask] = torch.masked_select(
                    coord_stack[..., 1] * sin_theta / theta, mask
                ).float()
                directions_stack[..., 2][mask] = -torch.masked_select(torch.cos(theta), mask).float()

            elif CameraType.EQUIRECTANGULAR.value in cam_types:
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

            elif CameraType.OMNIDIRECTIONALSTEREO_L.value in cam_types:
                ods_origins_circle, directions_stack = _compute_rays_for_omnidirectional_stereo("left")
                # assign final camera origins
                c2w[..., :3, 3] = ods_origins_circle

            elif CameraType.OMNIDIRECTIONALSTEREO_R.value in cam_types:
                ods_origins_circle, directions_stack = _compute_rays_for_omnidirectional_stereo("right")
                # assign final camera origins
                c2w[..., :3, 3] = ods_origins_circle

            elif CameraType.VR180_L.value in cam_types:
                vr180_origins, directions_stack = _compute_rays_for_vr180("left")
                # assign final camera origins
                c2w[..., :3, 3] = vr180_origins

            elif CameraType.VR180_R.value in cam_types:
                vr180_origins, directions_stack = _compute_rays_for_vr180("right")
                # assign final camera origins
                c2w[..., :3, 3] = vr180_origins

            elif CameraType.ORTHOPHOTO.value in cam_types:
                # here the focal length determine the imaging area, the smaller fx, the bigger imaging area.
                mask = (self.camera_type[true_indices] == CameraType.ORTHOPHOTO.value).squeeze(-1)
                dir_mask = torch.stack([mask, mask, mask], dim=0)
                # in orthophoto cam, all rays have same direction, dir = R @ [0, 0, 1], R will be applied following.
                directions_stack[dir_mask] = torch.tensor(
                    [0.0, 0.0, -1.0], dtype=directions_stack.dtype, device=directions_stack.device
                )
                # in orthophoto cam, ray origins are grids, then transform grids with c2w, c2w @ P.
                grids = coord[mask]
                grids[..., 1] *= -1.0  # convert to left-hand system.
                grids = torch.cat([grids, torch.zeros_like(grids[..., -1:]), torch.ones_like(grids[..., -1:])], dim=-1)
                grids = torch.matmul(c2w[mask], grids[..., None]).squeeze(-1)
                c2w[..., :3, 3][mask] = grids

            elif CameraType.FISHEYE624.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.FISHEYE624.value).squeeze(-1)  # (num_rays)
                coord_mask = torch.stack([mask, mask, mask], dim=0)

                # fisheye624 requires pixel coordinates to unproject, so we need to recomput the offsets in pixel coords.
                pcoord = torch.stack([x, y], -1)  # (num_rays, 2)
                pcoord_x_offset = torch.stack([x + 1, y], -1)  # (num_rays, 2)
                pcoord_y_offset = torch.stack([x, y + 1], -1)  # (num_rays, 2)

                # Stack image coordinates and image coordinates offset by 1, check shapes too
                pcoord_stack = torch.stack([pcoord, pcoord_x_offset, pcoord_y_offset], dim=0)  # (3, num_rays, 2)

                assert distortion_params is not None
                masked_coords = pcoord_stack[coord_mask, :]
                # The fisheye unprojection does not rely on planar/pinhole unprojection, thus the method needs
                # to access the focal length and principle points directly.
                camera_params = torch.cat(
                    [
                        fx[mask].unsqueeze(1),
                        fy[mask].unsqueeze(1),
                        cx[mask].unsqueeze(1),
                        cy[mask].unsqueeze(1),
                        distortion_params[mask, :],
                    ],
                    dim=1,
                )
                directions_stack[coord_mask] = camera_utils.fisheye624_unproject(masked_coords, camera_params)

            else:
                raise ValueError(f"Camera type {cam} not supported.")

        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

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

        metadata = (
            self._apply_fn_to_dict(self.metadata, lambda x: x[true_indices]) if self.metadata is not None else None
        )
        if metadata is not None:
            metadata["directions_norm"] = directions_norm[0].detach()
        else:
            metadata = {"directions_norm": directions_norm[0].detach()}

        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            times=times,
            metadata=metadata,
        )

    def to_json(
        self, camera_idx: int, image: Optional[Float[Tensor, "height width 2"]] = None, max_size: Optional[int] = None
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
        times = flattened[camera_idx].times
        if times is not None:
            times = times.item()
        json_ = {
            "type": "PinholeCamera",
            "cx": flattened[camera_idx].cx.item(),
            "cy": flattened[camera_idx].cy.item(),
            "fx": flattened[camera_idx].fx.item(),
            "fy": flattened[camera_idx].fy.item(),
            "camera_to_world": self.camera_to_worlds[camera_idx].tolist(),
            "camera_index": camera_idx,
            "times": times,
        }
        if image is not None:
            image_uint8 = (image * 255).detach().type(torch.uint8)
            if max_size is not None:
                image_uint8 = image_uint8.permute(2, 0, 1)

                # torchvision can be slow to import, so we do it lazily.
                import torchvision.transforms.functional as TF

                image_uint8 = TF.resize(image_uint8, max_size, antialias=None)  # type: ignore
                image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            data = cv2.imencode(".jpg", image_uint8)[1].tobytes()  # type: ignore
            json_["image"] = str("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
        return json_

    def get_intrinsics_matrices(self) -> Float[Tensor, "*num_cameras 3 3"]:
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
        self,
        scaling_factor: Union[Shaped[Tensor, "*num_cameras"], Shaped[Tensor, "*num_cameras 1"], float, int],
        scale_rounding_mode: str = "floor",
    ) -> None:
        """Rescale the output resolution of the cameras.

        Args:
            scaling_factor: Scaling factor to apply to the output resolution.
            scale_rounding_mode: round down or round up when calculating the scaled image height and width
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
        if scale_rounding_mode == "floor":
            self.height = (self.height * scaling_factor).to(torch.int64)
            self.width = (self.width * scaling_factor).to(torch.int64)
        elif scale_rounding_mode == "round":
            self.height = torch.floor(0.5 + (self.height * scaling_factor)).to(torch.int64)
            self.width = torch.floor(0.5 + (self.width * scaling_factor)).to(torch.int64)
        elif scale_rounding_mode == "ceil":
            self.height = torch.ceil(self.height * scaling_factor).to(torch.int64)
            self.width = torch.ceil(self.width * scaling_factor).to(torch.int64)
        else:
            raise ValueError("Scale rounding mode must be 'floor', 'round' or 'ceil'.")
