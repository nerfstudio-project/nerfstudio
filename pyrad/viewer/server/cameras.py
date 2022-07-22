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

"""Logic to render Camera objects in the Viewer"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from . import geometry as g
from .geometry import LineBasicMaterial, LineSegments, PointsGeometry


class ImagePlane(g.Mesh):
    """Returns an image rendered within a specified plane

    Args:
        image: image to be displayed
        height: height of image. Defaults to 1.
        width: width of image. Defaults to 1.
    """

    def __init__(self, image: np.ndarray, height: int = 1, width: int = 1):
        # TODO(ethan): decide how to deal with the height and width
        self.image = image
        geometry = g.PlaneGeometry([width, height])
        material = g.MeshBasicMaterial(
            map=g.ImageTexture(image=g.PngImage(cv2.imencode(".png", self.image[:, :, ::-1])[1].tobytes()))
        )
        super().__init__(geometry, material)


def get_camera_wireframe(scale: float = 0.3, f: int = 4, w: int = 1.5, h: int = 2) -> np.ndarray:
    """Returns a wireframe of a 3D line-plot of a camera symbol.
    At https://github.com/hangg7/mvs_visual/blob/275d382a824733a3187a8e3147be184dd6f14795/mvs_visual.py#L54.

    Args:
        scale: scale of rendering
        f: this is the focal length
        w: width
        h: height

    Returns:
        np.ndarray: stack of points corresponding to wireframe of camera
    """
    ul = np.array([-w, h, -f])
    ur = np.array([w, h, -f])
    ll = np.array([-w, -h, -f])
    lr = np.array([w, -h, -f])
    C = np.zeros(3)
    camera_points = [C, ul, C, ur, C, ll, C, lr, C]
    lines = np.stack(camera_points) * scale
    return lines


def get_plane_pts(
    focal_length: Tuple[float, float] = (1.0, 1.0),
    image_size: Tuple[int, int] = (10, 10),
    camera_scale: int = 1,
    scale_factor: float = 1 / 4,
) -> np.ndarray:
    """Returns points on the image plane given camera intrinsics

    Args:
        focal_length: focal length of camera. Defaults to (1.0, 1.0).
        image_size: height and width of image. Defaults to (10, 10).
        camera_scale: camera intrinsics scale. Defaults to 1.
        scale_factor: image scale. Defaults to 1/4.
    """
    Z = -(focal_length[0] + focal_length[1]) / 2 * camera_scale
    X0, Y0, X1, Y1 = (
        -image_size[0] / 2 * camera_scale,
        image_size[1] / 2 * camera_scale,
        image_size[0] / 2 * camera_scale,
        -image_size[1] / 2 * camera_scale,
    )

    # scale image to plane such that it can go outside of the x0x1 range.
    W, H = X1 - X0, Y0 - Y1
    w, h = image_size
    ratio = min(w / W, h / H)
    oW, oH = w / ratio, h / ratio  # pylint: disable=invalid-name

    X0, Y0, X1, Y1 = -oW / 2, oH / 2, oW / 2, -oH / 2
    wsteps, hsteps = int(w * scale_factor), int(h * scale_factor)
    Ys, Xs = np.meshgrid(
        np.linspace(Y0, Y1, num=hsteps),
        np.linspace(X0, X1, num=wsteps),
        indexing="ij",
    )
    Zs = np.ones_like(Xs) * Z
    plane_pts = np.stack([Xs, Ys, Zs], axis=-1)
    return plane_pts


def frustum(
    scale: float = 1.0,
    color: Optional[List[float]] = None,
    focal_length: int = 4,
    width: float = 1.5,
    height: int = 2,
) -> LineSegments:
    """Draws the camera frustums, returning line segments representing the frustums

    Args:
        scale: scale of the rendering. Defaults to 1.0.
        color: color of lines. Defaults to [0, 0, 0].
        focal_length: focal length of camera. Defaults to 4.
        width: width of wireframe. Defaults to 1.5.
        height: height of wireframe. Defaults to 2.
    """
    # color - color of lines using R, G, B. default is black
    if color is None:
        color = [0, 0, 0]

    # TODO(ethan): make the scale adjustable depending on the camera size
    # print("linewidth")
    camera_wireframe_lines = get_camera_wireframe(scale=scale, f=focal_length, w=width / 2.0, h=height / 2.0)
    N = len(camera_wireframe_lines)
    colors = np.array([color for _ in range(N)])
    line_segments = LineSegments(
        PointsGeometry(position=camera_wireframe_lines.astype(np.float32).T, color=colors.astype(np.float32).T),
        LineBasicMaterial(vertexColors=True, linewidth=10.0),
    )
    return line_segments
