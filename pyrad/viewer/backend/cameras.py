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

from .geometry import SceneElement, Geometry, Object, LineSegments, PointsGeometry, LineBasicMaterial

# import geometry as g
from . import geometry as g
import numpy as np
import cv2

"""
TODO(ethan): add some notes about the coordinate conventions of the cameras
"""


class Camera(SceneElement):
    def __init__(self):
        super().__init__()


class OrthographicCamera(Camera):
    def __init__(self, left, right, top, bottom, near, far, zoom=1):
        super().__init__()
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.near = near
        self.far = far
        self.zoom = zoom

    def lower(self):
        data = {
            "object": {
                "uuid": self.uuid,
                "type": "OrthographicCamera",
                "left": self.left,
                "right": self.right,
                "top": self.top,
                "bottom": self.bottom,
                "near": self.near,
                "far": self.far,
                "zoom": self.zoom,
            }
        }
        return data


class PerspectiveCamera(Camera):
    def __init__(self, aspect, fov, near, far, zoom=1):
        super().__init__()
        self.aspect = aspect
        self.fov = fov
        self.near = near
        self.far = far

    @staticmethod
    def from_intrinsics(focal_length, aspect_ratio):
        """TODO(ethan): finish this"""
        # convert the intrinsics matrix into three.js compatible format
        return PerspectiveCamera(aspect, fov, near, far)

    def lower(self):
        data = {
            "object": {
                "uuid": self.uuid,
                "type": "PerspectiveCamera",
                "aspect": self.aspect,
                "fov": self.fov,
                "near": self.near,
                "far": self.far,
            }
        }
        return data


class CameraHelper(Camera):
    """NOTE(ethan): Camera parent class might not make sense here?"""

    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def lower(self):
        data = {"object": {"uuid": self.uuid, "type": "CameraHelper", "camera": self.camera.lower()}}
        return data


class ImagePlane(g.Mesh):
    def __init__(self, image, height=1, width=1):
        """TODO(ethan): decide how to deal with the height and width"""
        self.image = image
        geometry = g.PlaneGeometry([width, height])
        material = g.MeshBasicMaterial(
            map=g.ImageTexture(image=g.PngImage(cv2.imencode(".png", self.image[:, :, ::-1])[1].tobytes()))
        )
        super().__init__(geometry, material)


def get_camera_wireframe(scale: float = 0.3, f=4, w=1.5, h=2):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    At https://github.com/hangg7/mvs_visual/blob/275d382a824733a3187a8e3147be184dd6f14795/mvs_visual.py#L54.
    Args:
        f (focal length): this is the focal length
    """
    ul = np.array([-w, h, -f])
    ur = np.array([w, h, -f])
    ll = np.array([-w, -h, -f])
    lr = np.array([w, -h, -f])
    C = np.zeros(3)
    camera_points = [C, ul, C, ur, C, ll, C, lr, C]
    lines = np.stack([x for x in camera_points]) * scale
    return lines


def get_plane_pts(focal_length=(1.0, 1.0), image_size=(10, 10), camera_scale=1, scale_factor=1 / 4):
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
    oW, oH = w / ratio, h / ratio

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


def frustum(scale=1.0, color=[0, 0, 0], focal_length=4, width=1.5, height=2):
    """TODO(ethan): make the scale adjustable depending on the camera size
    color - color of lines using R, G, B. default is black
    """
    # print("linewidth")
    camera_wireframe_lines = get_camera_wireframe(scale=scale, f=focal_length, w=width / 2.0, h=height / 2.0)
    N = len(camera_wireframe_lines)
    colors = np.array([color for _ in range(N)])
    line_segments = LineSegments(
        PointsGeometry(position=camera_wireframe_lines.astype(np.float32).T, color=colors.astype(np.float32).T),
        LineBasicMaterial(vertexColors=True, linewidth=10.0),
    )
    return line_segments
