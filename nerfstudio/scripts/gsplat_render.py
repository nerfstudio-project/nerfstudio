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

#!/usr/bin/env python
"""
This is a standalone PLY renderer based on gsplat lib (https://github.com/nerfstudio-project/gsplat)
It takes a render trajectory json file, whose format can either be Project Turner hemispin render
format or nerfstudio format. One can optionally specify the background color for the renders. 
The rendered images will be saved to a directory as png.
"""
from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from gsplat.sh import SphericalHarmonics

import json
import tyro
from pathlib import Path
from typing import List, NamedTuple, Optional
import numpy as np
import math
import torch
import torchvision
from tqdm import tqdm
import meshio


class CameraInfo(NamedTuple):
    uid: int
    viewmat: torch.Tensor
    projmat: torch.Tensor
    FovY: float
    FovX: float
    image_name: str
    width: int
    height: int


def n_sh_level(n_sh_coefs: int):
    if n_sh_coefs == 1:
        return 0
    if n_sh_coefs == 4:
        return 1
    if n_sh_coefs == 9:
        return 2
    if n_sh_coefs == 16:
        return 3
    if n_sh_coefs == 25:
        return 4
    raise ValueError(f"unsupported n_sh_coefs={n_sh_coefs}")


class GaussianModel(object):
    def __init__(self, point_cloud):
        """
        Load Gaussian Splat model from a PLY format file
        """
        ## pc must be copied otherwise the following error might be possible when converting to torch.Tensor:
        ## ValueError: given numpy array strides not a multiple of the element byte size. Copy the numpy array to reallocate the memory.
        pc = meshio.read(point_cloud).copy()

        self.xyz = torch.tensor(pc.points)
        num_points = self.xyz.shape[0]
        self.opacity = torch.tensor(pc.point_data["opacity"])[:, None]

        # load the gaussians parameters
        sx = torch.tensor(pc.point_data["scale_0"])
        sy = torch.tensor(pc.point_data["scale_1"])
        sz = torch.tensor(pc.point_data["scale_2"])
        self.scales = torch.cat((sx[:, None], sy[:, None], sz[:, None]), dim=1)

        w = torch.tensor(pc.point_data["rot_0"])
        x = torch.tensor(pc.point_data["rot_1"])
        y = torch.tensor(pc.point_data["rot_2"])
        z = torch.tensor(pc.point_data["rot_3"])
        self.rots = torch.cat((w[:, None], x[:, None], y[:, None], z[:, None]), dim=1)

        if "f_dc_0" in pc.point_data:
            r = torch.tensor(pc.point_data["f_dc_0"])
            g = torch.tensor(pc.point_data["f_dc_1"])
            b = torch.tensor(pc.point_data["f_dc_2"])
            sh_0 = torch.cat((r[:, None], g[:, None], b[:, None]), dim=1)

            f_rest_highest_idx = max([int(k.split("_")[-1]) for k in pc.point_data.keys() if k.startswith("f_rest_")])
            sh_rest = torch.zeros((num_points, f_rest_highest_idx + 1), dtype=torch.float32)
            f_rest_range = list(range(0, f_rest_highest_idx + 1))
            for idx in f_rest_range:
                sh_rest[:, idx] = torch.tensor(pc.point_data[f"f_rest_{idx}"])
            self.n_sh_coefs = (f_rest_highest_idx + 1) // 3 + 1
            self.n_sh_level = n_sh_level(self.n_sh_coefs)
            assert self.n_sh_level > 0

            self.sh_coefs = torch.zeros((num_points, self.n_sh_coefs, 3), dtype=torch.float32)
            self.sh_coefs[:, 0, :] = sh_0
            self.sh_coefs[:, 1:, :] = sh_rest.reshape(num_points, 3, self.n_sh_coefs - 1).transpose(1, 2)
        elif "red" in pc.point_data:
            self.colors = torch.zeros((num_points, 3), dtype=torch.float32)
            self.colors[:, 0] = torch.tensor(pc.point_data["red"], dtype=torch.uint8).float() / 255.0
            self.colors[:, 1] = torch.tensor(pc.point_data["green"], dtype=torch.uint8).float() / 255.0
            self.colors[:, 2] = torch.tensor(pc.point_data["blue"], dtype=torch.uint8).float() / 255.0
            self.n_sh_level = 0
        else:
            raise RuntimeError("Cannot import model color or spherical harmonics from PLY.")

        self.xyz = self.xyz.to(device="cuda")
        self.opacity = self.opacity.to(device="cuda")
        self.scales = self.scales.to(device="cuda")
        self.rots = self.rots.to(device="cuda")
        if self.n_sh_level > 0:
            self.sh_coefs = self.sh_coefs.to(device="cuda")
        else:
            self.colors = self.colors.to(device="cuda")

    def render(self, cam_info: CameraInfo, background: torch.Tensor):
        fx = fov2focal(cam_info.FovX, cam_info.width)
        fy = fov2focal(cam_info.FovY, cam_info.height)
        cx = cam_info.width / 2.0
        cy = cam_info.height / 2.0
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (cam_info.width + BLOCK_X - 1) // BLOCK_X,
            (cam_info.height + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

        viewmat = cam_info.viewmat.float().to(device="cuda")
        projmat = cam_info.projmat.float().to(device="cuda")
        xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            self.xyz,
            torch.exp(self.scales),
            1,
            self.rots / self.rots.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            fx,
            fy,
            cx,
            cy,
            cam_info.height,
            cam_info.width,
            tile_bounds,
        )  # type: ignore
        torch.cuda.synchronize()

        if self.n_sh_level > 0:
            c2w = torch.inverse(viewmat)
            viewdirs = self.xyz - c2w[None, :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = SphericalHarmonics.apply(self.n_sh_level, viewdirs, self.sh_coefs)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = self.colors
        rgb = RasterizeGaussians.apply(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(self.opacity),
            cam_info.height,
            cam_info.width,
            background,
        )  # type: ignore
        torch.cuda.synchronize()

        return torch.clamp(rgb, max=1.0)  # type: ignore


def projection_matrix(znear, zfar, fovx, fovy, device="cpu"):
    t = znear * math.tan(0.5 * fovy)
    b = -t
    right = znear * math.tan(0.5 * fovx)
    left = -right
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (right - left), 0.0, (right + left) / (right - left), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def load_trajectory(filename: Path):
    """Loads the cameras from a trajectory.json file like the ones used for rendering in nerfstudio

    Parameters
    ----------
    filename: Path
        JSON with the cameras in the nerfstudio camera-trajectory.json format

    Returns
    -------
    List[CameraInfo]
        list with the cameras in CameraInfo format

    """
    with filename.open("r") as fin:
        data = json.load(fin)

    width = data["render_width"]
    height = data["render_height"]

    bg_color = None
    if "crop" in data:
        bg_color = data["crop"]["crop_bg_color"]
        bg_color = [bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]

    trajectory = data["camera_path"]
    metadata = data["metadata"]
    crop = data["crop"]

    def cam_info(uid, cam):
        c2w = np.array(cam["camera_to_world"]).reshape(4, 4)
        # opencv to opengl
        c2w[0:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)

        # we are using nerfstudio convention here, where fov is in degrees
        # and corresponds to fovy. But gaussian splatting expects radians
        fovy = cam["fov"] * np.pi / 180
        fovx = focal2fov(fov2focal(fovy, height), width)

        projmat = projection_matrix(0.001, 1000, fovx, fovy, device="cuda")
        return CameraInfo(
            uid=uid,
            viewmat=torch.tensor(w2c),
            projmat=projmat,
            FovX=fovx,
            FovY=fovy,
            width=width,
            height=height,
            image_name=f"{uid:05}",
        )

    cam_infos = [cam_info(i, cam) for i, cam in enumerate(trajectory)]
    return cam_infos, bg_color, metadata, crop


def load_ns_transforms(filename: Path) -> List[CameraInfo]:
    """Loads the cameras from a nerfstudio transform.json file

    Parameters
    ----------
    filename: Path
        JSON with the cameras in the nerfstudio transforms.json format

    Returns
    -------
    List[CameraInfo]
        list with the cameras in CameraInfo format

    """
    with filename.open("r") as fin:
        data = json.load(fin)

    width = data["w"]
    height = data["h"]

    fl_x = data["fl_x"]
    fl_y = data["fl_y"]
    fovx = focal2fov(fl_x, width)
    fovy = focal2fov(fl_y, height)

    trajectory = data["frames"]

    def cam_info(cam):
        name = Path(cam["file_path"]).stem
        # we make the assumption that names are frame_000xx
        uid = int(name.split("_")[1])
        c2w = np.array(cam["transform_matrix"]).reshape(4, 4)
        # opencv to opengl
        c2w[0:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device="cuda")
        return CameraInfo(
            uid=uid,
            viewmat=torch.tensor(w2c),
            projmat=projmat,
            FovX=fovx,
            FovY=fovy,
            width=width,
            height=height,
            image_name=name,
        )

    cam_infos = [cam_info(cam) for cam in trajectory]
    return cam_infos


def render_set(
    render_path: Path, point_cloud: Path, cam_infos: List[CameraInfo], bg_color: Optional[List[float]] = None
):
    with torch.inference_mode():
        gaussians = GaussianModel(point_cloud)
        if bg_color is None:
            bg_color = [1, 1, 1]  # white background

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        for idx, cam_info in enumerate(tqdm(cam_infos, desc="Rendering progress")):
            with torch.no_grad():
                image = gaussians.render(cam_info, background)
            # writing image to disk may be slow, a better way is to directly encode a video
            torchvision.utils.save_image(image.permute(2, 0, 1), render_path / (cam_info.image_name + ".png"))


def main(
    point_cloud: Path,
    render_path: Path,
    camera_trajectory: Optional[Path] = None,
    ns_transform: Optional[Path] = None,
    bg_color: Optional[List[float]] = None,
):
    print(f"Rendering using PLY={point_cloud}")
    render_path.mkdir(parents=True, exist_ok=True)

    bg = None
    if camera_trajectory is not None:
        # 2023-12-05: TODO: metadata, crop are not used
        cam_infos, bg, metadata, crop = load_trajectory(camera_trajectory)
    elif ns_transform is not None:
        cam_infos = load_ns_transforms(ns_transform)
    else:
        raise ValueError("Either --camera-trajectory or --ns-trajectory must be provided")

    if bg_color:
        assert len(bg_color) == 3
        bg = bg_color

    render_set(render_path=render_path, point_cloud=point_cloud, cam_infos=cam_infos, bg_color=bg)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
