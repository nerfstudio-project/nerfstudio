#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

from datetime import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mediapy as media
import numpy as np
import open3d as o3d
import torch
import tyro
from rich.console import Console
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import (
    generate_ellipse_path,
    get_path_from_json,
    get_spiral_path,
)
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.utils import install_checks
from nerfstudio.configs.experiment_config import ExperimentConfig  # pylint: disable=unused-import


import yaml

CONSOLE = Console(width=120)


def _interpolate_trajectory(cameras: Cameras, num_views: int = 300):
    """calculate interpolate path"""

    c2ws = np.stack(cameras.camera_to_worlds.cpu().numpy())

    key_rots = Rotation.from_matrix(c2ws[:, :3, :3])
    key_times = list(range(len(c2ws)))
    slerp = Slerp(key_times, key_rots)
    interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
    render_c2ws = []
    for i in range(num_views):
        time = float(i) / num_views * (len(c2ws) - 1)
        cam_location = interp(time)
        cam_rot = slerp(time).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_location
        render_c2ws.append(c2w)
    render_c2ws = torch.from_numpy(np.stack(render_c2ws, axis=0))

    # use intrinsic of first camera
    camera_path = Cameras(
        fx=cameras[0].fx,
        fy=cameras[0].fy,
        cx=cameras[0].cx,
        cy=cameras[0].cy,
        height=cameras[0].height,
        width=cameras[0].width,
        camera_to_worlds=render_c2ws[:, :3, :4],
        camera_type=cameras[0].camera_type,
    )
    return camera_path


def _render_trajectory_video(
    meshfile: Path,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: str,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    merge_type: Literal["half", "concat"] = "half",
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    # cameras = cameras.to(pipeline.device)

    width = cameras[0].width[0].item()
    height = cameras[0].height[0].item()

    ply = o3d.io.read_triangle_mesh(str(meshfile))
    ply.compute_vertex_normals()
    ply.paint_uniform_color([1, 1, 1])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("rendering", width=width, height=height)

    vis.add_geometry(ply)
    vis.get_render_option().load_from_json("scripts/render.json")

    output_image_dir = output_filename.parent / output_filename.stem
    for render_name in rendered_output_names:
        output_image_dir_cur = output_image_dir / render_name
        output_image_dir_cur.mkdir(parents=True, exist_ok=True)

    num_frames = cameras.size
    index = -1
    rendered_images = []

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        nonlocal index
        nonlocal cameras
        nonlocal rendered_images
        if index >= 0:
            images = []
            for render_name in rendered_output_names:
                output_image_dir_cur = output_image_dir / render_name

                if render_name == "normal":
                    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
                elif render_name == "rgb":
                    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color

                vis.capture_screen_image(str(output_image_dir_cur / f"{index:05d}.png"), True)

                images.append(cv2.imread(str(output_image_dir_cur / f"{index:05d}.png"))[:, :, ::-1])
            if merge_type == "concat":
                images = np.concatenate(images, axis=1)
            elif merge_type == "half":
                mask = np.zeros_like(images[0])
                mask[:, : mask.shape[1] // 2, :] = 1
                images = images[0] * mask + images[1] * (1 - mask)
            rendered_images.append(images)
        index = index + 1
        if index < num_frames:

            param = ctr.convert_to_pinhole_camera_parameters()
            camera = cameras[index]
            width = camera.width[0].item()
            height = camera.height[0].item()
            fx = camera.fx[0].item()
            fy = camera.fy[0].item()
            cx = camera.cx[0].item()
            cy = camera.cy[0].item()
            camera = cameras[index]

            param.intrinsic.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

            extrinsic = np.eye(4)
            extrinsic[:3, :] = camera.camera_to_worlds.cpu().numpy()
            extrinsic[:3, 1:3] *= -1
            param.extrinsic = np.linalg.inv(extrinsic)

            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        else:
            vis.register_animation_callback(None)
            vis.destroy_window()

        return False

    vis.register_animation_callback(move_forward)
    vis.run()
    if output_format == "video":
        fps = len(rendered_images) / seconds
        # rendered_images = rendered_images + rendered_images[::-1]
        media.write_video(output_filename, rendered_images, fps=fps)


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    meshfile: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb", "normal"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename", "interpolate", "ellipse"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # pfs of the video
    fps: int = 24
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    merge_type: Literal["half", "concat"] = "half"

    data: AnnotatedDataParserUnion = field(default_factory=SDFStudioDataParserConfig)
    num_views: int = 300

    def main(self) -> None:
        """Main function."""

        install_checks.check_ffmpeg_installed()
        seconds = self.seconds
        if self.output_format == "video":
            assert str(self.output_path)[-4:] == ".mp4"

        if self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        elif self.traj == "interpolate":
            # load training data and interpolate path
            outputs = self.data.setup()._generate_dataparser_outputs()
            camera_path = _interpolate_trajectory(cameras=outputs.cameras, num_views=self.num_views)
            seconds = camera_path.size / 24
        elif self.traj == "spiral":
            outputs = self.data.setup()._generate_dataparser_outputs()
            camera_path = get_spiral_path(camera=outputs.cameras, steps=self.num_views, radius=1.0)
            seconds = camera_path.size / 24
        elif self.traj == "ellipse":
            outputs = self.data.setup()._generate_dataparser_outputs()
            camera_path = generate_ellipse_path(cameras=outputs.cameras, n_frames=self.num_views, const_speed=False)
            seconds = camera_path.size / self.fps
        else:
            assert_never(self.traj)

        _render_trajectory_video(
            self.meshfile,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            merge_type=self.merge_type,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
