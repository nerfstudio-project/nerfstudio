#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

import dcargs
import mediapy as media
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

console = Console(width=120)

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_name: Name of the renderer output to use.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Number for the output video.
    """
    console.print("[bold green]Creating trajectory video")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx).to(pipeline.device)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            if rendered_output_name not in outputs:
                console.rule("Error", style="red")
                console.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                console.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            image = outputs[rendered_output_name].cpu().numpy()
            images.append(image)

    fps = len(images) / seconds
    with console.status("[yellow]Saving video", spinner="bouncingBall"):
        media.write_video(output_filename, images, fps=fps)
    console.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    console.print(f"[green]Saved video to {output_filename}", justify="center")


@dataclasses.dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer output to use. rgb, depth, etc.
    rendered_output_name: str = "rgb"
    #  Trajectory to render.
    traj: Literal["spiral", "interp", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # A hack to double the number of samples for the nerfacto method.
    double_nerfacto_nerf_samples: bool = True
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            double_nerfacto_nerf_samples=self.double_nerfacto_nerf_samples,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
        )

        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
            # TODO(ethan): pass in the up direction of the camera
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif self.traj == "interp":
            # cameras_a = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
            # cameras_b = pipeline.datamanager.eval_dataloader.get_camera(image_idx=10)
            # camera_path = get_interpolated_camera_path(cameras, steps=30)
            raise NotImplementedError("Interpolated camera path not implemented.")
        elif self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_name=self.rendered_output_name,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: dcargs.extras.get_parser(RenderTrajectory)  # noqa
