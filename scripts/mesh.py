"""
Script to create a mesh from a trained model.

Poisson surface reconstruction from Open3D
http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html

1. load the model
2. get the training cameras
3. create the point cloud
4. clip the point cloud based on the AABB
5. run the poisson surface reconstruction
6. save the mesh

# TODO: switch to median depth
# TODO: combine the rgb and depth renderings together
# TODO: fix logic with rendered_resolution_scaling_factor so it copies the Cameras object rather than editing it

python scripts/mesh.py --load-config outputs/data-nerfstudio-poster/nerfacto/2022-10-20_085111/config.yml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import tyro
from pyntcloud import PyntCloud
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


def _render_trajectory(
    pipeline: Pipeline,
    cameras: Cameras,
    rendered_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
) -> List[np.ndarray]:
    """Helper function to create a video of a trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Number for the output video.

    Returns:
        List of images
    """
    CONSOLE.print("[bold green]Creating trajectory video")
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
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            image = outputs[rendered_output_name].cpu().numpy()
            images.append(image)
    return images


def main(load_config: Path, downscale_factor: int = 4):
    """Main function.

    Args:
        load_config: Path to config YAML file.
        downscale_factor: Downscale factor for the images.
    """
    _, pipeline, _ = eval_setup(load_config)
    dataparser_outputs = pipeline.datamanager.train_dataset.dataparser_outputs
    cameras = dataparser_outputs.cameras

    # TODO: combine depth and color together
    depth_images = _render_trajectory(
        pipeline, cameras, rendered_output_name="depth", rendered_resolution_scaling_factor=1.0 / downscale_factor
    )
    color_images = _render_trajectory(
        pipeline,
        cameras,
        rendered_output_name="rgb",
        rendered_resolution_scaling_factor=1.0,  # NOTE: another downscale factor causes a bug
    )
    # create the point cloud
    point_cloud = []
    for i, _ in enumerate(depth_images):
        ray_bundle = cameras.generate_rays(camera_indices=i)
        points = ray_bundle.origins + ray_bundle.directions * depth_images[i]
        point_cloud.append(points)
    point_cloud = torch.stack(point_cloud)
    point_cloud = point_cloud.reshape(-1, 3)
    color_cloud = torch.from_numpy(np.stack(color_images)).reshape(-1, 3)

    # TODO: clip the point cloud based on the AABB
    # currently this clips between -1 and 1
    valid = (
        (point_cloud[:, 0] > -1.0)
        & (point_cloud[:, 0] < 1.0)
        & (point_cloud[:, 1] > -1.0)
        & (point_cloud[:, 1] < 1.0)
        & (point_cloud[:, 2] > -1.0)
        & (point_cloud[:, 2] < 1.0)
    )
    point_cloud = point_cloud[valid]
    print(point_cloud.shape)

    pc = point_cloud.float().numpy()  # float
    cc = (color_cloud[valid] * 255.0).int().numpy().astype("uint8")  # uint8
    d = {"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "red": cc[:, 0], "green": cc[:, 1], "blue": cc[:, 2]}
    cloud = PyntCloud(pd.DataFrame(data=d))

    cloud.to_file("point_cloud_chair_01.ply")
    print("here")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
