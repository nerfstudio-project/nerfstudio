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
from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
import tyro
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
    rgb_output_name: str,
    depth_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Helper function to create a video of a trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.

    Returns:
        List of rgb images, list of depth images.
    """
    images = []
    depths = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
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
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            images.append(outputs[rgb_output_name].cpu().numpy())
            depths.append(outputs[depth_output_name].cpu().numpy())
    return images, depths


def main(
    load_config: Path,
    output_dir: Path,
    downscale_factor: int = 4,
    depth_output_name: str = "depth",
    rgb_output_name: str = "rgb",
):
    """Main function.

    Args:
        load_config: Path to config YAML file.
        output_dir: Path to output directory.
        downscale_factor: Downscale factor for the images.
        depth_output_name: Name of the depth output.
        rgb_output_name: Name of the RGB output.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    _, pipeline, _ = eval_setup(load_config)
    dataparser_outputs = pipeline.datamanager.train_dataset.dataparser_outputs
    cameras = dataparser_outputs.cameras

    color_images, depth_images = _render_trajectory(
        pipeline,
        cameras,
        rgb_output_name=rgb_output_name,
        depth_output_name=depth_output_name,
        rendered_resolution_scaling_factor=1.0 / downscale_factor,
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.float().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color_cloud[valid].float().numpy())

    # with CONSOLE.status("Cleaning Point Cloud", spinner="circleHalves") as status:
    CONSOLE.print("Cleaning Point Cloud")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    print("\033[A\033[A")
    CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

    # with CONSOLE.status("Estimating Normals", spinner="arrow") as status:
    CONSOLE.print("Estimating Normals")
    pcd.estimate_normals()
    print("\033[A\033[A")
    CONSOLE.print("[bold green]:white_check_mark: Estimating Normals")

    # with CONSOLE.status("Cleaning Normals", spinner="arrow"):
    #     pcd.orient_normals_consistent_tangent_plane(100)

    # with CONSOLE.status("Saving Point Cloud", spinner="pipe") as status:
    CONSOLE.print("Saving Point Cloud")
    o3d.io.write_point_cloud(str(output_dir / "point_cloud.ply"), pcd)
    print("\033[A\033[A")
    CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

    # with CONSOLE.status("Computing Mesh", spinner="aesthetic") as status:
    CONSOLE.print("Computing Mesh")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print("\033[A\033[A")
    CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

    # with CONSOLE.status("Saving Mesh", spinner="pipe") as status:
    CONSOLE.print("Saving Mesh")
    o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh)
    print("\033[A\033[A")
    CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
