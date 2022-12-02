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
Export utils such as structs, point cloud generation, and rendering code.
"""

# pylint: disable=no-member

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import pymeshlab
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


@dataclass
class Mesh:
    """Class for a mesh."""

    vertices: TensorType["num_verts", 3]
    """Vertices of the mesh."""
    faces: TensorType["num_faces", 3]
    """Faces of the mesh."""
    normals: TensorType["num_verts", 3]
    """Normals of the mesh."""
    colors: Optional[TensorType["num_verts", 3]] = None
    """Colors of the mesh."""


def get_mesh_from_pymeshlab_mesh(mesh: pymeshlab.Mesh) -> Mesh:
    """Get a Mesh from a pymeshlab mesh.
    See https://pymeshlab.readthedocs.io/en/0.1.5/classes/mesh.html for details.
    """
    return Mesh(
        vertices=torch.from_numpy(mesh.vertex_matrix()).float(),
        faces=torch.from_numpy(mesh.face_matrix()).long(),
        normals=torch.from_numpy(np.copy(mesh.vertex_normal_matrix())).float(),
        colors=torch.from_numpy(mesh.vertex_color_matrix()).float(),
    )


def get_mesh_from_filename(filename: str, target_num_faces: Optional[int] = None) -> Mesh:
    """Get a Mesh from a filename."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    if target_num_faces is not None:
        CONSOLE.print("Running meshing decimation with quadric edge collapse")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces)
    mesh = ms.current_mesh()
    return get_mesh_from_pymeshlab_mesh(mesh)


def generate_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    # pylint: disable=too-many-statements

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    points = []
    rgbs = []
    normals = []
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                outputs = pipeline.model(ray_bundle)
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
            rgb = outputs[rgb_output_name]
            depth = outputs[depth_output_name]
            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
            point = ray_bundle.origins + ray_bundle.directions * depth

            if use_bounding_box:
                comp_l = torch.tensor(bounding_box_min, device=point.device)
                comp_m = torch.tensor(bounding_box_max, device=point.device)
                assert torch.all(
                    comp_l < comp_m
                ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
                mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
                point = point[mask]
                rgb = rgb[mask]
                if normal_output_name is not None:
                    normal = normal[mask]

            points.append(point)
            rgbs.append(rgb)
            if normal_output_name is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.float().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.float().cpu().numpy())

    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.float().cpu().numpy())

    return pcd


def render_trajectory(
    pipeline: Pipeline,
    cameras: Cameras,
    rgb_output_name: str,
    depth_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    disable_distortion: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Helper function to create a video of a trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        disable_distortion: Whether to disable distortion.

    Returns:
        List of rgb images, list of depth images.
    """
    images = []
    depths = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    progress = Progress(
        TextColumn(":cloud: Computing rgb and depth images :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(
                camera_indices=camera_idx, disable_distortion=disable_distortion
            ).to(pipeline.device)
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
