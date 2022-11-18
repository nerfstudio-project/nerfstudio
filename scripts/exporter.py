"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import tyro
from rich.console import Console
from typing_extensions import Annotated, Literal

from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import (
    _render_trajectory,
    generate_point_cloud,
    get_mesh_from_filename,
)
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    estimate_normals: bool = False
    """Estimate normals for the point cloud."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 65536
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            estimate_normals=self.estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud")
        o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale factor for the images."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [256, 256, 256])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
        )

        mesh = get_mesh_from_filename(str(self.output_dir / "tsdf_mesh.ply"))

        # possibly texture with NeRF
        if self.texture_method == "nerf":
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(mesh, pipeline, self.px_per_uv_triangle, self.output_dir)


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    downscale_factor: int = 2
    """Downscale factor for the images."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [256, 256, 256])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        dataparser_outputs = pipeline.datamanager.train_dataset.dataparser_outputs
        cameras = dataparser_outputs.cameras

        color_images, depth_images = _render_trajectory(
            pipeline,
            cameras,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
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

        # TODO: optionally use normals from the network
        # with CONSOLE.status("Estimating Normals", spinner="arrow") as status:
        CONSOLE.print("Estimating Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Normals")

        # with CONSOLE.status("Cleaning Normals", spinner="arrow"):
        #     pcd.orient_normals_consistent_tangent_plane(100)

        # with CONSOLE.status("Saving Point Cloud", spinner="pipe") as status:
        CONSOLE.print("Saving Point Cloud")
        o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
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
        o3d.io.write_triangle_mesh(str(self.output_dir / "mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """
    NOT YET IMPLEMENTED
    Export a mesh using marching cubes.
    """

    def main(self) -> None:
        """Export mesh"""
        raise NotImplementedError("Marching cubes not implemented yet.")


Commands = Union[
    Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
    Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
    Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
    Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[Commands]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
