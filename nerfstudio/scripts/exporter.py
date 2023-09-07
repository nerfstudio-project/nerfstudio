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
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import (
    collect_camera_poses,
    generate_point_cloud,
    get_mesh_from_filename,
)
from nerfstudio.exporter.marching_cubes import (
    generate_mesh_with_multires_marching_cubes,
)
from nerfstudio.fields.sdf_field import SDFField
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Optional[Tuple[float, float, float]] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Optional[Tuple[float, float, float]] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, VanillaDataManager)
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, VanillaDataManager)
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # TODO: Make this work with Density Field
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert (
            self.resolution % 512 == 0
        ), f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
