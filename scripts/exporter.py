"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import mediapy as media
import numpy as np
import open3d as o3d
import torch
import trimesh
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torchtyping import TensorType
from tqdm import tqdm
from typing_extensions import Annotated

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.exporters import generate_point_cloud
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from nerfstudio.utils.tsdf import TSDF

CONSOLE = Console(width=120)


def _render_trajectory(
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
    device: str = "cuda"
    """Device to use for the TSDF operations."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        dataparser_outputs = pipeline.datamanager.train_dataset.dataparser_outputs

        # initialize the TSDF volume
        aabb = dataparser_outputs.scene_box.aabb
        if isinstance(self.resolution, int):
            volume_dims = torch.tensor([self.resolution] * 3)
        elif isinstance(self.resolution, List):
            volume_dims = torch.tensor(self.resolution)
        else:
            raise ValueError("Resolution must be an int or a list.")
        tsdf = TSDF.from_aabb(aabb, volume_dims=volume_dims)
        # move TSDF to device
        tsdf.to(self.device)

        cameras = dataparser_outputs.cameras
        # we turn off distortion when populating the TSDF
        color_images, depth_images = _render_trajectory(
            pipeline,
            cameras,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            disable_distortion=True,
        )

        # camera extrinsics and intrinsics
        c2w: TensorType["N", 3, 4] = cameras.camera_to_worlds.to(self.device)
        # make c2w homogeneous
        c2w = torch.cat([c2w, torch.zeros(c2w.shape[0], 1, 4, device=self.device)], dim=1)
        c2w[:, 3, 3] = 1
        K: TensorType["N", 3, 3] = cameras.get_intrinsics_matrices().to(self.device)
        color_images = torch.tensor(np.array(color_images), device=self.device).permute(
            0, 3, 1, 2
        )  # shape (N, 3, H, W)
        depth_images = torch.tensor(np.array(depth_images), device=self.device).permute(
            0, 3, 1, 2
        )  # shape (N, 1, H, W)

        CONSOLE.print("Integrating the TSDF")
        for i in range(0, len(c2w), self.batch_size):
            tsdf.integrate_tsdf(
                c2w[i : i + self.batch_size],
                K[i : i + self.batch_size],
                depth_images[i : i + self.batch_size],
                color_images=color_images[i : i + self.batch_size],
            )

        CONSOLE.print("Computing Mesh")
        mesh = tsdf.get_mesh()
        CONSOLE.print("Saving Mesh")
        tsdf.export_mesh(mesh, filename=str(self.output_dir / "mesh.ply"))


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
    device: str = "cuda"
    """Device to use for the TSDF operations."""
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


def edge_function(p, v0, v1):
    """https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/mesh/rasterize_meshes.html"""
    return (p[..., 0] - v0[..., 0]) * (v1[..., 1] - v0[..., 1]) - (p[..., 1] - v0[..., 1]) * (v1[..., 0] - v0[..., 0])


@dataclass
class ExportTextureMesh(Exporter):
    """
    Export a textured mesh with color computed from the NeRF.
    """

    px_per_uv_square: int = 7
    """Number of pixels per UV square."""
    input_ply_filename: Path = Path("mesh.ply")
    """PLY mesh filename to texture."""
    device: str = "cuda"
    """Device to use for torch operations."""

    def main(self) -> None:
        """Export textured mesh"""
        # pylint: disable=too-many-statements

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # load the mesh
        mesh = trimesh.load(str(self.input_ply_filename))
        vertices = torch.from_numpy(mesh.vertices)
        faces = torch.from_numpy(mesh.faces)
        vertex_normals = torch.from_numpy(np.copy(mesh.vertex_normals))
        assert len(vertices) == len(vertex_normals), "Number of vertices and vertex normals must be equal"

        num_squares = len(faces) / 2
        squares_per_side = math.ceil(math.sqrt(num_squares))
        num_pixels = squares_per_side * self.px_per_uv_square

        lr = self.px_per_uv_square / num_pixels
        px = 1.0 / num_pixels
        edge_len = (self.px_per_uv_square - 1 - 2) / num_pixels
        scalar = (self.px_per_uv_square - 2 - 2) / (self.px_per_uv_square - 1 - 2)
        # uv coords (upper left and lower right)
        uv_coords_upper_left = torch.tensor([[0, 0], [edge_len, 0], [0, edge_len]])
        # scale for bilinear interpolation reasons
        uv_coords_upper_left = (
            uv_coords_upper_left
        ) * scalar + px/2
        lower_right = [lr, lr - 3*px] # NOTE(ethan): -3*px depends on the number of pixels per square
        uv_coords_lower_right = torch.tensor(
            [
                lower_right, # lower right
                [3*px, lr - 3*px], # lower left # NOTE(ethan): -3*px depends on the number of pixels per square
                [lr, 0], # upper right
            ]
        )
        # scale for bilinear interpolation reasons
        uv_coords_lower_right = (
            uv_coords_lower_right - torch.tensor(lower_right)
        ) * scalar + torch.tensor(lower_right) - px/2
        uv_coords_square = torch.stack([uv_coords_upper_left, uv_coords_lower_right], dim=0)  # (2, 3, 2)
        uv_coords_square = uv_coords_square.reshape(1, 1, 6, 2)  # (6, 2)
        square_offsets = (
            torch.stack(
                torch.meshgrid(torch.arange(squares_per_side), torch.arange(squares_per_side), indexing="xy"), dim=-1
            )
            * lr
        )
        uv_coords_square = uv_coords_square + square_offsets.view(
            squares_per_side, squares_per_side, 1, 2
        )  # (squares_per_side, squares_per_side, 6, 2)
        texture_coordinates = uv_coords_square.view(-1, 3, 2)[: len(faces)]  # (num_faces, 3, 2)

        uv_indices = torch.stack(
            torch.meshgrid(torch.arange(num_pixels), torch.arange(num_pixels), indexing="xy"), dim=-1
        )
        linspace = torch.linspace(px / 2, 1 - px / 2, num_pixels)
        uv_coords = torch.stack(torch.meshgrid(linspace, linspace, indexing="xy"), dim=-1)
        print("uv_coords", uv_coords.shape)

        u_index = torch.div(uv_indices[..., 0], self.px_per_uv_square, rounding_mode="floor")
        v_index = torch.div(uv_indices[..., 1], self.px_per_uv_square, rounding_mode="floor")
        square_index = v_index * squares_per_side + u_index
        u_offset = uv_indices[..., 0] % self.px_per_uv_square
        v_offset = uv_indices[..., 1] % self.px_per_uv_square
        lower_right = (u_offset + v_offset) >= (self.px_per_uv_square - 2)
        triangle_index = square_index * 2 + lower_right
        # triangle_index[
        #     (u_offset + v_offset) == (self.px_per_uv_square - 1)
        # ] = 0  # we don't use the diagonals of the squares
        triangle_index = torch.clamp(triangle_index, min=0, max=len(faces) - 1)

        nearby_uv_coords = texture_coordinates[triangle_index]
        nearby_vertices = vertices[faces[triangle_index]]
        nearby_normals = vertex_normals[faces[triangle_index]]

        print("uv_coords", uv_coords.shape)  # (num_pixels, num_pixels, 2)
        print("nearby_uv_coords", nearby_uv_coords.shape)  # (num_pixels, num_pixels, 3, 2)
        print("nearby_vertices", nearby_vertices.shape)  # (num_pixels, num_pixels, 3, 3)
        print("nearby_normals", nearby_normals.shape)  # (num_pixels, num_pixels, 3, 3)

        # compute barycentric coordinates
        v0 = nearby_uv_coords[..., 0, :]  # (num_pixels, num_pixels, 2)
        v1 = nearby_uv_coords[..., 1, :]  # (num_pixels, num_pixels, 2)
        v2 = nearby_uv_coords[..., 2, :]  # (num_pixels, num_pixels, 2)
        p = uv_coords  # (num_pixels, num_pixels, 2)
        epsilon = 1e-8
        area = edge_function(v2, v0, v1) + epsilon  # 2 x face area.
        w0 = edge_function(p, v1, v2) / area
        w1 = edge_function(p, v2, v0) / area
        w2 = edge_function(p, v0, v1) / area
        u = w0
        v = w1
        w = w2
        # check which ones are outside the triangle
        outside = (u < 0) | (v < 0) | (w < 0)

        # load the NeRF and create the texture
        _, pipeline, _ = eval_setup(self.load_config)

        # color the texture image

        # compute the length of the rays we want to render
        # we can make a good guess for this by looking at the average distance between vertices on the triangle faces
        neary_vertices_dev = nearby_vertices.to(self.device)
        dist = torch.mean(torch.norm(neary_vertices_dev[..., 0, :] - neary_vertices_dev[..., 1, :], dim=-1))
        offset = dist

        origins = (
            nearby_vertices[..., 0, :] * u[..., None]
            + nearby_vertices[..., 1, :] * v[..., None]
            + nearby_vertices[..., 2, :] * w[..., None]
        ).to(self.device)
        directions = -(
            nearby_normals[..., 0, :] * u[..., None]
            + nearby_normals[..., 1, :] * v[..., None]
            + nearby_normals[..., 2, :] * w[..., None]
        ).to(self.device)
        # normalize the direction vector to make it a unit vector
        directions = torch.nn.functional.normalize(directions, dim=-1)

        origins = origins - offset * directions
        pixel_area = torch.ones_like(origins[..., 0:1])
        camera_indices = torch.zeros_like(origins[..., 0:1])
        nears = torch.zeros_like(origins[..., 0:1])
        fars = torch.ones_like(origins[..., 0:1]) * offset * 2.0
        camera_ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            nears=nears,
            fars=fars,
        )

        CONSOLE.print("Rendering texture...")
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        # save the texture image
        texture_image = outputs["rgb"].cpu().numpy()
        # texture_image[outside] = 0.0
        media.write_image(str(self.output_dir / "material_0.png"), texture_image)

        # create the .mtl file
        lines_mtl = [
            "# Generated with nerfstudio",
            "newmtl material_0",
            "Ka 0.40000000 0.40000000 0.40000000",
            "Kd 0.40000000 0.40000000 0.40000000",
            "Ks 0.40000000 0.40000000 0.40000000",
            "Ns 1.00000000",
            "map_Kd material_0.png",
        ]
        lines_mtl = [line + "\n" for line in lines_mtl]
        file_mtl = open(  # pylint: disable=consider-using-with
            self.output_dir / "material_0.mtl", "w", encoding="utf-8"
        )
        file_mtl.writelines(lines_mtl)
        file_mtl.close()

        # create the .obj file
        lines_obj = ["# Generated with nerfstudio", "mtllib material_0.mtl", "usemtl material_0"]
        lines_obj = [line + "\n" for line in lines_obj]
        file_obj = open(self.output_dir / "mesh.obj", "w", encoding="utf-8")  # pylint: disable=consider-using-with
        file_obj.writelines(lines_obj)

        # write the geometric vertices
        for i in tqdm(range(len(vertices))):
            vertex = vertices[i]
            line = f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
            file_obj.write(line)

        # write the texture coordinates
        for i in tqdm(range(len(faces))):
            for uv in texture_coordinates[i]:
                line = f"vt {uv[0]} {1.0 - uv[1]}\n"
                file_obj.write(line)

        # write the vertex normals
        for i in tqdm(range(len(vertex_normals))):
            normal = vertex_normals[i]
            line = f"vn {normal[0]} {normal[1]} {normal[2]}\n"
            file_obj.write(line)

        # write the faces
        for i in tqdm(range(len(faces))):
            face = faces[i]
            v1 = face[0] + 1
            v2 = face[1] + 1
            v3 = face[2] + 1
            vt1 = i * 3 + 1
            vt2 = i * 3 + 2
            vt3 = i * 3 + 3
            vn1 = v1
            vn2 = v2
            vn3 = v3
            line = f"f {v1}/{vt1}/{vn1} {v2}/{vt2}/{vn2} {v3}/{vt3}/{vn3}\n"
            file_obj.write(line)

        file_obj.close()


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
    Annotated[ExportTextureMesh, tyro.conf.subcommand(name="texture")],
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
