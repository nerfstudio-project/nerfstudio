"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import open3d as o3d
import tyro
from rich.console import Console
from typing_extensions import Annotated

from nerfstudio.model_components.exporters import generate_point_cloud
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
    """Class to export meshes with the poisson algorithm."""

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
    """Class to export meshes with the poisson algorithm."""

    def main(self) -> None:
        """Export mesh"""
        raise NotImplementedError("TSDF mesh reconstruction not implemented yet.")


@dataclass
class ExportPoissonMesh(Exporter):
    """Exports a mesh using poisson mesh reconstruction."""

    def main(self) -> None:
        """Export mesh"""
        raise NotImplementedError("Poisson mesh reconstruction not implemented yet.")


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Exports a mesh using marching cubes."""

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
