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

# pylint: disable=too-few-public-methods

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import tyro
from rich.console import Console
from typing_extensions import Annotated

from nerfstudio.model_components.meshers import (  # MesherConfig,
    MarchingCubesMesherConfig,
    PoissonMesherConfig,
    TSDFMesherConfig,
)
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class ExportMesh:
    """Export the mesh from a YML config to a folder."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output directory.
    output_dir: Path
    # Mesher config to use.
    mesher_config: Any = TSDFMesherConfig()

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(self.load_config)
        mesher = self.mesher_config.setup()
        mesher.export_mesh(pipeline, output_dir=self.output_dir)


class ExportTSDFMesh(ExportMesh):
    """Export TSDF mesh config."""

    mesher_config = TSDFMesherConfig()


class ExportPoissonMesh(ExportMesh):
    """Export Poisson mesh config."""

    mesher_config = PoissonMesherConfig()


class ExportMarchingCubesMesh(ExportMesh):
    """Export Marching Cubes mesh config."""

    mesher_config = MarchingCubesMesherConfig()


Commands = Union[
    Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
    Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
    Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
