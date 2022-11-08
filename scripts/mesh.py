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

from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console

from nerfstudio.model_components.meshers import (
    MarchingCubesMesherConfig,
    PoissonMesherConfig,
    TSDFMesherConfig,
)
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


AnnotatedMesherUnion = tyro.conf.OmitSubcommandPrefixes[  # Omit prefixes of flags in subcommands.
    tyro.extras.subcommand_type_from_defaults(
        {
            "tsdf": TSDFMesherConfig(),
            "poisson": PoissonMesherConfig(),
            "marching-cubes": MarchingCubesMesherConfig(),
        },
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )
]
"""Union over possible mesher types, annotated with metadata for tyro. This is the
same as the vanilla union, but results in shorter subcommand names."""


@dataclass
class ExportMesh:
    """Export the mesh from a YML config to a folder."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output directory.
    output_dir: Path
    # Mesher config to use.
    mesher_config: AnnotatedMesherUnion = TSDFMesherConfig()

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(self.load_config)
        mesher = self.mesher_config.setup()
        mesher.export_mesh(pipeline, output_dir=self.output_dir)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ExportMesh).main()


if __name__ == "__main__":
    entrypoint()
