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
Script to texture an existing mesh file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro

from nerfstudio.exporter import texture_utils
from nerfstudio.exporter.exporter_utils import get_mesh_from_filename
from nerfstudio.utils.eval_utils import eval_setup


@dataclass
class TextureMesh:
    """
    Export a textured mesh with color computed from the NeRF.
    """

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""
    input_mesh_filename: Path
    """Mesh filename to texture."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV square."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export textured mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # load the Mesh
        mesh = get_mesh_from_filename(str(self.input_mesh_filename), target_num_faces=self.target_num_faces)

        # load the Pipeline
        _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")

        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        texture_utils.export_textured_mesh(
            mesh=mesh,
            pipeline=pipeline,
            output_dir=self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[TextureMesh]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(TextureMesh)  # noqa
