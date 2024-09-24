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

#!/usr/bin/env python
"""
ns-large-export
"""

from __future__ import annotations

import shutil
import typing
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import tyro
from typing_extensions import Literal

from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path = None
    """Path to the config YAML file."""
    output_dir: Path = None
    """Path to the output directory."""


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"
    """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
    spherical harmonics coefficients."""

    def write_ply(
        self,
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        if self.output_dir is not None:
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)

        filename = self.output_dir / (filename + ".ply")

        # Ensure count matches the length of all tensors
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")

            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

    def export(self):
        """
        Main function for exporting a 3D Gaussian Splatting model

        :return: An ordered dictionary mapping property names to numpy arrays of float or uint8 values
        """
        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        map_to_tensors = OrderedDict()

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if self.ply_color_mode == "rgb":
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                colors = (colors * 255).astype(np.uint8)
                map_to_tensors["red"] = colors[:, 0]
                map_to_tensors["green"] = colors[:, 1]
                map_to_tensors["blue"] = colors[:, 2]
            elif self.ply_color_mode == "sh_coeffs":
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

            if model.config.sh_degree > 0:
                if self.ply_color_mode == "rgb":
                    CONSOLE.print(
                        "Warning: model has higher level of spherical harmonics, ignoring them and only export rgb."
                    )
                elif self.ply_color_mode == "sh_coeffs":
                    # transpose(1, 2) was needed to match the sh order in Inria version
                    shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                    shs_rest = shs_rest.reshape((n, -1))
                    for i in range(shs_rest.shape[-1]):
                        map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

            if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
                crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                assert crop_obb is not None
                mask = crop_obb.within(torch.from_numpy(positions)).numpy()
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][mask]

                n = map_to_tensors["x"].shape[0]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]

        # Return map_to_tensors
        return map_to_tensors


def load_cells(filepath: Path):
    """
    Loads cell boundaries from the cell_boundaries.txt file generated during ns-large-train.

    :param filepath: Path to the cell_boundaries.txt file.
    :return: dict
        A dictionary where the keys are (row, col) tuples and the values are dictionaries with 'min' and 'max' keys
        containing the boundary points.
    """
    cell_boundaries = {}

    with open(filepath, 'r') as file:
        while True:
            line = file.readline().strip()
            if not line:
                break

            # Parse the row and column
            row, col = map(int, line.split())

            # Parse the boundary points (min and max)
            min_line = file.readline().strip()
            max_line = file.readline().strip()

            min_x, min_z = map(float, min_line.split())
            max_x, max_z = map(float, max_line.split())

            # Store the boundary information in the dictionary
            cell_boundaries[(row, col)] = {
                'min': [min_x, min_z],
                'max': [max_x, max_z]
            }

    return cell_boundaries


class Merger:
    def __init__(self, cells: dict, raw_splats: dict):
        """
        Initializes the Merger class with the cell boundaries dictionary and the raw GS results of the trained split scenes

        :param cells: Dictionary containing cell boundaries resulted from splitting
        :param raw_splats: Dictionary containing GS results of each cell
        """
        self.cells = cells
        self.raw_splats = raw_splats

    def cull_gaussians(self):
        """
        Culls Gaussian primitives positioned outside the 2D ground plane bounding box of the cells

        :return: A dictionary of "splat maps" for each cell containing the remaining valid Gaussians
        """
        new_splats = {}
        for pos, cell in self.cells.items():
            splats_map = self.raw_splats[pos]
            count = splats_map["x"].shape[0]

            # Find remaining Gaussians that are placed within the cell boundaries
            remaining = np.zeros(count, dtype=bool)
            for index in range(count):
                # Note that the Y and Z coordinates are interchanged from the COLMAP world coordinate system to nerfstudio's
                is_x_in_boundary = cell["min"][0] <= splats_map["x"][index] <= cell["max"][0]
                is_y_in_boundary = cell["min"][1] <= splats_map["y"][index] <= cell["max"][1]
                if is_x_in_boundary and is_y_in_boundary:
                    remaining[index] = True

            CONSOLE.log(f"Cell {pos}: {count} splats before culling, {np.sum(remaining)} splats after culling")

            # Cull Gaussians that are not in the cell bounding box
            for key in splats_map.keys():
                splats_map[key] = splats_map[key][remaining]

            new_splats[pos] = splats_map

        # Clear memory
        del self.raw_splats

        return new_splats

    def merge_splats(self):
        """
        Main function for merging the split 3DGS results of the full scene

        :return: An ordered dictionary containing all the Gaussians composing the complete scene
        """
        merged_splats = OrderedDict()
        split_splats = self.cull_gaussians()

        # Merge all the split splats together
        CONSOLE.log("Merging all remaining Gaussians...")
        for pos in self.cells.keys():
            splats_map = split_splats[pos]
            for key, value in splats_map.items():
                if key not in merged_splats:
                    merged_splats[key] = value.copy()  # Initialize with the first array
                else:
                    merged_splats[key] = np.concatenate((merged_splats[key], value), axis=0)

        return merged_splats


def copy_directory(src: Path, dst: Path):
    """
    Copies the contents of the source directory to the destination directory.

    Args:
        src (Path): The source directory path.
        dst (Path): The destination directory path.
    """
    try:
        shutil.copytree(src, dst)
        print(f"Directory copied from {src} to {dst}.")
    except FileExistsError:
        print(f"Error: Destination directory '{dst}' already exists.")
    except FileNotFoundError:
        print(f"Error: Source directory '{src}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main(data_dir: Path, train_dir: Path, output_dir: Path = Path("outputs"), project_name: str = "large-nerfstudio-project"):
    """
    Main function to merge the split 3D Gaussian Splatting results of a large scale scene
    and export the complete large scale scene under PLY file format.

    Args:
        data_dir (Path): Path to the data directory containing COLMAP outputs and images.
        train_dir (Path): Path to the project training outputs directory containing each cell's config and model files
        output_dir (Path, optional): Directory to store output configurations and results.
                                     Defaults to "outputs".
        project_name (str): Name of the NeRF project. Defaults to "large-nerfstudio-project".
    """
    # Check if the data path is valid
    if not (data_dir.is_dir() and train_dir.is_dir()):
        print("Error: The provided data and/or train path are not valid directories.")
        return

    # Get cell boundaries
    cells_path = data_dir / "exports" / "cell_boundaries.txt"
    cells = load_cells(cells_path)

    # Export split GS results and load into raw splats dict
    raw_splats = {}
    for row, col in cells.keys():
        # Copy the raw images of the full scene to the cell's COLMAP directory
        raw_images_path = data_dir / "images"
        cell_images_path = data_dir / "exports" / f"{row}-{col}" / "images"
        copy_directory(raw_images_path, cell_images_path)

        # Extract config path
        config_path = (next((train_dir / f"{row}-{col}" / "splatfacto").iterdir())) / "config.yml"

        # Export split scene and add it to raw splats
        exporter = ExportGaussianSplat(load_config=config_path)
        splat = exporter.export()
        raw_splats[(row, col)] = splat

        # Clean up by removing the copied images directory after training
        shutil.rmtree(cell_images_path)

    # Initialize Merger class
    merger = Merger(cells, raw_splats)

    # Merge GS results into one full scene
    merged_splats = merger.merge_splats()

    # Clear memory
    del raw_splats

    # Export full scene
    CONSOLE.log("Exporting full scene to PLY format...")
    exporter = ExportGaussianSplat(output_dir=output_dir)
    print(merged_splats["x"].shape[0])
    exporter.write_ply(count=merged_splats["x"].shape[0], map_to_tensors=merged_splats, filename=project_name)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
