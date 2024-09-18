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
ns-large-train
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import shutil
import shlex

import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro

from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class Splitter:
    def __init__(self, data_dir: Path, rows: int, cols: int):
        """
        Initializes the Splitter with the specified data directory and grid dimensions.

        Args:
            data_dir (Path): Path to the data directory containing COLMAP outputs and images.
            rows (int): Number of rows to divide the scene into.
            cols (int): Number of columns to divide the scene into.
        """
        self.rows = rows
        self.cols = cols
        self.data_dir = data_dir
        # Initialize a 2D grid to hold cell boundary information
        self.cells = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        # Initialize placeholders for COLMAP data
        self.cameras = None
        self.images = None
        self.points = None

    def load_scene(self):
        """
        Loads COLMAP scene data (cameras, images, points) from the specified data directory.
        It checks for both text and binary versions of COLMAP output files.
        """
        # Path to COLMAP sparse reconstruction directory
        colmap_path = self.data_dir / "colmap" / "sparse" / "0"

        # Load cameras from COLMAP output
        if (colmap_path / "cameras.txt").exists():
            self.cameras = colmap_utils.read_cameras_text(colmap_path / "cameras.txt")
        elif (colmap_path / "cameras.bin").exists():
            self.cameras = colmap_utils.read_cameras_binary(colmap_path / "cameras.bin")

        # Load images from COLMAP output
        if (colmap_path / "images.txt").exists():
            self.images = colmap_utils.read_images_text(colmap_path / "images.txt")
        elif (colmap_path / "images.bin").exists():
            self.images = colmap_utils.read_images_binary(colmap_path / "images.bin")

        # Load 3D points from COLMAP output
        if (colmap_path / "points3D.txt").exists():
            self.points = colmap_utils.read_points3D_text(colmap_path / "points3D.txt")
        elif (colmap_path / "points3D.bin").exists():
            self.points = colmap_utils.read_points3D_binary(colmap_path / "points3D.bin")

    def create_cells(self):
        """
        Divides the scene into a grid of cells based on the X and Z coordinates of the 3D points.
        It calculates the bounding box of all points and determines the size of each cell.
        """
        # Extract the X and Z coordinates of all 3D points and convert to a NumPy array
        xyz = np.array([point.xyz for point in self.points.values()])

        # Determine the minimum and maximum X and Z values to define the scene's bounding box
        min_x = np.min(xyz[:, 0])  # Minimum X value
        min_z = np.min(xyz[:, 2])  # Minimum Z value
        max_x = np.max(xyz[:, 0])  # Maximum X value
        max_z = np.max(xyz[:, 2])  # Maximum Z value

        # Define the bounding box with min and max coordinates
        bounding_box = {
            "min": [min_x, min_z],
            "max": [max_x, max_z]
        }

        # Calculate the size of each cell based on the bounding box and grid dimensions
        col_size = (bounding_box['max'][0] - bounding_box['min'][0]) / self.cols
        row_size = (bounding_box['max'][1] - bounding_box['min'][1]) / self.rows

        # Populate the cells grid with min and max coordinates for each cell
        for row in range(self.rows):
            for col in range(self.cols):
                self.cells[row][col] = {
                    "min": [
                        bounding_box['min'][0] + col_size * col,
                        bounding_box['max'][1] - row_size * (row + 1)
                    ],
                    "max": [
                        bounding_box['min'][0] + col_size * (col + 1),
                        bounding_box['max'][1] - row_size * row
                    ]
                }

    def export_colmap_scene(self, images, points, scene_name: str):
        """
        Exports the COLMAP scene data for a specific cell by writing the cameras, images, and points
        to binary COLMAP files.

        Args:
            images (dict): Dictionary of images belonging to the cell.
            points (dict): Dictionary of 3D points belonging to the cell.
            scene_name (str): Name of the scene/cell for directory structure.
        """
        # Define the path where the COLMAP data for the cell will be saved
        colmap_path = self.data_dir / "exports" / scene_name / "colmap" / "sparse" / "0"
        # Create the directory structure if it doesn't exist
        Path(colmap_path).mkdir(parents=True, exist_ok=True)

        # Write cameras, images, and points to COLMAP binary files
        colmap_utils.write_cameras_binary(self.cameras, colmap_path / "cameras.bin")
        colmap_utils.write_images_binary(images, colmap_path / "images.bin")
        colmap_utils.write_points3D_binary(points, colmap_path / "points3D.bin")

    def split_scene(self):
        """
        Splits the entire scene into valid cells based on the distribution of 3D points and images.
        It exports each valid cell's COLMAP data and records the boundaries of each cell.

        Returns:
            tuple: A tuple containing a list of valid cell coordinates and a dictionary mapping each cell
                   to its list of image names.
        """
        self.load_scene()  # Load COLMAP data (cameras, images, points)
        self.create_cells()  # Divide the scene into cells

        # Initialize dictionaries to hold points and images for each cell
        cell_points = {(r, c): {} for r in range(self.rows) for c in range(self.cols)}
        cell_images = {(r, c): {} for r in range(self.rows) for c in range(self.cols)}
        # Dictionary to track the frequency of each image in each cell
        cell_image_freqs = {(r, c): defaultdict(int) for r in range(self.rows) for c in range(self.cols)}

        # Iterate over each 3D point to assign it to the appropriate cell
        for point_id, point in self.points.items():
            # Extract the X and Z coordinates of the point
            x, z = point.xyz[0], point.xyz[2]

            # Find the cell that the point belongs to
            for row in range(self.rows):
                found_cell = False
                for col in range(self.cols):
                    cell = self.cells[row][col]
                    # Check if the point's coordinates fall within the current cell's boundaries
                    if cell['min'][0] <= x < cell['max'][0] and cell['min'][1] <= z < cell['max'][1]:
                        # Assign the point to the cell
                        cell_points[(row, col)][point_id] = point

                        # Assign images that see this point to the cell
                        for pt_image_id in point.image_ids:
                            if pt_image_id in self.images.keys():
                                cell_images[(row, col)][pt_image_id] = self.images[pt_image_id]
                                # Increment the frequency count for the image in this cell
                                cell_image_freqs[(row, col)][pt_image_id] += 1

                        # Mark that the cell has been found for this point
                        found_cell = True
                        break  # No need to check other cells

                if found_cell:
                    break  # Move to the next point once the cell is found

        # Ensure the exports directory exists to store cell data
        exports_dir = self.data_dir / "exports"
        Path(exports_dir).mkdir(parents=True, exist_ok=True)

        valid_cells = []  # List to store coordinates of valid cells
        image_paths_by_cells = {}  # Dictionary to map each valid cell to its image paths

        # Open the cell boundaries file to record the boundaries of each valid cell
        with open(exports_dir / "cell_boundaries.txt", 'w') as boundary_file:
            # Iterate over each cell to determine its validity
            for row in range(self.rows):
                for col in range(self.cols):
                    cell_num_images = len(cell_images[(row, col)])  # Number of images in the cell
                    cell_num_points = len(cell_points[(row, col)])  # Number of points in the cell

                    # Prune images that are insignificant (appear less frequently relative to points)
                    images_to_remove = [
                        image_id for image_id, freq in cell_image_freqs[(row, col)].items()
                        if freq / cell_num_points < 0.001
                    ]
                    for image_id in images_to_remove:
                        del cell_images[(row, col)][image_id]

                    # Skip cells that are empty or irrelevant based on image-to-point ratio and minimum points
                    if not cell_images[(row, col)]:
                        continue
                    elif cell_num_images / cell_num_points > 0.5 or cell_num_points < 10:
                        continue

                    # Export the COLMAP scene data for the valid cell
                    self.export_colmap_scene(
                        cell_images[(row, col)],
                        cell_points[(row, col)],
                        f"{row}-{col}"
                    )

                    # Record the cell's coordinates as valid
                    valid_cells.append((row, col))

                    # Record the list of image names for the valid cell
                    image_paths_by_cells[(row, col)] = [
                        image.name for image in cell_images[(row, col)].values()
                    ]

                    # Retrieve the cell's boundary points
                    cell = self.cells[row][col]
                    min_point = cell['min']
                    max_point = cell['max']

                    # Write the cell's boundaries to the boundaries file
                    boundary_file.write(f"{row} {col}\n")
                    boundary_file.write(f"{min_point[0]} {min_point[1]}\n")
                    boundary_file.write(f"{max_point[0]} {max_point[1]}\n")

        # Return the list of valid cells and their corresponding image paths
        return valid_cells, image_paths_by_cells


def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    trainer.train()


def _distributed_worker(
        local_rank: int,
        main_func: Callable,
        world_size: int,
        num_devices_per_machine: int,
        machine_rank: int,
        dist_url: str,
        config: TrainerConfig,
        timeout: timedelta = DEFAULT_TIMEOUT,
        device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and handles the
       training process on multiple processes.

    Args:
        local_rank: Current rank of process.
        main_func: Function that will be called by the distributed workers.
        world_size: Total number of gpus available.
        num_devices_per_machine: Number of GPUs per machine.
        machine_rank: Rank of this machine.
        dist_url: URL to connect to for distributed jobs, including protocol
            E.g., "tcp://127.0.0.1:8686".
            It can be set to "auto" to automatically select a free port on localhost.
        config: TrainerConfig specifying training regimen.
        timeout: Timeout of the distributed workers.

    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank

    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_devices_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comms.LOCAL_PROCESS_GROUP = pg

    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank)
    comms.synchronize()
    dist.destroy_process_group()
    return output


def launch(
        main_func: Callable,
        num_devices_per_machine: int,
        num_machines: int = 1,
        machine_rank: int = 0,
        dist_url: str = "auto",
        config: Optional[TrainerConfig] = None,
        timeout: timedelta = DEFAULT_TIMEOUT,
        device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one process
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(method: str, data: Path, output_dir: Path = "outputs", project_name: str = "nerfstudio-project", rows: int = 16, cols: int = 16) -> None:
    """Main function."""
    # Check if the data path is valid
    if not ((data / "colmap" / "sparse" / "0").is_dir() and (data / "images").is_dir()):
        print(f"Error: The provided data path '{data}' is not a valid directory.")
        return

    print(f"Method name: {method}")
    print(f"Valid data path: {data}")
    print(f"Rows: {rows}, Cols: {cols}")

    splitter = Splitter(data, rows, cols)
    valid_cells, image_paths_by_cell = splitter.split_scene()

    if valid_cells is not None:
        print("Splitting was done successfully")

    for row, col in valid_cells:
        # Define scene directory path
        scene_dir = splitter.data_dir / "exports" / f"{row}-{col}"

        # Create the images folder in the scene directory
        images_dir = scene_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Get the list of image paths for the current cell
        image_paths = image_paths_by_cell[(row, col)]

        # Copy only the images listed in image_paths to the new scene directory
        for image_name in image_paths:
            src_image_path = splitter.data_dir / "images" / image_name
            dest_image_path = images_dir / image_name
            shutil.copy2(src_image_path, dest_image_path)

        # Generate config
        if method == "splatfacto":
            arguments = f"splatfacto --output-dir={output_dir} --experiment-name={row}-{col} --project-name={project_name} --pipeline.model.cull-scale-thresh=1000 --viewer.quit-on-train-completion=True colmap --data={scene_dir} --center-method=none --orientation-method=none --auto-scale-poses=False"
        else:
            arguments = f"{method} --output-dir={output_dir} --experiment-name={row}-{col} --project-name={project_name} --viewer.quit-on-train-completion=True colmap --data={scene_dir} --center-method=none --orientation-method=none --auto-scale-poses=False"

        args_list = shlex.split(arguments)  # Split the arguments string into a list of arguments

        config = tyro.cli(
            AnnotatedBaseConfigUnion,
            args=args_list,
            description=convert_markup_to_ansi(__doc__),
        )

        config.set_timestamp()

        # Print and save config
        config.print_to_terminal()
        config.save_config()

        launch(
            main_func=train_loop,
            num_devices_per_machine=config.machine.num_devices,
            device_type=config.machine.device_type,
            num_machines=config.machine.num_machines,
            machine_rank=config.machine.machine_rank,
            dist_url=config.machine.dist_url,
            config=config,
        )

        shutil.rmtree(images_dir)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
