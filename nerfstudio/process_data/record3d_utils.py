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

"""Helper functions for processing record3d data."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import io


def record3d_to_json(
    images_paths: List[Path],
    metadata_path: Path,
    output_dir: Path,
    indices: np.ndarray,
    ply_dirname: Optional[Path],
    voxel_size: Optional[float],
) -> int:
    """Converts Record3D's metadata and image paths to a JSON file.

    Args:
        images_paths: list of image paths.
        metadata_path: Path to the Record3D metadata JSON file.
        output_dir: Path to the output directory.
        indices: Indices to sample the metadata_path. Should be the same length as images_paths.
        ply_dirname: Path to the directory of exported ply files.

    Returns:
        The number of registered images.
    """

    assert len(images_paths) == len(indices)

    metadata_dict = io.load_from_json(metadata_path)

    poses_data = np.array(metadata_dict["poses"])  # (N, 3, 4)
    # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    camera_to_worlds = np.concatenate(
        [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
        axis=-1,
    ).astype(np.float32)
    camera_to_worlds = camera_to_worlds[indices]

    homogeneous_coord = np.zeros_like(camera_to_worlds[..., :1, :])
    homogeneous_coord[..., :, 3] = 1
    camera_to_worlds = np.concatenate([camera_to_worlds, homogeneous_coord], -2)

    frames = []
    for i, im_path in enumerate(images_paths):
        c2w = camera_to_worlds[i]
        frame = {
            "file_path": im_path.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    # Camera intrinsics
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    focal_length = K[0, 0]

    H = metadata_dict["h"]
    W = metadata_dict["w"]

    # TODO(akristoffersen): The metadata dict comes with principle points,
    # but caused errors in image coord indexing. Should update once that is fixed.
    cx, cy = W / 2, H / 2

    out = {
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": cx,
        "cy": cy,
        "w": W,
        "h": H,
        "camera_model": CAMERA_MODELS["perspective"].name,
    }

    out["frames"] = frames

    # If .ply directory exists add the sparse point cloud for gsplat point initialization
    if ply_dirname is not None:
        assert ply_dirname.exists(), f"Directory not found: {ply_dirname}"
        assert ply_dirname.is_dir(), f"Path given is not a directory: {ply_dirname}"

        # Create sparce point cloud
        pcd = o3d.geometry.PointCloud()
        for ply_filename in ply_dirname.iterdir():
            temp_pcd = o3d.io.read_point_cloud(str(ply_filename))
            pcd += temp_pcd.voxel_down_sample(voxel_size=voxel_size)

        # Save point cloud
        points3D = np.asarray(pcd.points)
        pcd.points = o3d.utility.Vector3dVector(points3D)
        o3d.io.write_point_cloud(str(output_dir / "sparse_pc.ply"), pcd, write_ascii=True)
        out["ply_file_path"] = "sparse_pc.ply"

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)
