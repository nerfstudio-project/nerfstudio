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

"""Helper utils for processing polycam data into the nerfstudio format."""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from nerfstudio.process_data import process_data_utils
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import io
from nerfstudio.utils.rich_utils import CONSOLE


def polycam_to_json(
    image_filenames: List[Path],
    depth_filenames: List[Path],
    glb_filename: Optional[Path],
    cameras_dir: Path,
    output_dir: Path,
    min_blur_score: float = 0.0,
    crop_border_pixels: int = 0,
) -> List[str]:
    """Convert Polycam data into a nerfstudio dataset.

    Args:
        image_filenames: List of paths to the original images.
        depth_filenames: List of paths to the original depth maps.
        cameras_dir: Path to the polycam cameras directory.
        output_dir: Path to the output directory.
        min_blur_score: Minimum blur score to use an image. Images below this value will be skipped.
        crop_border_pixels: Number of pixels to crop from each border of the image.

    Returns:
        Summary of the conversion.
    """
    use_depth = len(image_filenames) == len(depth_filenames)
    data = {}
    data["camera_model"] = CAMERA_MODELS["perspective"].value
    # Needs to be a string for camera_utils.auto_orient_and_center_poses
    data["orientation_override"] = "none"

    frames = []
    skipped_frames = 0
    for i, image_filename in enumerate(image_filenames):
        json_filename = cameras_dir / f"{image_filename.stem}.json"
        frame_json = io.load_from_json(json_filename)
        if "blur_score" in frame_json and frame_json["blur_score"] < min_blur_score:
            skipped_frames += 1
            continue
        frame = {}
        frame["fl_x"] = frame_json["fx"]
        frame["fl_y"] = frame_json["fy"]
        frame["cx"] = frame_json["cx"] - crop_border_pixels
        frame["cy"] = frame_json["cy"] - crop_border_pixels
        frame["w"] = frame_json["width"] - crop_border_pixels * 2
        frame["h"] = frame_json["height"] - crop_border_pixels * 2
        frame["file_path"] = f"./images/frame_{i+1:05d}{image_filename.suffix}"
        if use_depth:
            frame["depth_file_path"] = f"./depth/frame_{i+1:05d}{depth_filenames[i].suffix}"
        # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
        frame["transform_matrix"] = [
            [frame_json["t_20"], frame_json["t_21"], frame_json["t_22"], frame_json["t_23"]],
            [frame_json["t_00"], frame_json["t_01"], frame_json["t_02"], frame_json["t_03"]],
            [frame_json["t_10"], frame_json["t_11"], frame_json["t_12"], frame_json["t_13"]],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append(frame)
    data["frames"] = frames

    if glb_filename is not None:
        # If the .glb is populated, use it to save a pointcloud for splatfacto init
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(glb_filename), enable_post_processing=True)
        textures = np.asarray(mesh.textures[0])  # 2D images of color
        vert_points = np.asarray(mesh.vertices)  # 3D positions of verts
        tri_ids = np.asarray(mesh.triangles)  # indices of the vertices
        points = vert_points[tri_ids.flatten()]  # get the 3D positions of the vertices
        uvs = np.asarray(mesh.triangle_uvs)  # get the uv coords of the vertices
        # convert uv coord to texture integer index
        tex_ids = (uvs[:, 1] * textures.shape[0]).astype(int), (uvs[:, 0] * textures.shape[1]).astype(int)
        colors = textures[tex_ids[0], tex_ids[1]]
        pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pointcloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        pointcloud = pointcloud.remove_duplicated_points()
        # align the pointcloud to the coord system of cameras, which is provided inside the mesh_info.json file
        mesh_info_json = json.load(open(glb_filename.parent / "mesh_info.json"))
        transform = np.array(mesh_info_json["alignmentTransform"]).reshape(4, 4).T
        pointcloud = pointcloud.transform(np.linalg.inv(transform))
        # shift the axes coordinates to match the nerfstudio ones (same as the cameras' coord system)
        pointcloud.points = o3d.utility.Vector3dVector(np.array(pointcloud.points)[:, [2, 0, 1]])
        o3d.io.write_point_cloud(str(output_dir / "point_cloud.ply"), pointcloud)
        data["ply_file_path"] = "point_cloud.ply"

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    summary = []
    if skipped_frames > 0:
        summary.append(f"Skipped {skipped_frames} frames due to low blur score.")
    summary.append(f"Final dataset is {len(image_filenames) - skipped_frames} frames.")

    if len(image_filenames) - skipped_frames == 0:
        CONSOLE.print("[bold red]No images remain after filtering, exiting")
        sys.exit(1)

    return summary


def process_images(
    polycam_image_dir: Path,
    image_dir: Path,
    crop_border_pixels: int = 15,
    max_dataset_size: int = 600,
    num_downscales: int = 3,
    verbose: bool = True,
) -> Tuple[List[str], List[Path]]:
    """
    Process RGB images only

    Args:
        polycam_image_dir: Path to the directory containing RGB Images
        image_dir: Output directory for processed images
        crop_border_pixels: Number of pixels to crop from each border of the image. Useful as borders may be
                            black due to undistortion.
        max_dataset_size: Max number of images to train on. If the dataset has more, images will be sampled
                            approximately evenly. If -1, use all images.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
                        will downscale the images by 2x, 4x, and 8x.
        verbose: If True, print extra logging.
    Returns:
        summary_log: Summary of the processing.
        polycam_image_filenames: List of processed images paths
    """
    summary_log = []
    polycam_image_filenames, num_orig_images = process_data_utils.get_image_filenames(
        polycam_image_dir, max_dataset_size
    )

    # Copy images to output directory
    copied_image_paths = process_data_utils.copy_images_list(
        polycam_image_filenames,
        image_dir=image_dir,
        crop_border_pixels=crop_border_pixels,
        verbose=verbose,
        num_downscales=num_downscales,
    )
    num_frames = len(copied_image_paths)

    copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]

    if max_dataset_size > 0 and num_frames != num_orig_images:
        summary_log.append(f"Started with {num_frames} images out of {num_orig_images} total")
        summary_log.append(
            "To change the size of the dataset add the argument --max_dataset_size to larger than the "
            f"current value ({max_dataset_size}), or -1 to use all images."
        )
    else:
        summary_log.append(f"Started with {num_frames} images")

    # Save json
    if num_frames == 0:
        CONSOLE.print("[bold red]No images found, exiting")
        sys.exit(1)

    return summary_log, polycam_image_filenames


def process_depth_maps(
    polycam_depth_dir: Path,
    depth_dir: Path,
    num_processed_images: int,
    crop_border_pixels: int = 15,
    max_dataset_size: int = 600,
    num_downscales: int = 3,
    verbose: bool = True,
) -> Tuple[List[str], List[Path]]:
    """
    Process Depth maps from polycam only

    Args:
        polycam_depth_dir: Path to the directory containing depth maps
        depth_dir: Output directory for processed depth maps
        num_processed_images: Number of RGB processed that must match the number of depth maps
        crop_border_pixels: Number of pixels to crop from each border of the image. Useful as borders may be
                            black due to undistortion.
        max_dataset_size: Max number of images to train on. If the dataset has more, images will be sampled
                         approximately evenly. If -1, use all images.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
                        will downscale the images by 2x, 4x, and 8x.
        verbose: If True, print extra logging.
    Returns:
        summary_log: Summary of the processing.
        polycam_depth_maps_filenames: List of processed depth maps paths
    """
    summary_log = []
    polycam_depth_maps_filenames, num_orig_depth_maps = process_data_utils.get_image_filenames(
        polycam_depth_dir, max_dataset_size
    )

    # Copy depth images to output directory
    copied_depth_maps_paths = process_data_utils.copy_and_upscale_polycam_depth_maps_list(
        polycam_depth_maps_filenames,
        depth_dir=depth_dir,
        num_downscales=num_downscales,
        crop_border_pixels=crop_border_pixels,
        verbose=verbose,
    )

    num_processed_depth_maps = len(copied_depth_maps_paths)

    # assert same number of images as depth maps
    if num_processed_images != num_processed_depth_maps:
        raise ValueError(
            f"Expected same amount of depth maps as images. "
            f"Instead got {num_processed_images} images and {num_processed_depth_maps} depth maps"
        )

    if crop_border_pixels > 0 and num_processed_depth_maps != num_orig_depth_maps:
        summary_log.append(f"Started with {num_processed_depth_maps} images out of {num_orig_depth_maps} total")
        summary_log.append(
            "To change the size of the dataset add the argument --max_dataset_size to larger than the "
            f"current value ({crop_border_pixels}), or -1 to use all images."
        )
    else:
        summary_log.append(f"Started with {num_processed_depth_maps} images")

    return summary_log, polycam_depth_maps_filenames
