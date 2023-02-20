# Copyright 2022 The Nerfstudio Team. All rights reserved.
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


import json
import os
from pathlib import Path
from typing import Dict, Optional

import appdirs
import cv2
import numpy as np
import pycolmap
import requests
from rich.console import Console
from rich.progress import track
from typing_extensions import Literal

from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)


def get_colmap_version(colmap_cmd: str, default_version=3.8) -> float:
    """Returns the version of COLMAP.
    This code assumes that colmap returns a version string of the form
    "COLMAP 3.8 ..." which may not be true for all versions of COLMAP.

    Args:
        default_version: Default version to return if COLMAP version can't be determined.
    Returns:
        The version of COLMAP.
    """
    output = run_command(colmap_cmd, verbose=False)
    assert output is not None
    for line in output.split("\n"):
        if line.startswith("COLMAP"):
            return float(line.split(" ")[1])
    CONSOLE.print(f"[bold red]Could not find COLMAP version. Using default {default_version}")
    return default_version


def get_vocab_tree() -> Path:
    """Return path to vocab tree. Downloads vocab tree if it doesn't exist.

    Returns:
        The path to the vocab tree.
    """
    vocab_tree_filename = Path(appdirs.user_data_dir("nerfstudio")) / "vocab_tree.fbow"

    if not vocab_tree_filename.exists():
        r = requests.get("https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin", stream=True)
        vocab_tree_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_tree_filename, "wb") as f:
            total_length = r.headers.get("content-length")
            assert total_length is not None
            for chunk in track(
                r.iter_content(chunk_size=1024),
                total=int(total_length) / 1024 + 1,
                description="Downloading vocab tree...",
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return vocab_tree_filename


def run_colmap(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    camera_mask_path: Optional[Path] = None,
    gpu: bool = True,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    colmap_cmd: str = "colmap",
) -> None:
    """Runs COLMAP on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        camera_mask_path: Path to the camera mask.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
        matching_method: Matching method to use.
        colmap_cmd: Path to the COLMAP executable.
    """

    colmap_version = get_colmap_version(colmap_cmd)

    colmap_database_path = colmap_dir / "database.db"
    if colmap_database_path.exists():
        # Can't use missing_ok argument because of Python 3.7 compatibility.
        colmap_database_path.unlink()

    # Feature extraction
    feature_extractor_cmd = [
        f"{colmap_cmd} feature_extractor",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {camera_model.value}",
        f"--SiftExtraction.use_gpu {int(gpu)}",
    ]
    if camera_mask_path is not None:
        feature_extractor_cmd.append(f"--ImageReader.camera_mask_path {camera_mask_path}")
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    with status(msg="[bold yellow]Running COLMAP feature extractor...", spinner="moon", verbose=verbose):
        run_command(feature_extractor_cmd, verbose=verbose)

    CONSOLE.log("[bold green]:tada: Done extracting COLMAP features.")

    # Feature matching
    feature_matcher_cmd = [
        f"{colmap_cmd} {matching_method}_matcher",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--SiftMatching.use_gpu {int(gpu)}",
    ]
    if matching_method == "vocab_tree":
        vocab_tree_filename = get_vocab_tree()
        feature_matcher_cmd.append(f"--VocabTreeMatching.vocab_tree_path {vocab_tree_filename}")
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    with status(msg="[bold yellow]Running COLMAP feature matcher...", spinner="runner", verbose=verbose):
        run_command(feature_matcher_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done matching COLMAP features.")

    # Bundle adjustment
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        f"{colmap_cmd} mapper",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]
    if colmap_version >= 3.7:
        mapper_cmd.append("--Mapper.ba_global_function_tolerance 1e-6")

    mapper_cmd = " ".join(mapper_cmd)

    with status(
        msg="[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(mapper_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment.")
    with status(msg="[bold yellow]Refine intrinsics...", spinner="dqpb", verbose=verbose):
        bundle_adjuster_cmd = [
            f"{colmap_cmd} bundle_adjuster",
            f"--input_path {sparse_dir}/0",
            f"--output_path {sparse_dir}/0",
            "--BundleAdjustment.refine_principal_point 1",
        ]
        run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done refining intrinsics.")


def colmap_to_json(
    recon_dir: Path,
    output_dir: Path,
    camera_model: CameraModel,
    camera_mask_path: Optional[Path] = None,
    image_id_to_depth_path: Optional[Dict[int, Path]] = None,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_mask_path: Path to the camera mask.
        camera_model: Camera model used.

    Returns:
        The number of registered images.
    """

    recon = pycolmap.Reconstruction(recon_dir)
    cam_id_to_camera = recon.cameras
    im_id_to_image = recon.images

    # Only support first camera
    CAMERA_ID = 1
    camera_params = cam_id_to_camera[CAMERA_ID].params

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.
        rotation = im_data.rotation_matrix()
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = Path(f"./images/{im_data.name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
        if image_id_to_depth_path is not None:
            depth_path = image_id_to_depth_path[im_id]
            frame["depth_file_path"] = str(depth_path)
        frames.append(frame)

    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[1]),
        "cx": float(camera_params[2]),
        "cy": float(camera_params[3]),
        "w": cam_id_to_camera[CAMERA_ID].width,
        "h": cam_id_to_camera[CAMERA_ID].height,
        "camera_model": camera_model.value,
    }

    if camera_model == CameraModel.OPENCV:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "p1": float(camera_params[6]),
                "p2": float(camera_params[7]),
            }
        )
    if camera_model == CameraModel.OPENCV_FISHEYE:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "k3": float(camera_params[6]),
                "k4": float(camera_params[7]),
            }
        )

    out["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)


def create_sfm_depth(
    recon_dir: Path,
    output_dir: Path,
    verbose: bool = True,
    depth_scale_to_integer_factor: float = 1000.0,
    min_depth: float = 0.001,
    max_depth: float = 10000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 2,
) -> Dict[int, Path]:
    """Converts COLMAP's points3d.bin to sparse depth map images encoded as
    16-bit "millimeter depth" PNGs.

    Notes:
     * This facility does NOT use COLMAP dense reconstruction; it creates depth
        maps from sparse SfM points here.
     * COLMAP does *not* reconstruct metric depth unless you give it calibrated
        (metric) intrinsics as input. Therefore, "depth" in this function has
        potentially ambiguous units.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        verbose: If True, logs progress of depth image creation.
        depth_scale_to_integer_factor: Use this parameter to tune the conversion of
          raw depth measurements to integer depth values.  This value should
          be equal to 1. / `depth_unit_scale_factor`, where
          `depth_unit_scale_factor` is the value you provide at training time.
          E.g. for millimeter depth, leave `depth_unit_scale_factor` at 1e-3
          and depth_scale_to_integer_factor at 1000.
        min_depth: Discard points closer than this to the camera.
        max_depth: Discard points farther than this from the camera.
        max_repoj_err: Discard points with reprojection error greater than this
          amount (in pixels).
        min_n_visible: Discard 3D points that have been triangulated with fewer
          than this many frames.
    Returns:
        Depth file paths indexed by COLMAP image id
    """

    recon = pycolmap.Reconstruction(recon_dir)
    ptid_to_info = recon.points3D
    im_id_to_image = recon.images
    cam_id_to_camera = recon.cameras

    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    if verbose:
        iter_images = track(
            im_id_to_image.items(), total=len(im_id_to_image.items()), description="Creating depth maps ..."
        )
    else:
        iter_images = iter(im_id_to_image.items())

    image_id_to_depth_path = {}
    for im_id, im_data in iter_images:
        # Get only keypoints that have corresponding triangulated 3D points
        p2ds = im_data.get_valid_points2D()

        xyz_world = np.array([ptid_to_info[p2d.point3D_id].xyz for p2d in p2ds])

        # COLMAP OpenCV convention: z is always positive
        z = (im_data.rotation_matrix() @ xyz_world.T)[-1] + im_data.tvec[-1]

        # Mean reprojection error in image space
        errors = np.array([ptid_to_info[p2d.point3D_id].error for p2d in p2ds])

        # Number of frames in which each frame is visible
        n_visible = np.array([ptid_to_info[p2d.point3D_id].track.length() for p2d in p2ds])

        # Note: these are *unrectified* pixel coordinates that should match the original input
        # no matter the camera model
        uv = np.array([p2d.xy for p2d in p2ds])
        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z

        # E.g. if `depth` is metric and in units of meters, and `depth_scale_to_integer_factor`
        # is 1000, then `depth_img` will be integer millimeters.
        depth_img = (depth_scale_to_integer_factor * depth).astype(np.uint16)

        out_name = str(im_data.name)
        depth_path = output_dir / out_name
        cv2.imwrite(str(depth_path), depth_img)

        image_id_to_depth_path[im_id] = depth_path
    return image_id_to_depth_path


def get_matching_summary(num_intial_frames: int, num_matched_frames: int) -> str:
    """Returns a summary of the matching results.

    Args:
        num_intial_frames: The number of initial frames.
        num_matched_frames: The number of matched frames.

    Returns:
        A summary of the matching results.
    """
    match_ratio = num_matched_frames / num_intial_frames
    if match_ratio == 1:
        return "[bold green]COLMAP found poses for all images, CONGRATS!"
    if match_ratio < 0.4:
        result = f"[bold red]COLMAP only found poses for {num_matched_frames / num_intial_frames * 100:.2f}%"
        result += " of the images. This is low.\nThis can be caused by a variety of reasons,"
        result += " such poor scene coverage, blurry images, or large exposure changes."
        return result
    if match_ratio < 0.8:
        result = f"[bold yellow]COLMAP only found poses for {num_matched_frames / num_intial_frames * 100:.2f}%"
        result += " of the images.\nThis isn't great, but may be ok."
        result += "\nMissing poses can be caused by a variety of reasons, such poor scene coverage, blurry images,"
        result += " or large exposure changes."
        return result
    return f"[bold green]COLMAP found poses for {num_matched_frames / num_intial_frames * 100:.2f}% of the images."
