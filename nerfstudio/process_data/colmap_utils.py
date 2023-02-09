"""
Here we have modified code taken from COLMAP for parsing data in the COLMAP format.
Original file at:
https://github.com/colmap/colmap/blob/1a4d0bad2e90aa65ce997c9d1779518eaed998d5/scripts/python/read_write_model.py.
"""

# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import json
import os
import struct
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import Dict, Optional, Tuple

import appdirs
import numpy as np
import requests
from rich.console import Console
from rich.progress import track
from typing_extensions import Literal

from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)


@dataclass
class ColmapCameraModel:
    """Camera model"""

    model_id: int
    """Model identifier"""
    model_name: str
    """Model name"""
    num_params: int
    """Number of parameters"""


@dataclass
class Camera:
    """Camera"""

    id: int
    """Camera identifier"""
    model: str
    """Camera model"""
    width: int
    """Image width"""
    height: int
    """Image height"""
    params: np.ndarray
    """Camera parameters"""


@dataclass
class Image:
    """Data the corresponds to a single image"""

    id: int
    """Image identifier"""
    qvec: np.ndarray
    """Quaternion vector"""
    tvec: np.ndarray
    """Translation vector"""
    camera_id: int
    """Camera identifier"""
    name: str
    """Image name"""
    xys: np.ndarray
    """2D points"""
    point3d_ids: np.ndarray
    """Point3D identifiers"""


@dataclass
class Point3D:
    """Data that corresponds to a single 3D point"""

    id: int
    """Point3D identifier"""
    xyz: np.ndarray
    """3D point"""
    rgb: np.ndarray
    """Color"""
    error: float
    """Reconstruction error"""
    image_ids: np.ndarray
    """Image identifiers"""
    point2d_idxs: np.ndarray
    """Point2D indices"""


COLMAP_CAMERA_MODELS = [
    ColmapCameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    ColmapCameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    ColmapCameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    ColmapCameraModel(model_id=3, model_name="RADIAL", num_params=5),
    ColmapCameraModel(model_id=4, model_name="OPENCV", num_params=8),
    ColmapCameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    ColmapCameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    ColmapCameraModel(model_id=7, model_name="FOV", num_params=5),
    ColmapCameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    ColmapCameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    ColmapCameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
]
COLMAP_CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in COLMAP_CAMERA_MODELS}
COLMAP_CAMERA_MODEL_NAMES = {camera_model.model_name: camera_model for camera_model in COLMAP_CAMERA_MODELS}


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


def read_next_bytes(fid: BufferedReader, num_bytes: int, format_char_sequence, endian_character: str = "<"):
    """Read and unpack the next bytes from a binary file.

    Args:
        fid: Open file
        num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        endian_character: Any of {@, =, <, >, !}
        Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path: Path) -> Dict[int, Camera]:
    """Parse COLMAP cameras.txt file into a dictionary of Camera objects.

    Args:
        path: Path to cameras.txt file.
    Returns:
        Dictionary of Camera objects.
    """
    cameras = {}
    with open(path, encoding="utf-8") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras


def read_cameras_binary(path_to_model_file: Path) -> Dict[int, Camera]:
    """Parse COLMAP cameras.bin file into a dictionary of Camera objects.

    Args:
        path_to_model_file: Path to cameras.bin file.
    Returns:
        Dictionary of Camera objects.
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = COLMAP_CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = COLMAP_CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path: Path) -> Dict[int, Image]:
    """Parse COLMAP images.txt file into a dictionary of Image objects.

    Args:
        path: Path to images.txt file.
    Returns:
        Dictionary of Image objects.
    """
    images = {}
    with open(path, encoding="utf-8") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3d_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3d_ids=point3d_ids,
                )
    return images


def read_images_binary(path_to_model_file: Path) -> Dict[int, Image]:
    """Parse COLMAP images.bin file into a dictionary of Image objects.

    Args:
        path_to_model_file: Path to images.bin file.
    Returns:
        Dictionary of Image objects.
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2d = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2d, format_char_sequence="ddq" * num_points2d)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3d_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3d_ids=point3d_ids,
            )
    return images


def read_points3d_text(path) -> Dict[int, Point3D]:
    """Parse COLMAP points3D.txt file into a dictionary of Point3D objects.

    Args:
        path: Path to points3D.txt file.
    Returns:
        Dictionary of Point3D objects.
    """
    points3d = {}
    with open(path, encoding="utf-8") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3d_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2d_idxs = np.array(tuple(map(int, elems[9::2])))
                points3d[point3d_id] = Point3D(
                    id=point3d_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2d_idxs=point2d_idxs
                )
    return points3d


def read_points3d_binary(path_to_model_file: Path) -> Dict[int, Point3D]:
    """Parse COLMAP points3D.bin file into a dictionary of Point3D objects.

    Args:
        path_to_model_file: Path to points3D.bin file.
    Returns:
        Dictionary of Point3D objects.
    """
    points3d = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3d_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2d_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3d[point3d_id] = Point3D(
                id=point3d_id, xyz=xyz, rgb=rgb, error=float(error), image_ids=image_ids, point2d_idxs=point2d_idxs
            )
    return points3d


def detect_model_format(path: Path, ext: str) -> bool:
    """Detect the format of the model file.

    Args:
        path: Path to the model file.
        ext: Extension to test.
    Returns:
        True if the model file is the tested extension, False otherwise.
    """

    if (
        os.path.isfile(path / f"cameras{ext}")
        and os.path.isfile(path / f"images{ext}")
        and os.path.isfile(path / f"points3D{ext}")
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path: Path, ext: Optional[str] = None) -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    """Read a COLMAP model from a directory.

    Args:
        path: Path to the model directory.
        ext: Extension of the model files. If None, the function will try to detect the format.
    Returns:
        Tuple of dictionaries of Camera, Image, and Point3D objects.
    """
    # try to detect the extension automatically
    if ext is None:
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            raise ValueError("Provide model format: '.bin' or '.txt'")

    if ext == ".txt":
        cameras = read_cameras_text(path / f"cameras{ext}")
        images = read_images_text(path / f"images{ext}")
        points3d = read_points3d_text(path / f"points3D{ext}")
    else:
        cameras = read_cameras_binary(path / f"cameras{ext}")
        images = read_images_binary(path / f"images{ext}")
        points3d = read_points3d_binary(path / f"points3D{ext}")
    return cameras, images, points3d


def qvec2rotmat(qvec) -> np.ndarray:
    """Convert quaternion to rotation matrix.

    Args:
        qvec: Quaternion vector of shape (4,).
    Returns:
        Rotation matrix of shape (3, 3).
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    """Convert rotation matrix to quaternion.

    Args:
        R: Rotation matrix of shape (3, 3).
    Returns:
        Quaternion vector of shape (4,).
    """
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = R.flat
    K = (
        np.array(
            [
                [rxx - ryy - rzz, 0, 0, 0],
                [ryx + rxy, ryy - rxx - rzz, 0, 0],
                [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
                [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[np.array([3, 0, 1, 2]), np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


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
    cameras_path: Path,
    images_path: Path,
    output_dir: Path,
    camera_model: CameraModel,
    camera_mask_path: Optional[Path] = None,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        cameras_path: Path to the cameras.bin file.
        images_path: Path to the images.bin file.
        output_dir: Path to the output directory.
        camera_mask_path: Path to the camera mask.
        camera_model: Camera model used.

    Returns:
        The number of registered images.
    """

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)

    # Images were renamed to frame_{i:05d}.{ext} and
    # the filenames needs to be replaced in the transforms.json as well
    original_filenames = [x.name for x in images.values()]
    # Sort was used in nerfstudio.process_data.process_data_utils:get_image_filenames
    original_filenames.sort()
    # Build the map to the new filenames
    filename_map = {name: f"frame_{i+1:05d}{os.path.splitext(name)[-1]}" for i, name in enumerate(original_filenames)}

    # Only supports one camera
    camera_params = cameras[1].params

    frames = []
    for _, im_data in images.items():
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = filename_map[im_data.name]
        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
        frames.append(frame)

    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[1]),
        "cx": float(camera_params[2]),
        "cy": float(camera_params[3]),
        "w": cameras[1].width,
        "h": cameras[1].height,
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
