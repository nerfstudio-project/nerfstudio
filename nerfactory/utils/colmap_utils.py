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

import os
import struct
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CameraModel:
    """Camera model.

    Attributes:
        model_id: Model identifier.
        model_name: Model name.
        num_params: Number of parameters.
    """

    model_id: int
    model_name: str
    num_params: int


@dataclass
class Camera:
    """Camera

    Attributes:
        camera_id: Camera identifier.
        model: Camera model.
        width: Image width.
        height: Image height.
        params: Camera parameters.
    """

    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    """Data the corresponds to a single image.

    Attributes:
        id: Image identifier.
        qvec: Quaternion vector.
        tvec: Translation vector.
        camera_id: Camera identifier.
        name: Image name.
        xys: 2D points.
        point3D_ids: Point3D identifiers.
    """

    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray


@dataclass
class Point3D:
    """Data that corresponds to a single 3D point.

    Attributes:
        id: Point3D identifier.
        xyz: 3D point.
        rgb: Color.
        error: Reconstruction error.
        image_ids: Image identifiers.
        point2d_idxs: Point2D indices.
    """

    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2d_idxs: np.ndarray


CAMERA_MODELS = [
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
]
CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {camera_model.model_name: camera_model for camera_model in CAMERA_MODELS}


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
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
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
        True if the model file is the tested extenstion, False otherwise.
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
