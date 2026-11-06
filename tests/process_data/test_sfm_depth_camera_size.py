from pathlib import Path

import cv2
import numpy as np

from nerfstudio.data.utils.colmap_parsing_utils import (
    Camera,
    Image,
    Point3D,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)
from nerfstudio.process_data.colmap_utils import create_sfm_depth


def test_sfm_depth_uses_each_images_camera(tmp_path: Path):
    cameras = {key: Camera(key, "PINHOLE", w, h, [1, 1, 0, 0]) for key, w, h in [(3, 4, 2), (8, 9, 5)]}
    images = {
        i: Image(i, np.array([1, 0, 0, 0]), np.zeros(3), camera_id, f"{i}.png", np.array([xy]), np.array([1]))
        for i, camera_id, xy in [(1, 3, [2, 1]), (2, 8, [7, 4])]
    }
    point = Point3D(1, np.array([0, 0, 2]), np.zeros(3, dtype=int), 0.0, np.array([1, 2]), np.array([0, 0]))
    write_cameras_binary(cameras, tmp_path / "cameras.bin")
    write_images_binary(images, tmp_path / "images.bin")
    write_points3D_binary({1: point}, tmp_path / "points3D.bin")

    paths = create_sfm_depth(tmp_path, tmp_path, verbose=False)
    for i, image in images.items():
        depth = cv2.imread(str(paths[i]), cv2.IMREAD_UNCHANGED)
        camera = cameras[image.camera_id]
        assert depth.shape == (camera.height, camera.width)
        x, y = image.xys[0]
        assert depth[y, x] == 2000
