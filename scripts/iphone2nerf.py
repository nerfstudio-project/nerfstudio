# credit: Hang Gao (hangg@berkeley.edu)

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames and pose data from the whole sequence of \
        the raw Record3D capture, and convert to nerf format."
    )
    parser.add_argument("--capture_dir", type=str, help="input path to the images")
    parser.add_argument("--output_root", "-o", type=str, help="output path")
    parser.add_argument("--step", "-s", type=int, default=1, help="use every x'th frame.")
    parser.add_argument(
        "--aabb_scale",
        default=16,
        choices=["1", "2", "4", "8", "16"],
        help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Input Paths.
    capture_dir = Path(args.capture_dir)
    capture_imgs_dir = capture_dir / "rgb"
    metadata_path = capture_dir / "metadata.json"

    # Output Paths.
    output_root = Path(args.output_root)
    output_images = output_root / "images"
    output_transforms = output_root / "transforms.json"

    with open(str(metadata_path)) as f:
        metadata_dict = json.load(f)

    # Camera intrinsics
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    AABB_SCALE = args.aabb_scale
    focal_length = K[0, 0]
    principal_point = K[:2, -1]
    skew = 0
    pixel_aspect_ratio = 1.0
    tangential_distortion = np.array([0.0, 0.0])

    H = metadata_dict["h"]
    W = metadata_dict["w"]

    angle_x = math.atan(W / (focal_length * 2)) * 2
    angle_y = math.atan(H / (focal_length * 2)) * 2

    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": focal_length,
        "fl_y": focal_length,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": principal_point[0],
        "cy": principal_point[1],
        "w": W,
        "h": H,
        "aabb_scale": AABB_SCALE,
        "frames": [],
    }

    input_image_files = os.listdir(capture_imgs_dir)

    for i in range(0, len(input_image_files), args.step):
        pass


if __name__ == "__main__":
    main()
