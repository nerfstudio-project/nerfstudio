# credit: Hang Gao (hangg@berkeley.edu)

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place"
    )

    parser.add_argument("--capture_dir", type=str, help="input path to the images")
    parser.add_argument("--output_root", "-o", help="output path")
    args = parser.parse_args()
    return args
