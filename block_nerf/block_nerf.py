import glob
import json
import os
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def split_transforms(path: Path, splits: int):
    transforms = load_json(path)
    frames = transforms["frames"]
    split_frames = np.array_split(frames, splits)

    image_indexes = []
    new_transforms = []
    for split in split_frames:
        new_transforms.append(
            {
                "camera_model": transforms["camera_model"],
                "fl_x": transforms["fl_x"],
                "fl_y": transforms["fl_y"],
                "cx": transforms["cx"],
                "cy": transforms["cy"],
                "w": transforms["w"],
                "h": transforms["h"],
                "k1": transforms["k1"],
                "k2": transforms["k2"],
                "p1": transforms["p1"],
                "p2": transforms["p2"],
                "frames": split.tolist(),
            }
        )
        if len(image_indexes) == 0:
            image_indexes.append(len(split))
        else:
            image_indexes.append(image_indexes[-1] + len(split))
    return new_transforms, image_indexes


def write_transforms(transforms: list, image_indexes: list, path: Path):
    original_images_path = Path("images")
    images = glob.glob(f"{original_images_path}/*.png")
    images.sort()

    for i, transform in enumerate(transforms):
        split_path = path / f"{i}"
        split_path.mkdir(parents=True, exist_ok=True)
        with open(split_path / "transforms.json", "w") as f:
            json.dump(transform, f, indent=4)

        image_path = split_path / "images"
        image_path.mkdir(parents=True, exist_ok=True)

        for j in range(image_indexes[i]):
            os.symlink(f"{images[j]}", f"{image_path}/{j}.png")