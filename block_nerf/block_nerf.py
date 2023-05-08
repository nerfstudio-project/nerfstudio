import copy
import glob
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup


def load_json(path: Union[Path, str]):
    with open(path, "r") as f:
        return json.load(f)


def split_transforms(path: Path, splits: int, overlap: int = 10):
    transforms = load_json(path)
    frames = transforms["frames"]
    split_frames = np.array_split(frames, splits)
    image_indexes: List[Tuple[int, int]] = []

    # Add overlap to the split_frames
    assert overlap < len(split_frames[0])
    for i, split in enumerate(split_frames):
        if i == 0:
            split_frames[i] = np.concatenate((split, split_frames[i + 1][:overlap]))
            image_indexes.append((0, len(split) + overlap))
        elif i == len(split_frames) - 1:
            split_frames[i] = np.concatenate((split_frames[i - 1][-overlap * 2 : -overlap], split))
            start_index = image_indexes[-1][1] - 2 * overlap
            image_indexes.append((start_index, start_index + len(split) + overlap))
        else:
            split_frames[i] = np.concatenate(
                (split_frames[i - 1][-overlap * 2 : -overlap], split, split_frames[i + 1][:overlap])
            )
            start_index = image_indexes[-1][1] - 2 * overlap
            image_indexes.append((start_index, start_index + len(split) + 2 * overlap))

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
    return new_transforms, image_indexes


def write_transforms(transforms: list, image_indexes: List[Tuple[int, int]], path: Path):
    original_images_path = path / "images"
    images = glob.glob(f"{original_images_path}/*.png")
    images.sort()

    for i, transform in enumerate(transforms):
        split_path = path / f"block_{i}"  # f"block_{i}" is the block_name.
        split_path.mkdir(parents=True, exist_ok=True)
        with open(split_path / "transforms.json", "w") as f:
            json.dump(transform, f, indent=4)

        image_path = split_path / "images"
        image_path.mkdir(parents=True, exist_ok=True)

        for j in range(image_indexes[i][0], image_indexes[i][1]):
            shutil.copyfile(f"{images[j]}", f"{image_path}/{images[j].split('/')[-1]}")


def transform_camera_path_to_original_space(camera_path_path: Path, pipeline):
    """
    The camera_path_path must be the camera path created in the pipeline.
    """
    camera_path = load_json(camera_path_path)

    poses = np.array([np.array(camera["camera_to_world"]) for camera in camera_path["camera_path"]]).reshape(-1, 4, 4)
    poses = torch.tensor(poses)[:, :-1, :].float()
    transformed_poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
        poses, camera_convention="opengl"
    ).numpy()
    for i, camera in enumerate(camera_path["camera_path"]):
        temp = np.array(transformed_poses[i])
        temp = np.vstack((temp, np.array([0, 0, 0, 1])))
        camera["camera_to_world"] = temp.reshape(16).tolist()

    export_path = camera_path_path.parent / "camera_path_nerf_coordinates.json"
    with open(export_path, "w") as f:
        json.dump(camera_path, f, indent=4)

    print("✅ Created transformed camera path at: ", export_path)
    return export_path


def transform_camera_path(camera_path_path: Path, dataparser_transform_path: Path, export_path: Optional[Path] = None):
    """
    Transform a un-transformed camera path to a transformed camera path, in the respective transform's coordinate system.
    """
    camera_path = load_json(camera_path_path)
    dataparser_transform = load_json(dataparser_transform_path)

    t = np.array(dataparser_transform["transform"])
    s = dataparser_transform["scale"]

    for i, camera in enumerate(camera_path["camera_path"]):
        c2w = np.array(camera["camera_to_world"]).reshape(4, 4)
        c2w = (t @ c2w) * s
        c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))
        camera["camera_to_world"] = c2w.reshape(16).tolist()

    export_path = export_path if export_path is not None else camera_path_path.parent / "camera_path_transformed.json"
    with open(export_path, "w") as f:
        json.dump(camera_path, f, indent=4)

    print("✅ Created transformed camera path at: ", export_path)
    return export_path


def transform_to_single_camera_path(
    camera_path_path: Path, block_lookup: Dict[str, str], dataparser_transform_paths: Dict[str, Path], export_dir: Path
):
    """
    Transform a un-transformed camera path to a transformed camera path, in the respective transform's coordinate system.
    dataparser_transform_paths is a dictionary of block_name: dataparser_transform_path
    """
    assert set(block_lookup.values()) == set(dataparser_transform_paths.keys())

    original_camera_path = load_json(camera_path_path)
    new_camera_path = copy.deepcopy(original_camera_path)

    # Load the dataparser_transforms within the dictionary
    dataparser_transforms = {}
    for block_name, dataparser_transform_path in dataparser_transform_paths.items():
        dataparser_transform = load_json(dataparser_transform_path)
        dataparser_transforms[block_name] = dataparser_transform

    for i, camera in enumerate(new_camera_path["camera_path"]):
        block_name = block_lookup[str(i)]
        t = np.array(dataparser_transforms[block_name]["transform"])
        s = dataparser_transforms[block_name]["scale"]

        c2w = np.array(camera["camera_to_world"]).reshape(4, 4)
        c2w = (t @ c2w) * s
        c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))
        camera["camera_to_world"] = c2w.reshape(16).tolist()

    export_path = export_dir / "combined_camera_path.json"
    with open(export_path, "w") as f:
        json.dump(new_camera_path, f, indent=4)

    print(f"✅ Created transformed camera path for {len(block_lookup)} blocks at: ", export_path)
    return export_path


def create_block_lookup(exp_path: Path, camera_path_path: Path, block_transforms: Dict[str, Path]):
    transforms = {block_name: load_json(transform_path) for block_name, transform_path in block_transforms.items()}
    camera_path = load_json(camera_path_path)

    # Define a function to compute the Euclidean distance between two 4x4 homogeneous matrices
    def matrix_distance(m1: np.ndarray, m2: np.ndarray) -> float:
        return np.linalg.norm(np.array(m1) - np.array(m2))

    # Create a lookup table that maps each camera path to its closest transform matrix
    lookup_table = {}
    for i, camera in enumerate(camera_path["camera_path"]):
        # Get the location from the camera_to_world matrix
        query_location = np.array(camera["camera_to_world"]).reshape(4, 4)[:3, 3]

        distances = {
            block_name: [
                matrix_distance(query_location, np.array(frame["transform_matrix"])[:3, 3])
                for frame in transform["frames"]
            ]
            for block_name, transform in transforms.items()
        }
        closest_block = min(distances, key=lambda block_name: min(distances[block_name]))
        closest_distance = min(distances[closest_block])
        if closest_distance > 5:
            print("⚠️ Warning: Closest distance is greater than 5: ", closest_distance)

        # print(f"Closest index: {closest_index}")
        # print(f"Closest distance: {closest_distance}")
        # print(f"Query location: {tuple(query_location)}")
        lookup_table[f"{i}"] = closest_block

    export_path = exp_path / "lookup_table.json"
    with open(export_path, "w") as f:
        json.dump(lookup_table, f, indent=4)

    print("✅ Created lookup table at: ", export_path)


def _test_block_nerf():
    # Leave all variables with "source" as the prefix as they are the original variables.
    source_camera_path_path = Path("block_nerf/camera_path_one_lap_final_copy.json")
    source_exp_path = Path("data/images/exp_combined_baseline_2")
    source_config_path = source_exp_path / "exp_combined_baseline_2/nerfacto/2023-04-10_140345/config.yml"

    # Change the variables with "target" as the prefix to the variables you want to transform.
    target_exp_path = Path("data/images/exp_combined_baseline_block_nerf_2/1")
    target_dataparser_transforms_path = (
        target_exp_path / "exp_combined_baseline_block_nerf_2-1/nerfacto/2023-04-11_130124/dataparser_transforms.json"
    )

    eval_num_rays_per_chunk = 1 << 15  # Same as 2^15
    _, pipeline, _ = eval_setup(
        source_config_path,
        eval_num_rays_per_chunk=eval_num_rays_per_chunk,
        test_mode="inference",
    )

    original_camera_path_path = transform_camera_path_to_original_space(source_camera_path_path, pipeline)

    transform_camera_path(
        original_camera_path_path,
        target_dataparser_transforms_path,
        export_path=target_exp_path / "camera_path_transformed.json",
    )


def get_block_lookup(exp_path: Path, block_transforms: Dict[str, Path], camera_path_path: Path) -> Dict[str, str]:
    lookup_path = exp_path / "lookup_table.json"
    if not lookup_path.exists():
        print("⚠️ Warning: Lookup table does not exist. Creating one now.")
        create_block_lookup(exp_path, camera_path_path=camera_path_path, block_transforms=block_transforms)
    return load_json(lookup_path)


if __name__ == "__main__":
    original_camera_path = Path("block_nerf/camera_path_nerf_coordinates.json")
    exp_path = Path("data/images/exp_combined_baseline_block_nerf_3")
    block_paths = []  # TODO: Not correct
    create_block_lookup(exp_path, original_camera_path)
