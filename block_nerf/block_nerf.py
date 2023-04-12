import glob
import json
import shutil
from pathlib import Path

import numpy as np
import torch


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
    original_images_path = path / "images"
    images = glob.glob(f"{original_images_path}/*.png")
    images.sort()

    for i, transform in enumerate(transforms):
        split_path = path / f"{i}"
        split_path.mkdir(parents=True, exist_ok=True)
        with open(split_path / "transforms.json", "w") as f:
            json.dump(transform, f, indent=4)

        image_path = split_path / "images"
        image_path.mkdir(parents=True, exist_ok=True)

        for j in range(0 if i == 0 else image_indexes[i-1], image_indexes[i]):
            shutil.copyfile(f"{images[j]}", f"{image_path}/{images[j].split('/')[-1]}")

def transform_camera_path_to_original_space(camera_path_path: Path, pipeline: VanillaPipeline):
    """
        The camera_path_path must be the camera path created in the pipeline.
    """
    camera_path = load_json(camera_path_path)

    poses = []
    for i, camera in enumerate(camera_path["camera_path"]):
        c2w = np.array(camera["camera_to_world"]).reshape(4, 4)
        poses.append(c2w)
    
    poses = torch.tensor(poses)[:, :-1, :]
    transformed_poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(poses).numpy()
    for i, camera in enumerate(camera_path["camera_path"]):
        camera["camera_to_world"] = transformed_poses[i].reshape(16).tolist()
    
    export_path = camera_path_path.parent / "camera_path_transformed_original.json"
    with open(export_path, "w") as f:
        json.dump(camera_path, f, indent=4)
    
    return export_path


def transform_camera_path(camera_path_path: Path, dataparser_transform_path: Path):
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
    
    with open(camera_path_path.parent / "camera_path_transformed.json", "w") as f:
        json.dump(camera_path, f, indent=4)

if __name__ == "__main__":
    camera_path_path = Path("block_nerf/camera_path_one_lap_final_copy.json")
    exp_path = Path("data/images/exp_combined_baseline_2")
    config_path = exp_path / "exp_combined_baseline_2/nerfacto/2023-04-10_140345/config.yml"
    dataparser_transform_path = exp_path / "exp_combined_baseline_2/nerfacto/2023-04-10_140345/dataparser_transforms.json"
    
    eval_num_rays_per_chunk = 1 << 15 # Same as 2^15
    _, pipeline, _ = eval_setup(
        config_path,
        eval_num_rays_per_chunk=eval_num_rays_per_chunk,
        test_mode="inference",
    )

    original_camera_path_path = transform_camera_path_to_original_space(camera_path_path, pipeline)

    transform_camera_path(
        original_camera_path_path,
        dataparser_transform_path,
    )
