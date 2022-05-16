"""
Code to handle loading friends datasets.
"""

import os
import numpy as np

import torch

from mattport.nerf.dataset.colmap_utils import read_cameras_binary, read_images_binary, read_pointsTD_binary
from mattport.nerf.dataset.structs import DatasetInputs, PointCloud, Semantics
from mattport.utils import profiler
from mattport.utils.io import load_from_json


@profiler.time_function
def load_friends_data(basedir, downscale_factor=1, split="train", include_semantics=True, include_point_cloud=False):
    """_summary_

    Args:
        basedir (_type_): _description_
        downscale_factor (float, optional): _description_. Defaults to 1.0.
        split (str, optional): _description_. Defaults to "train".
        include_point_cloud (bool): whether or not to include the point cloud
    """
    # pylint: disable=unused-argument

    # --- image filenames ----
    images_data = read_images_binary(os.path.join(basedir, "colmap", "images.bin"))
    # `image_path` is only the end of the filename, including the extension e.g., `.jpg`
    image_paths = sorted(os.listdir(os.path.join(basedir, "images")))

    image_path_to_image_id = {}
    image_id_to_image_path = {}
    for v in images_data.values():
        image_path_to_image_id[v.name] = v.id
        image_id_to_image_path[v.id] = v.name
    # TODO: handle the splits differently
    image_filenames = [os.path.join(basedir, "images", image_path) for image_path in image_paths]

    # --- intrinsics ---
    cameras_data = read_cameras_binary(os.path.join(basedir, "colmap", "cameras.bin"))
    intrinsics = []
    for image_path in image_paths:
        cam = cameras_data[image_path_to_image_id[image_path]]
        assert len(cam.params) == 3
        focal_length = cam.params[0]  # f (fx and fy)
        cx = cam.params[1]  # cx
        cy = cam.params[2]  # cy
        intrinsics.append([cx, cy, focal_length])
    intrinsics = torch.tensor(intrinsics).float()

    # --- camera_to_world (extrinsics) ---
    camera_to_world = []
    bottom_row = np.array([0, 0, 0, 1.0]).reshape(1, 4)
    for image_path in image_paths:
        image_data = images_data[image_path_to_image_id[image_path]]
        rot = image_data.qvec2rotmat()
        trans = image_data.tvec.reshape(3, 1)
        c2w = np.concatenate([np.concatenate([rot, trans], 1), bottom_row], 0)
        camera_to_world.append(c2w)
    camera_to_world = torch.tensor(np.array(camera_to_world)).float()
    camera_to_world = torch.inverse(camera_to_world)
    camera_to_world = camera_to_world[:, :3]
    camera_to_world[..., 1:3] *= -1

    # --- masks to mask out things (e.g., people) ---

    # --- semantics ---
    semantics = Semantics()
    if include_semantics:
        thing_filenames = [
            image_filename.replace("/images/", "/segmentations/thing/").replace(".jpg", ".png")
            for image_filename in image_filenames
        ]
        stuff_filenames = [
            image_filename.replace("/images/", "/segmentations/stuff/").replace(".jpg", ".png")
            for image_filename in image_filenames
        ]
        panoptic_classes = load_from_json(os.path.join(basedir, "panoptic_classes.json"))
        stuff_classes = panoptic_classes["stuff"]
        thing_classes = panoptic_classes["thing"]
        semantics = Semantics(
            stuff_classes=stuff_classes,
            stuff_filenames=stuff_filenames,
            thing_classes=thing_classes,
            thing_filenames=thing_filenames,
        )

    # --- possibly transform ---
    # TODO:

    # -- set the bounding box ---
    # TODO:

    # Possibly include the sparse point cloud from COLMAP in the dataset inputs.
    # NOTE(ethan): this will be common across the different splits.
    point_cloud = PointCloud()
    if include_point_cloud:
        points_3d = read_pointsTD_binary(os.path.join(basedir, "colmap", "points3D.bin"))
        xyz = torch.tensor(np.array([p_value.xyz for p_id, p_value in points_3d.items()]))
        rgb = torch.tensor(np.array([p_value.rgb for p_id, p_value in points_3d.items()]))
        point_cloud.xyz = xyz
        point_cloud.rgb = rgb

    dataset_inputs = DatasetInputs(
        image_filenames=image_filenames,
        downscale_factor=downscale_factor,
        intrinsics=intrinsics / downscale_factor,
        camera_to_world=camera_to_world,
        semantics=semantics,
        point_cloud=point_cloud,
    )
    return dataset_inputs
