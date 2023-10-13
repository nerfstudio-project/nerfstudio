"""
Regional Nerfacto DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union
import os.path as osp

import torch
import json

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from regional_nerfacto.utils.dino_dataloader import DinoDataloader
from pathlib import Path

import numpy as np
import math

import requests
from io import BytesIO
from PIL import Image
from math import pi, sin, log, tan, floor, cos


@dataclass
class RNerfDataManagerConfig(VanillaDataManagerConfig):
    """RNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: RNerfDataManager)


class RNerfDataManager(VanillaDataManager):
    """RNerf DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RNerfDataManagerConfig

    def __init__(
        self,
        config: RNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        enu2nerf, nerf2enu, enu2nerf_points, nerf2enu_points, osm_image, osm_scale = self.create_enu_mapping()

        self.enu2nerf = enu2nerf
        self.nerf2enu = nerf2enu
        self.enu2nerf_points = enu2nerf_points
        self.nerf2enu_points = nerf2enu_points
        self.osm_image = osm_image
        self.osm_scale = osm_scale

        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)

        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality
        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path,
        )
        torch.cuda.empty_cache()

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        batch["dino"] = self.dino_dataloader(ray_indices)
        return ray_bundle, batch

    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None

    def create_enu_mapping(self):
        transforms = self._find_transform(self.train_dataparser_outputs.image_filenames[0])

        def lat_lon_to_tile_xy(lat, lon, zoom):
            n = 2.0 ** zoom
            xtile = (lon + 180.0) / 360.0 * n
            ytile = (1.0 - log(math.tan(lat * pi / 180.0) + 1.0 / math.cos(lat * pi / 180.0)) / pi) / 2.0 * n
            return int(xtile), int(ytile)

        def osm_scale_in_meters_per_pixel(latitude_degrees, zoom):
            earth_circumference = 40075016.686  # in meters
            tile_size = 256  # OSM tile size in pixels
            latitude_radians = latitude_degrees * pi / 180.0
            scale = (earth_circumference * math.cos(latitude_radians)) / (tile_size * 2**zoom)
            return scale

        def get_osm_tile(lat, lon, zoom=15):
            xtile, ytile = lat_lon_to_tile_xy(lat, lon, zoom)
            # print(xtile, ytile)
            # url = f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
            # print(url)
            # response = requests.get(url)
            tile_path = f"regional_nerf/regional_nerfacto/map_tiles/{xtile}_{ytile}.png"
            image = Image.open(tile_path).convert("RGB")
            return image

        if transforms is not None:
            meta = json.load(open(transforms, "r"))
            if "scale" in meta.keys():
                transform_scale = meta["scale"]
            else:
                transform_scale = 1.0
            if "lat" in meta.keys() and "lon" in meta.keys():
                rclat = np.radians(meta["lat"])
                rclng = np.radians(meta["lon"])
                rot_ECEF2ENUV = np.array([[-math.sin(rclng),                math.cos(rclng),                              0],
                                [-math.sin(rclat)*math.cos(rclng), -math.sin(rclat)*math.sin(rclng), math.cos(rclat)],
                                [math.cos(rclat)*math.cos(rclng),  math.cos(rclat)*math.sin(rclng),  math.sin(rclat)]])
            
                osm_image = get_osm_tile(meta["lat"], meta["lon"], zoom=15)
                osm_image = np.array(osm_image)
                osm_scale = osm_scale_in_meters_per_pixel(meta["lat"], zoom=15)
            else:
                rot_ECEF2ENUV = np.eye(3)
                osm_image = np.zeros((100, 100, 3))
                osm_scale = 1.0
            osm_image = torch.from_numpy(osm_image).permute(2, 0, 1).float() / 255.0
            osm_image = osm_image.to(self.device)

        dataparser_scale = self.train_dataparser_outputs.dataparser_scale
        dataparser_transform = self.train_dataparser_outputs.dataparser_transform   # 3 x 4

        dataparser_rotation = dataparser_transform[:3, :3].to(self.device)
        dataparser_translation = dataparser_transform[:3, 3].to(self.device)

        dataparser_transform = torch.eye(4, device=self.device)
        dataparser_transform[:3, :3] = dataparser_rotation
        dataparser_transform[:3, 3] = dataparser_translation

        dataparser_transform_inv = torch.eye(4, device=self.device)
        dataparser_transform_inv[:3, :3] = dataparser_rotation.T
        dataparser_transform_inv[:3, 3] = -dataparser_rotation.T @ dataparser_translation
        
        # Camera poses in transformed reference frame
        camera_poses = self.train_dataparser_outputs.cameras.camera_to_worlds   # N x 3 x 4 
        camera_poses = torch.cat(
                [camera_poses, torch.tensor([[[0, 0, 0, 1]]], dtype=camera_poses.dtype).repeat(len(camera_poses), 1, 1)], 1
            )

        # print("dataparser_transform", dataparser_transform)
        # print("dataparser_transform_inv", dataparser_transform_inv)
        # print("dataparser_scale", dataparser_scale)
        # print("transform_scale", transform_scale)

        def enu2nerf(poses):
            """
            poses: N x 4 x 4
            """
            # Scale to match the scale of the dataparser
            poses[..., :3, 3] *= transform_scale
            # Transform to the dataparser reference frame

            poses = dataparser_transform @ poses

            poses[..., :3, 3] *= dataparser_scale
            return poses

        def nerf2enu(poses):
            """
            poses: N x 4 x 4
            """
            # Scale to match the scale of the dataparser
            poses[..., :3, 3] /= dataparser_scale
            # Transform to the dataparser reference frame
            poses = dataparser_transform_inv @ poses
            poses[..., :3, 3] /= transform_scale
            return poses

        def enu2nerf_points(points):
            """
            points: N x 3
            """
            # Convert to pose via identity rotation
            points *= transform_scale
            points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            points = dataparser_transform @ points[..., None]
            points = points[..., :3, 0]
            points *= dataparser_scale
            return points

        def nerf2enu_points(points):
            """
            points: ... x 3
            """
            # Convert to pose via identity rotation
            points /= dataparser_scale
            points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            points = dataparser_transform_inv @ points[..., None]
            points = points[..., :3, 0]
            
            points /= transform_scale
            return points

        return enu2nerf, nerf2enu, enu2nerf_points, nerf2enu_points, osm_image, osm_scale



        