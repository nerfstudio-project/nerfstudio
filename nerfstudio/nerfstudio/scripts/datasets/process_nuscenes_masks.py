# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import cv2
import numpy as np
import tyro
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from tqdm import tqdm


@dataclass
class ProcessNuScenesMasks:
    """Use cuboid detections to render masks for dynamic objects."""

    data_dir: Path
    """Path to NuScenes dataset."""
    output_dir: Path
    """Path to the output directory."""
    version: Literal["v1.0-mini", "v1.0-trainval"] = "v1.0-mini"
    """Which version of the dataset to process."""
    velocity_thresh: float = 0.75
    """Minimum speed for object to be considered dynamic."""
    cameras: Tuple[Literal["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"], ...] = (
        "FRONT",
        "FRONT_LEFT",
        "FRONT_RIGHT",
        "BACK",
        "BACK_LEFT",
        "BACK_RIGHT",
    )
    """Which cameras to use."""

    verbose: bool = False
    """If True, print extra logging and visualize images on screen."""

    def main(self) -> None:
        """Generate NuScenes dynamic object masks."""

        nusc = NuScenesDatabase(
            version=self.version,
            dataroot=str(self.data_dir.absolute()),
            verbose=self.verbose,
        )
        cameras = ["CAM_" + camera for camera in self.cameras]

        for camera in cameras:
            (self.output_dir / "masks" / camera).mkdir(parents=True, exist_ok=True)

        # get samples for scene
        samples = [samp for samp in nusc.sample]

        # sort by timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        # get which instances are moving in any frame (these are what we mask)
        instances = nusc.instance
        for instance in instances:
            is_dynamic = False
            ann_token = instance["first_annotation_token"]
            while ann_token:
                velocity = nusc.box_velocity(ann_token)
                if not np.linalg.norm(velocity) < self.velocity_thresh:
                    is_dynamic = True
                    break
                ann_token = nusc.get("sample_annotation", ann_token)["next"]
            instance["is_dynamic"] = is_dynamic
        instances_is_dynamic = {instance["token"]: instance["is_dynamic"] for instance in instances}

        for sample in tqdm(samples):
            viz = []
            for camera in cameras:
                camera_data = nusc.get("sample_data", sample["data"][camera])
                calibrated_sensor = nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
                intrinsics = np.array(calibrated_sensor["camera_intrinsic"])

                _, boxes, _ = nusc.get_sample_data(sample["data"][camera], box_vis_level=BoxVisibility.ANY)
                # TODO: BoxVisibility.ANY misses boxes that are partially behind the camera leading to missed masks
                # Instead use BoxVisibility.NONE and make sure to rasterize box faces correctly

                mask = np.ones((900, 1600), dtype=np.uint8)
                # If is backcam, mask the truck of the ego vehicle
                if camera == "CAM_BACK":
                    mask[-100:] = 0

                for box in boxes:
                    # Dont' mask out static objects (static in all frames)
                    instance_token = nusc.get("sample_annotation", box.token)["instance_token"]
                    if not instances_is_dynamic[instance_token]:
                        continue

                    # Project box to image plane and rasterize each face
                    corners_3d = box.corners()
                    corners = view_points(corners_3d, intrinsics, normalize=True)[:2, :]
                    corners = np.round(corners).astype(int).T

                    # Type ignores needed because fillPoly expects cv2.Mat
                    cv2.fillPoly(mask, [corners[[0, 1, 2, 3]]], 0)  # front # type: ignore
                    cv2.fillPoly(mask, [corners[[4, 5, 6, 7]]], 0)  # back # type: ignore
                    cv2.fillPoly(mask, [corners[[0, 1, 5, 4]]], 0)  # top # type: ignore
                    cv2.fillPoly(mask, [corners[[2, 3, 7, 6]]], 0)  # bottom # type: ignore
                    cv2.fillPoly(mask, [corners[[0, 3, 7, 4]]], 0)  # left # type: ignore
                    cv2.fillPoly(mask, [corners[[1, 2, 6, 5]]], 0)  # right # type: ignore

                maskname = os.path.split(camera_data["filename"])[1].replace("jpg", "png")
                cv2.imwrite(
                    str(self.output_dir / "masks" / camera / maskname),
                    mask * 255,  # type: ignore
                )

                if self.verbose:
                    img = cv2.imread(str(self.data_dir / camera_data["filename"]))
                    mask = ~mask.astype(bool)
                    img[mask, :] = img[mask, :] - np.minimum(img[mask, :], 100)
                    viz.append(img)

            if self.verbose:
                if len(viz) == 6:
                    viz = np.vstack((np.hstack(viz[:3]), np.hstack(viz[3:])))
                    viz = cv2.resize(viz, (int(1600 * 3 / 3), int(900 * 2 / 3)))
                elif len(viz) == 3:
                    viz = np.hstack(viz[:3])
                    viz = cv2.resize(viz, (int(1600 * 3 / 3), int(900 / 3)))
                elif len(viz) == 1:
                    viz = viz[0]
                else:
                    raise ValueError("Only support 1 or 3 or 6 cameras for viz")
                cv2.imshow("", viz)
                cv2.waitKey(1)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessNuScenesMasks).main()


if __name__ == "__main__":
    entrypoint()
