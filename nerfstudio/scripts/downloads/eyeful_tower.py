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


import collections
import copy
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import awscli.clidriver
import numpy as np
import tyro

from nerfstudio.scripts.downloads.utils import DatasetDownload

eyefultower_downloads = [
    "all",
    "apartment",
    "kitchen",
    "office1a",
    "office1b",
    "office2",
    "office_view1",
    "office_view2",
    "riverview",
    "seating_area",
    "table",
    "workshop",
]


@dataclass
class EyefulTowerResolutionMetadata:
    folder_name: str
    width: int
    height: int
    extension: str


eyefultower_resolutions = {
    "all": None,
    "jpeg_2k": EyefulTowerResolutionMetadata("images-jpeg-2k", 1368, 2048, "jpg"),
    "jpeg_4k": EyefulTowerResolutionMetadata("images-jpeg-4k", 2736, 4096, "jpg"),
    "jpeg_8k": EyefulTowerResolutionMetadata("images-jpeg", 5784, 8660, "jpg"),
    "exr_2k": EyefulTowerResolutionMetadata("images-2k", 1368, 2048, "exr"),
}

if TYPE_CHECKING:
    EyefulTowerCaptureName = str
    EyefulTowerResolution = str
else:
    EyefulTowerCaptureName = tyro.extras.literal_type_from_choices(eyefultower_downloads)
    EyefulTowerResolution = tyro.extras.literal_type_from_choices(eyefultower_resolutions.keys())


@dataclass
class EyefulTowerDownload(DatasetDownload):
    """Download the EyefulTower dataset.

    Use the --help flag with the `eyefultower` subcommand to see all available datasets.
    Find more information about the dataset at https://github.com/facebookresearch/EyefulTower.
    """

    capture_name: Tuple[EyefulTowerCaptureName, ...] = ()
    resolution_name: Tuple[EyefulTowerResolution, ...] = ()

    @staticmethod
    def scale_metashape_transform(xml_tree: ET.ElementTree, target_width: int, target_height: int):
        transformed = copy.deepcopy(xml_tree)

        root = transformed.getroot()
        assert len(root) == 1
        chunk = root[0]
        sensors = chunk.find("sensors")
        assert sensors is not None

        for sensor in sensors:
            resolution = sensor.find("resolution")
            assert resolution is not None, "Resolution not found in EyefulTower camera.xml"
            original_width = int(resolution.get("width"))  # type: ignore
            original_height = int(resolution.get("height"))  # type: ignore

            if original_width > original_height:
                target_width, target_height = max(target_width, target_height), min(target_width, target_height)
            else:
                target_height, target_width = max(target_width, target_height), min(target_width, target_height)

            resolution.set("width", str(target_width))
            resolution.set("height", str(target_height))

            calib = sensor.find("calibration")
            assert calib is not None, "Calibration not found in EyefulTower sensor"

            calib_resolution = calib.find("resolution")
            assert calib_resolution is not None
            calib_resolution.set("width", str(target_width))
            calib_resolution.set("height", str(target_height))

            # Compute each scale individually and average for better rounding
            x_scale = target_width / original_width
            y_scale = target_height / original_height
            scale = (x_scale + y_scale) / 2.0

            f = calib.find("f")
            assert f is not None and f.text is not None, "f not found in calib"
            f.text = str(float(f.text) * scale)

            cx = calib.find("cx")
            assert cx is not None and cx.text is not None, "cx not found in calib"
            cx.text = str(float(cx.text) * x_scale)

            cy = calib.find("cy")
            assert cy is not None and cy.text is not None, "cy not found in calib"
            cy.text = str(float(cy.text) * y_scale)

            # TODO: Maybe update pixel_width / pixel_height / focal_length / layer_index?

        return transformed

    def convert_cameras_to_nerfstudio_transforms(
        self, cameras: dict, splits: dict, target_width: int, target_height: int, extension: str
    ):
        output = {}

        distortion_models = [c["distortionModel"] for c in cameras["KRT"]]
        distortion_model = list(set(distortion_models))
        assert len(distortion_model) == 1
        distortion_model = distortion_model[0]
        if distortion_model == "RadialAndTangential":
            output["camera_model"] = "OPENCV"
        elif distortion_model == "Fisheye":
            output["camera_model"] = "OPENCV_FISHEYE"
        else:
            raise NotImplementedError(f"Camera model {distortion_model} not implemented")

        split_sets = {k: set(v) for k, v in splits.items()}

        frames = []
        split_filenames = collections.defaultdict(list)
        for camera in cameras["KRT"]:
            frame = {}
            # TODO EXR
            frame["file_path"] = camera["cameraId"] + f".{extension}"
            for split in split_sets:
                if camera["cameraId"] in split_sets[split]:
                    split_filenames[split].append(frame["file_path"])

            original_width = camera["width"]
            original_height = camera["height"]
            if original_width > original_height:
                target_width, target_height = max(target_width, target_height), min(target_width, target_height)
            else:
                target_height, target_width = max(target_width, target_height), min(target_width, target_height)
            x_scale = target_width / original_width
            y_scale = target_height / original_height

            frame["w"] = target_width
            frame["h"] = target_height
            K = np.array(camera["K"]).T  # Data stored as column-major
            frame["fl_x"] = K[0][0] * x_scale
            frame["fl_y"] = K[1][1] * y_scale
            frame["cx"] = K[0][2] * x_scale
            frame["cy"] = K[1][2] * y_scale

            if distortion_model == "RadialAndTangential":
                # pinhole: [k1, k2, p1, p2, k3]
                frame["k1"] = camera["distortion"][0]
                frame["k2"] = camera["distortion"][1]
                frame["k3"] = camera["distortion"][4]
                frame["k4"] = 0.0
                frame["p1"] = camera["distortion"][2]
                frame["p2"] = camera["distortion"][3]
            elif distortion_model == "Fisheye":
                # fisheye: [k1, k2, k3, _, _, _, p1, p2]
                frame["k1"] = camera["distortion"][0]
                frame["k2"] = camera["distortion"][1]
                frame["k3"] = camera["distortion"][2]
                frame["p1"] = camera["distortion"][6]
                frame["p2"] = camera["distortion"][7]
            else:
                raise NotImplementedError("This shouldn't happen")

            T = np.array(camera["T"]).T  # Data stored as column-major
            T = np.linalg.inv(T)
            T = T[[2, 0, 1, 3], :]
            T[:, 1:3] *= -1
            frame["transform_matrix"] = T.tolist()

            frames.append(frame)

        frames = sorted(frames, key=lambda f: f["file_path"])

        output["frames"] = frames
        output["train_filenames"] = split_filenames["train"]
        output["val_filenames"] = split_filenames["test"]
        return output

    def subsample_nerfstudio_transforms(self, transforms: dict, n: int):
        target = min(len(transforms["frames"]), n)
        indices = np.round(np.linspace(0, len(transforms["frames"]) - 1, target)).astype(int)

        frames = []
        for i in indices:
            frames.append(transforms["frames"][i])

        output = copy.deepcopy(transforms)
        output["frames"] = frames

        # Remove the unused files from the splits
        filenames = {f["file_path"] for f in frames}
        for key in ["train_filenames", "val_filenames"]:
            output[key] = sorted(list(set(transforms[key]) & filenames))

        return output

    def download(self, save_dir: Path):
        if len(self.capture_name) == 0:
            self.capture_name = ("riverview",)
            print(
                f"No capture specified, using {self.capture_name} by default.",
                "Add `--help` to this command to see all available captures.",
            )

        if len(self.resolution_name) == 0:
            self.resolution_name = ("jpeg_2k",)
            print(
                f"No resolution specified, using {self.resolution_name} by default.",
                "Add `--help` to this command to see all available resolutions.",
            )

        captures = set()
        for capture in self.capture_name:
            if capture == "all":
                captures.update([c for c in eyefultower_downloads if c != "all"])
            else:
                captures.add(capture)
        captures = sorted(captures)
        if len(captures) == 0:
            print("WARNING: No EyefulTower captures specified. Nothing will be downloaded.")

        resolutions = set()
        for resolution in self.resolution_name:
            if resolution == "all":
                resolutions.update([r for r in eyefultower_resolutions.keys() if r != "all"])
            else:
                resolutions.add(resolution)
        resolutions = sorted(resolutions)
        if len(resolutions) == 0:
            print("WARNING: No EyefulTower resolutions specified. Nothing will be downloaded.")

        driver = awscli.clidriver.create_clidriver()

        for i, capture in enumerate(captures):
            base_url = f"s3://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/EyefulTower/{capture}/"
            output_path = save_dir / "eyefultower" / capture
            includes = []
            for resolution in resolutions:
                includes.extend(["--include", f"{eyefultower_resolutions[resolution].folder_name}/*"])
            command = (
                ["s3", "sync", "--no-sign-request", "--only-show-errors", "--exclude", "images*/*"]
                + includes
                + [base_url, str(output_path)]
            )
            print(f"[EyefulTower Capture {i+1: >2d}/{len(captures)}]: '{capture}'")
            print(
                f"    Downloading resolutions {resolutions}",
                f"to '{output_path.resolve()}' with command `aws {' '.join(command)}` ...",
                end=" ",
                flush=True,
            )
            driver.main(command)
            print("done!")

            # After downloading, we'll insert an appropriate cameras.xml file into each directory
            # It's quick enough that we can just redo it every time this is called, regardless
            # of whether new data is downloaded.
            xml_input_path = output_path / "cameras.xml"
            if not xml_input_path.exists:
                print("    WARNING: cameras.xml not found. Scaled cameras.xml will not be generated.")
            else:
                tree = ET.parse(output_path / "cameras.xml")

                for resolution in resolutions:
                    metadata = eyefultower_resolutions[resolution]
                    xml_output_path = output_path / metadata.folder_name / "cameras.xml"
                    print(
                        f"    Generating cameras.xml for '{resolution}' to '{xml_output_path.resolve()}' ... ",
                        end=" ",
                        flush=True,
                    )
                    scaled_tree = self.scale_metashape_transform(tree, metadata.width, metadata.height)
                    scaled_tree.write(xml_output_path)
                    print("done!")

            json_input_path = output_path / "cameras.json"
            splits_input_path = output_path / "splits.json"
            if not json_input_path.exists:
                print("    WARNING: cameras.json not found. transforms.json will not be generated.")
            elif not splits_input_path.exists:
                print("    WARNING: splits.json not found. transforms.json will not be generated.")
            else:
                with open(json_input_path, "r") as f:
                    cameras = json.load(f)

                with open(splits_input_path, "r") as f:
                    splits = json.load(f)

                for resolution in resolutions:
                    metadata = eyefultower_resolutions[resolution]
                    json_output_path = output_path / metadata.folder_name / "transforms.json"
                    print(
                        f"    Generating transforms.json for '{resolution}' to '{json_output_path.resolve()}' ... ",
                        end=" ",
                        flush=True,
                    )
                    transforms = self.convert_cameras_to_nerfstudio_transforms(
                        cameras, splits, metadata.width, metadata.height, metadata.extension
                    )

                    with open(json_output_path, "w", encoding="utf8") as f:
                        json.dump(transforms, f, indent=4)

                    for count, name in [
                        (300, "transforms_300.json"),
                        (int(len(cameras["KRT"]) // 2), "transforms_half.json"),
                    ]:
                        subsampled = self.subsample_nerfstudio_transforms(transforms, count)
                        with open(json_output_path.with_name(name), "w", encoding="utf8") as f:
                            json.dump(subsampled, f, indent=4)

                    print("done!")
