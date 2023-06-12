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

"""
Depth dataset.
"""

import os
from typing import Dict

import numpy as np

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from PIL import Image
import torch
from rich.progress import Console, track
from pathlib import Path
import json




class DepthDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        #if there are no depth images than we want to generate them all with zoe depth
        if len(dataparser_outputs.image_filenames) > 0 and ("depth_filenames" not in dataparser_outputs.metadata.keys() or dataparser_outputs.metadata["depth_filenames"] is None):
            depth_paths = []
            tranforms = self._find_transform(dataparser_outputs.image_filenames[0])
            data = dataparser_outputs.image_filenames[0].parent
            meta = json.load(open(tranforms, "r"))
            frames = meta['frames']
            filenames = [data / frames[j]['file_path'].split('/')[-1] for j in range(len(frames))]
            os.makedirs(dataparser_outputs.image_filenames[0].parent / "depth", exist_ok=True)
            repo = "isl-org/ZoeDepth"
            self.zoe = torch.compile(torch.hub.load(repo, "ZoeD_NK", pretrained=True).cuda())

            for i in track(range(len(filenames)), description="Generating depth images"):
                image_filename = filenames[i]
                pil_image = Image.open(image_filename)
                image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
                if len(image.shape) == 2:
                    image = image[:, :, None].repeat(3, axis=2)
                image = torch.from_numpy(image.astype("float32") / 255.0)

                #BAD: FIX BY FINDING DEVICE
                with torch.no_grad():
                    image = torch.permute(image, (2, 0, 1)).unsqueeze(0).cuda()
                    depth_numpy = self.zoe.infer(image).squeeze().unsqueeze(-1).cpu().numpy()
                    
                depth_paths.append(image_filename.parent / "depth" / f"depth{i}.npy")
                meta['frames'][i]['depth_file_path'] = str(Path(meta['frames'][i]['file_path']).parent / "depth" / f"depth{i}.npy")
                np.save(depth_paths[-1], depth_numpy)

            json.dump(meta, open(tranforms, "w"))


            dataparser_outputs.metadata["depth_filenames"] = depth_paths
            self.metadata["depth_filenames"] = depth_paths
            dataparser_outputs.metadata["depth_unit_scale_factor"] = 1.0
            self.metadata["depth_unit_scale_factor"] = 1.0
            
        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )

        return {"depth_image": depth_image}

    def _find_transform(self, image_path : Path) -> Path:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        raise FileNotFoundError("Could not find transforms.json in any parent directory of image_path")
