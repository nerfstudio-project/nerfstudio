# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
"""Depth data parser."""
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Optional, Type


from nerfstudio.data.dataparsers import nerfstudio_dataparser
from nerfstudio.utils.io import load_from_json


@dataclass
class DepthDataParserConfig(nerfstudio_dataparser.NerfstudioDataParserConfig):
    """Depth dataset config"""

    _target: Type = field(default_factory=lambda: DepthDataParser)
    """target class to instantiate"""


@dataclass
class DepthDataParser(nerfstudio_dataparser.Nerfstudio):
    """Nerfstudio-based depth data parser"""

    config: DepthDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        dataparser_outputs = super()._generate_dataparser_outputs(split)
        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        depth_paths = []
        for frame in meta["frames"]:
            depth_file = self._get_fname(PurePath(frame["depth_file_path"]), data_dir, downsample_folder_prefix='depths_')
            depth_paths.append(depth_file)
        indices = nerfstudio_dataparser.get_indices_for_split(split,
                                                              len(depth_paths),
                                                              self.config.train_split_percentage)

        if len(indices) != len(dataparser_outputs.image_filenames):
            raise ValueError("Invalid number of depth images provided.")

        dataparser_outputs.metadata = {
            "depth_paths": [depth_paths[i] for i in indices],
        }
        return dataparser_outputs
