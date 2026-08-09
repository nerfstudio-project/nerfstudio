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

"""Tests for importing Metashape camera data."""

import json
from pathlib import Path

from nerfstudio.process_data.metashape_utils import metashape_to_json


def test_metashape_tangential_distortion_conversion(tmp_path: Path):
    xml_path = tmp_path / "cameras.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="frame">
        <resolution width="1000" height="800"/>
        <calibration>
          <f>500</f>
          <p1>0.001</p1>
          <p2>-0.002</p2>
        </calibration>
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" sensor_id="0" label="image">
        <transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform>
      </camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )

    metashape_to_json({"image": Path("images/image.jpg")}, xml_path, tmp_path)

    transforms = json.loads((tmp_path / "transforms.json").read_text(encoding="utf-8"))
    assert transforms["frames"][0]["p1"] == -0.002
    assert transforms["frames"][0]["p2"] == 0.001
