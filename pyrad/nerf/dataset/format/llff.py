# Copyright 2022 The Plenoptix Team. All rights reserved.
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
Code for loading the LLFF data.
"""


def load_llff_data(basedir, downscale_factor=1, split="train", include_point_cloud=False):
    """_summary_

    Args:
        basedir (_type_): _description_
        downscale_factor (float, optional): _description_. Defaults to 1.0.
        split (str, optional): _description_. Defaults to "train".
        include_point_cloud (bool): whether or not to include the point cloud
    """
    # pylint: disable=unused-argument
    raise NotImplementedError
