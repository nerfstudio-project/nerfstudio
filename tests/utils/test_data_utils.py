# Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

from unittest.mock import Mock

import numpy as np
import pytest
from packaging.version import Version
from PIL import Image

from nerfstudio.data.utils import data_utils


@pytest.mark.parametrize("pillow_version, uses_extents", [("12.2.0", True), ("12.1.0", False)])
def test_set_encoder_image(monkeypatch, pillow_version, uses_extents):
    image = Image.new("L", (2, 2))
    encoder = Mock()
    monkeypatch.setattr(data_utils, "_PILLOW_VERSION", Version(pillow_version))

    data_utils._set_encoder_image(encoder, image)

    if uses_extents:
        encoder.setimage.assert_called_once_with(image.im, (0, 0) + image.size)
    else:
        encoder.setimage.assert_called_once_with(image.im)


@pytest.mark.parametrize(
    "values",
    [
        np.array([[1, 2], [3, 4]], dtype=np.uint8),
        np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=np.uint8,
        ),
    ],
)
def test_pil_to_numpy(values):
    image = Image.fromarray(values)

    result = data_utils.pil_to_numpy(image)

    np.testing.assert_array_equal(result, values)
    assert result.flags.writeable
