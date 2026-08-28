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

import numpy as np
import pytest
from PIL import Image

from nerfstudio.data.utils.data_utils import pil_to_numpy


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

    result = pil_to_numpy(image)

    np.testing.assert_array_equal(result, values)
    assert result.flags.writeable
