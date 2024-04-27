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

from collections import OrderedDict
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from nerfstudio.scripts.exporter import ExportGaussianSplat


def test_export_gaussian_splat_write_ply(tmp_path: Path):
    filename = str(tmp_path / "test_export_gaussian_splat_write_ply.ply")
    count = 10
    colors_original = np.random.randint(0, 255, size=(count,), dtype=np.uint8)
    x_original = np.random.rand(count).astype(np.float32)
    y_original = np.random.rand(count).astype(np.float32)
    z_original = np.random.rand(count).astype(np.float64)  # should convert to float32

    map_to_tensors: OrderedDict[str, np.ndarray] = OrderedDict(
        [
            ("colors", colors_original),
            ("x", x_original),
            ("y", y_original),
            ("z", z_original),
        ]
    )

    ExportGaussianSplat.write_ply(filename, count, map_to_tensors)

    # Read the PLY file
    pcd = o3d.t.io.read_point_cloud(filename)
    positions = pcd.point.positions.numpy()
    colors = pcd.point.colors.numpy()

    # Assertions
    # Check if the file exists and has the correct number of points
    assert len(positions) == count, "Mismatch in the number of points written and read."

    # Check colors
    np.testing.assert_array_equal(colors_original, colors[:, 0], "Colors do not match.")

    # Check x, y, and z coordinates with epsilon tollerence
    epsilon = 1e-5  # Adjust based on expected precision
    np.testing.assert_allclose(x_original, positions[:, 0], atol=epsilon, err_msg="X coordinates do not match.")
    np.testing.assert_allclose(y_original, positions[:, 1], atol=epsilon, err_msg="Y coordinates do not match.")
    np.testing.assert_allclose(
        z_original.astype(np.float32), positions[:, 2], atol=epsilon, err_msg="Z coordinates do not match."
    )


def test_export_gaussian_splat_write_ply_mismatched_count(tmp_path: Path):
    filename = str(tmp_path / "test_export_gaussian_splat_write_ply_mismatched_count.ply")
    count = 10
    map_to_tensors: OrderedDict[str, np.ndarray] = OrderedDict(
        [
            ("x", np.random.rand(count + 1).astype(np.float32)),
            ("y", np.random.rand(count).astype(np.float32)),
            ("z", np.random.rand(count - 1).astype(np.float32)),
        ]
    )
    with pytest.raises(ValueError):
        ExportGaussianSplat.write_ply(filename, count, map_to_tensors)


if __name__ == "__main__":
    # Run the test
    test_export_gaussian_splat_write_ply(Path("."))
    test_export_gaussian_splat_write_ply_mismatched_count(Path("."))
