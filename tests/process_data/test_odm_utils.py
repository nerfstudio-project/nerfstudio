"""Regression tests for OpenSfM camera intrinsics in ODM conversions."""

import json
from pathlib import Path

import pytest

from nerfstudio.process_data.odm_utils import cameras2nerfds


@pytest.mark.parametrize(
    "camera, expected",
    [
        ({"projection_type": "perspective", "focal": 0.5}, {"fl_x": 400, "cx": 399.5, "cy": 299.5}),
        ({"projection_type": "brown", "focal": 0.5, "c_x": 0.1, "c_y": -0.05}, {"cx": 479.5, "cy": 259.5}),
        (
            {"projection_type": "brown", "width": 600, "height": 800, "focal": 0.5, "c_x": 0.1, "c_y": -0.05},
            {"fl_x": 400, "fl_y": 400, "cx": 379.5, "cy": 359.5},
        ),
        ({"projection_type": "fisheye_opencv", "focal": 0.5, "k4": 0.004}, {"k4": 0.004}),
        ({"projection_type": "spherical"}, {"fl_x": 400, "fl_y": 600, "cx": 400, "cy": 300}),
    ],
)
def test_odm_intrinsics(tmp_path, camera, expected):
    cameras_path = tmp_path / "cameras.json"
    cameras_path.write_text(json.dumps({"camera": {"width": 800, "height": 600, **camera}}))
    reconstruction_path = tmp_path / "reconstruction.json"
    shot = {"rotation": [0, 0, 0], "translation": [0, 0, 0]}
    reconstruction_path.write_text(json.dumps([{"shots": {"sample.jpg": shot}}]))
    cameras2nerfds(
        {"sample": Path("sample.jpg")}, cameras_path, tmp_path / "shots.geojson", reconstruction_path, tmp_path
    )
    frame = json.loads((tmp_path / "transforms.json").read_text())["frames"][0]
    for key, value in expected.items():
        assert frame[key] == pytest.approx(value)
