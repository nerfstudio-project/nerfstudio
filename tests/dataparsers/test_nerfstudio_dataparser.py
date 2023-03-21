"""
Nerfstudio dataparser
"""
# pylint: disable=all
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pytest import fixture


@fixture
def mocked_dataset(tmp_path: Path):
    """Mocked dataset with transforms"""
    (tmp_path / "images_4").mkdir()
    frames = []
    for i in range(10):
        Image.new("RGB", (100, 150)).save(tmp_path / "images_4" / f"img_{i}.png")
        frames.append(
            {
                "file_path": f"img_{i}.png",
                "transform_matrix": np.eye(4).tolist(),
            }
        )
    with (tmp_path / "transforms.json").open("w+", encoding="utf8") as f:
        json.dump({"fl_x": 2, "fl_y": 3, "cx": 4, "cy": 5, "h": 150, "w": 100, "frames": frames}, f)
    return tmp_path


@pytest.mark.parametrize("orientation_method", ["up", "none", "pca"])
def test_nerfstudio_dataparser_no_filelist(mocked_dataset, orientation_method):
    """Tests basic load"""
    assert (mocked_dataset / "images_4").exists()
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
        DataparserOutputs,
        Nerfstudio,
        NerfstudioDataParserConfig,
    )

    parser: Nerfstudio = NerfstudioDataParserConfig(
        data=mocked_dataset,
        downscale_factor=4,
        orientation_method=orientation_method,
        center_method="none",
        auto_scale_poses=False,
    ).setup()

    paths = set()
    for split in ("train", "test", "val"):
        out = parser.get_dataparser_outputs(split)
        assert isinstance(out, DataparserOutputs)
        assert len(out.image_filenames) > 0
        paths.update(out.image_filenames)
    train_files = set(parser.get_dataparser_outputs("train").image_filenames)
    assert len(train_files.intersection(parser.get_dataparser_outputs("val").image_filenames)) == 0
    assert len(train_files.intersection(parser.get_dataparser_outputs("test").image_filenames)) == 0
    assert len(paths) == 10


def test_nerfstudio_dataparser_split_filelist(mocked_dataset):
    """Tests basic load"""
    assert (mocked_dataset / "images_4").exists()
    with open(mocked_dataset / "transforms.json", "r+") as f:
        data = json.load(f)
        data["train_filenames"] = ["img_0.png", "img_1.png"]
        data["val_filenames"] = ["img_2.png", "img_3.png"]
        data["test_filenames"] = ["img_4.png", "img_5.png"]
        f.seek(0)
        f.truncate(0)
        json.dump(data, f)

    from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
        DataparserOutputs,
        Nerfstudio,
        NerfstudioDataParserConfig,
    )

    parser: Nerfstudio = NerfstudioDataParserConfig(
        data=mocked_dataset,
        downscale_factor=4,
        orientation_method="none",
        center_method="none",
        auto_scale_poses=False,
    ).setup()

    assert parser.get_dataparser_outputs("train").image_filenames == [
        mocked_dataset / "images_4/img_0.png",
        mocked_dataset / "images_4/img_1.png",
    ]
    assert parser.get_dataparser_outputs("val").image_filenames == [
        mocked_dataset / "images_4/img_2.png",
        mocked_dataset / "images_4/img_3.png",
    ]
    assert parser.get_dataparser_outputs("test").image_filenames == [
        mocked_dataset / "images_4/img_4.png",
        mocked_dataset / "images_4/img_5.png",
    ]
