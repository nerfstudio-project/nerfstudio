"""
Process images test
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.utils.colmap_parsing_utils import Camera
from nerfstudio.data.utils.colmap_parsing_utils import Image as ColmapImage
from nerfstudio.data.utils.colmap_parsing_utils import (
    write_cameras_binary,
    write_images_binary,
)
from scripts.process_data import ProcessImages


def test_process_images_skip_colmap(tmp_path: Path):
    """
    Test ns-process-data images
    """
    # Mock a colmap sparse model
    width = 100
    height = 150
    sparse_path = tmp_path / "sparse" / "0"
    sparse_path.mkdir(exist_ok=True, parents=True)
    (tmp_path / "images").mkdir(exist_ok=True, parents=True)
    write_cameras_binary(
        {1: Camera(1, "OPENCV", width, height, [110, 110, 50, 75, 0, 0, 0, 0, 0, 0])},
        sparse_path / "cameras.bin",
    )
    frames = {}
    num_frames = 10
    qvecs = np.random.uniform(size=(num_frames, 4))
    tvecs = np.random.uniform(size=(num_frames, 3))
    # original_poses = np.concatenate(
    #     (
    #         np.concatenate(
    #             (
    #                 np.stack(list(map(qvec2rotmat, qvecs))),
    #                 tvecs[:, :, None],
    #             ),
    #             -1,
    #         ),
    #         np.array([[[0, 0, 0, 1]]], dtype=qvecs.dtype).repeat(num_frames, 0),
    #     ),
    #     -2,
    # )
    for i in range(num_frames):
        frames[i + 1] = ColmapImage(i + 1, qvecs[i], tvecs[i], 1, f"image_{i}.png", [], [])
        Image.new("RGB", (width, height)).save(tmp_path / "images" / f"image_{i}.png")
    write_images_binary(frames, sparse_path / "images.bin")

    # Mock missing COLMAP and ffmpeg in the dev env
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(tmp_path / "mocked_bin") + f":{old_path}"
    (tmp_path / "mocked_bin").mkdir()
    (tmp_path / "mocked_bin" / "colmap").touch(mode=0o777)
    (tmp_path / "mocked_bin" / "ffmpeg").touch(mode=0o777)

    # Run ProcessImages
    cmd = ProcessImages(tmp_path / "images", tmp_path / "nerfstudio", colmap_model_path=sparse_path, skip_colmap=True)
    cmd.main()
    os.environ["PATH"] = old_path

    assert (tmp_path / "nerfstudio" / "transforms.json").exists()
    parser = NerfstudioDataParserConfig(
        data=tmp_path / "nerfstudio",
        downscale_factor=None,
        orientation_method="none",  # orientation_method,
        center_method="none",
        auto_scale_poses=False,
    ).setup()
    outputs = parser.get_dataparser_outputs("train")
    assert len(outputs.image_filenames) == 9
    assert torch.is_tensor(outputs.dataparser_transform)
    # TODO @jkulhanek: Add tests if the loaded poses are the same
