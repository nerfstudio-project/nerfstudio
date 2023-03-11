"""
Process images test
"""
from pathlib import Path

import numpy as np
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
    for i in range(num_frames):
        frames[i + 1] = ColmapImage(i + 1, qvecs[i], tvecs[i], 1, f"image_{i}.png", [], [])
        Image.new("RGB", (width, height)).save(tmp_path / "images" / f"image_{i}.png")
    write_images_binary(frames, sparse_path / "images.bin")

    # Run ProcessImages
    cmd = ProcessImages(tmp_path / "images", tmp_path / "nerfstudio", colmap_model_path=sparse_path, skip_colmap=True)
    cmd.main()

    assert (tmp_path / "nerfstudio" / "transforms.json").exists()
    parser = NerfstudioDataParserConfig(
        data=tmp_path / "nerfstudio", downscale_factor=None, orientation_method="none", center_method="none"
    ).setup()
    outputs = parser.get_dataparser_outputs("train")
    assert len(outputs.image_filenames) == 9
