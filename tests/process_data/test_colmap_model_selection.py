"""Tests for selecting a COLMAP reconstruction when multiple models are present."""

import json
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from nerfstudio.data.utils.colmap_parsing_utils import (
    Camera,
    Image as ColmapImage,
    Point3D,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)
from nerfstudio.process_data.colmap_utils import find_best_colmap_model, get_colmap_model_image_count
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset


def _write_binary_model(model_dir: Path, image_names: List[str], num_points: int) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_binary(
        {1: Camera(1, "PINHOLE", 32, 32, np.array([25.0, 25.0, 16.0, 16.0]))},
        model_dir / "cameras.bin",
    )
    images = {
        image_id: ColmapImage(
            image_id,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([float(image_id), 0.0, 0.0]),
            1,
            image_name,
            np.empty((0, 2)),
            np.empty((0,), dtype=np.int64),
        )
        for image_id, image_name in enumerate(image_names, start=1)
    }
    write_images_binary(images, model_dir / "images.bin")
    points = {
        point_id: Point3D(
            point_id,
            np.array([float(point_id), 0.0, 1.0]),
            np.array([255, 0, 0], dtype=np.uint8),
            0.0,
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
        for point_id in range(1, num_points + 1)
    }
    write_points3D_binary(points, model_dir / "points3D.bin")


def _make_source_images(source_dir: Path, num_images: int) -> List[str]:
    source_dir.mkdir()
    image_names = [f"image_{i}.png" for i in range(num_images)]
    for image_name in image_names:
        Image.new("RGB", (32, 32)).save(source_dir / image_name)
    return image_names


def _mock_external_tools(tmp_path: Path, monkeypatch) -> None:
    mocked_bin = tmp_path / "bin"
    mocked_bin.mkdir()
    for executable in ("colmap", "ffmpeg"):
        path = mocked_bin / executable
        path.touch(mode=0o755)
    monkeypatch.setenv("PATH", f"{mocked_bin}{os.pathsep}{os.environ['PATH']}")


def test_find_best_colmap_model_prefers_registered_images(tmp_path: Path):
    sparse_dir = tmp_path / "sparse"
    _write_binary_model(sparse_dir / "0", ["a.png", "b.png", "c.png"], 100)
    _write_binary_model(sparse_dir / "5", ["a.png", "b.png", "c.png"], 101)
    _write_binary_model(sparse_dir / "2", ["a.png", "b.png", "c.png", "d.png"], 1)

    assert find_best_colmap_model(sparse_dir) == sparse_dir / "2"
    assert get_colmap_model_image_count(sparse_dir / "2") == 4


def test_find_best_colmap_model_breaks_exact_ties_by_model_path(tmp_path: Path):
    sparse_dir = tmp_path / "sparse"
    _write_binary_model(sparse_dir / "5", ["a.png", "b.png"], 3)
    _write_binary_model(sparse_dir / "2", ["a.png", "b.png"], 3)

    assert find_best_colmap_model(sparse_dir) == sparse_dir / "2"


def test_find_best_colmap_model_skips_invalid_models(tmp_path: Path):
    sparse_dir = tmp_path / "sparse"
    invalid_model = sparse_dir / "0"
    invalid_model.mkdir(parents=True)
    (invalid_model / "cameras.bin").write_bytes(b"corrupt")
    _write_binary_model(sparse_dir / "2", ["a.png", "b.png"], 1)

    assert find_best_colmap_model(sparse_dir) == sparse_dir / "2"


def test_images_converter_uses_best_direct_sparse_model(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    image_names = _make_source_images(source_dir, 8)

    sparse_dir = output_dir / "colmap" / "sparse"
    _write_binary_model(sparse_dir / "0", image_names[:3], 2)
    _write_binary_model(sparse_dir / "1", image_names, 4)
    _mock_external_tools(tmp_path, monkeypatch)

    converter = ImagesToNerfstudioDataset(data=source_dir, output_dir=output_dir, skip_colmap=True)
    converter.main()

    assert converter.colmap_model_path == Path("colmap/sparse/1")
    frames = json.loads((output_dir / "transforms.json").read_text())["frames"]
    assert len(frames) == len(image_names)


def test_images_converter_uses_best_hloc_nested_model(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    image_names = _make_source_images(source_dir, 8)

    sparse_dir = output_dir / "colmap" / "sparse"
    _write_binary_model(sparse_dir / "0", image_names[:3], 2)
    _write_binary_model(sparse_dir / "0" / "models" / "0", image_names, 4)
    _mock_external_tools(tmp_path, monkeypatch)

    converter = ImagesToNerfstudioDataset(data=source_dir, output_dir=output_dir, skip_colmap=True)
    converter.main()

    assert converter.colmap_model_path == Path("colmap/sparse/0/models/0")
    frames = json.loads((output_dir / "transforms.json").read_text())["frames"]
    assert len(frames) == len(image_names)
