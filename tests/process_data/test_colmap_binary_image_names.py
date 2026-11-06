from pathlib import Path

import numpy as np

from nerfstudio.data.utils.colmap_parsing_utils import Image, read_images_binary, write_images_binary


def test_multibyte_image_name_roundtrip(tmp_path: Path):
    image = Image(
        id=7,
        qvec=np.array([1, 0, 0, 0]),
        tvec=np.zeros(3),
        camera_id=3,
        name="室内/café.png",
        xys=np.array([[2.5, 1.0]]),
        point3D_ids=np.array([17]),
    )
    path = tmp_path / "images.bin"
    write_images_binary({image.id: image}, path)
    restored = read_images_binary(path)[image.id]

    assert restored.name == image.name
    np.testing.assert_array_equal(restored.xys, image.xys)
    np.testing.assert_array_equal(restored.point3D_ids, image.point3D_ids)
