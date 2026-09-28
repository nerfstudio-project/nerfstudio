"""COLMAP images.txt names can contain spaces."""

from pathlib import Path

from nerfstudio.data.utils.colmap_parsing_utils import read_images_text


def test_image_name_keeps_spaces(tmp_path: Path):
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "\n".join(
            [
                "# Image list with two lines of data per image:",
                "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 my photo.jpg",
                "1.5 2.5 7",
                "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 plain.jpg",
                "0.0 0.0 -1",
                "",
            ]
        )
    )

    images = read_images_text(images_txt)

    assert images[1].name == "my photo.jpg"
    assert images[1].point3D_ids.tolist() == [7]
    assert images[1].xys.tolist() == [[1.5, 2.5]]
    assert images[2].name == "plain.jpg"
    assert images[2].point3D_ids.tolist() == [-1]
