from pathlib import Path
from unittest import mock

import pytest

from nerfstudio.process_data.process_data_utils import convert_video_to_images


@pytest.mark.parametrize("empty_video", [False, True])
def test_invalid_video_preserves_existing_frames(tmp_path: Path, empty_video: bool):
    image_dir = tmp_path / "images"
    old_frames = [directory / "frame_00001.png" for directory in (image_dir, tmp_path / "images_2")]
    for frame in old_frames:
        frame.parent.mkdir()
        frame.write_bytes(b"previous frame")
    video = tmp_path / "capture.mp4"
    if empty_video:
        video.touch()

    with mock.patch("nerfstudio.process_data.process_data_utils.get_num_frames_in_video", return_value=0):
        with pytest.raises(SystemExit):
            convert_video_to_images(video, image_dir, num_frames_target=3, num_downscales=1)

    assert all(frame.read_bytes() == b"previous frame" for frame in old_frames)
