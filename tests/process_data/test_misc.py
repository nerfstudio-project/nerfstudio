"""
Test misc data utils
"""

import os
import re
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat
from nerfstudio.process_data.process_data_utils import convert_video_to_images


def test_scalar_first_scalar_last_quaternions():
    """Many nerfstudio datasets use quaternions for pose rotations.  Some use
    scalar-first format and others use scalar-last.  However internally
    nerfstudio uses rotation *matrices* / projection matrices.  This test
    helps document differences in quaternion encodings & conversion to
    rotation matrices.

    FMI see call sites in nerfstudio where the various libraries are used. You
    might also consult this detailed reference:
     * "Why and How to Avoid the Flipped Quaternion Multiplication"
        Sommer et al.
        https://arxiv.org/abs/1801.07478
    """

    # Pick an arbitrary euler rotation; note that different communities
    # also use different rotation axes (e.g. Euler vs Tait-Bryan ...)
    rot = Rotation.from_euler("xyz", np.array([10.0, 20.0, 30.0]), degrees=True)

    # scalar-last
    xyzw = rot.as_quat()
    assert np.allclose(xyzw, np.array([0.03813458, 0.18930786, 0.23929834, 0.95154852]))

    # scalar-first
    wxyz = xyzw[[3, 0, 1, 2]]
    assert np.allclose(wxyz, np.array([0.95154852, 0.03813458, 0.18930786, 0.23929834]))

    # Expected Rotation matrix
    # fmt: off
    R_expected = np.array(
        [
            [ 0.81379768, -0.44096961,  0.37852231],
            [ 0.46984631,  0.88256412,  0.01802831],
            [-0.34202014,  0.16317591,  0.92541658]
        ]
    )
    # fmt: on

    # Record3D / scipy
    R = Rotation.from_quat(xyzw).as_matrix()
    assert np.allclose(R, R_expected)

    # Nuscenes / pyquaternion
    R = Quaternion(wxyz).rotation_matrix
    assert np.allclose(R, R_expected)

    # COLMAP
    # TODO(1480) use pycolmap
    # R = pycolmap.qvec_to_rotmat(wxyz)
    R = qvec2rotmat(wxyz)
    assert np.allclose(R, R_expected)


def test_process_video_conversion_with_seed(tmp_path: Path):
    """
    Test convert_video_to_images by creating a mock video and ensuring correct frame extraction with seed.
    """

    # Inner functions needed for the unit tests
    def create_mock_video(video_path: Path, frame_dir: Path, num_frames=10, frame_rate=1):
        """Creates a mock video from a series of frames using OpenCV."""

        first_frame = cv2.imread(str(frame_dir / "frame_0.png"))
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, frame_rate, (width, height))

        for i in range(num_frames):
            frame_path = frame_dir / f"frame_{i}.png"
            frame = cv2.imread(str(frame_path))
            out.write(frame)
        out.release()

    def extract_frame_numbers(ffmpeg_command: str):
        """Extracts the frame numbers from the ffmpeg command"""

        pattern = r"eq\(n\\,(\d+)\)"
        matches = re.findall(pattern, ffmpeg_command)
        frame_numbers = [int(match) for match in matches]
        return frame_numbers

    # Create a video directory with path video
    video_dir = tmp_path / "video"
    video_dir.mkdir(exist_ok=True)

    # Set parameters for mock video
    video_path = video_dir / "mock_video.mp4"
    num_frames = 10
    frame_height = 150
    frame_width = 100
    frame_rate = 1

    # Create the mock video
    for i in range(num_frames):
        img = Image.new("RGB", (frame_width, frame_height), (0, 0, 0))
        img.save(video_dir / f"frame_{i}.png")
    create_mock_video(video_path, video_dir, num_frames=num_frames, frame_rate=frame_rate)

    # Call convert_video_to_images
    image_output_dir = tmp_path / "extracted_images"
    num_frames_target = 5
    num_downscales = 1
    crop_factor = (0.0, 0.0, 0.0, 0.0)

    # Mock missing COLMAP and ffmpeg in the dev env
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(tmp_path / "mocked_bin") + f":{old_path}"
    (tmp_path / "mocked_bin").mkdir()
    (tmp_path / "mocked_bin" / "colmap").touch(mode=0o777)
    (tmp_path / "mocked_bin" / "ffmpeg").touch(mode=0o777)

    # Return value of 10 for the get_num_frames_in_video run_command call
    with mock.patch("nerfstudio.process_data.process_data_utils.run_command", return_value="10") as mock_run_func:
        summary_log, extracted_frame_count = convert_video_to_images(
            video_path=video_path,
            image_dir=image_output_dir,
            num_frames_target=num_frames_target,
            num_downscales=num_downscales,
            crop_factor=crop_factor,
            verbose=False,
            random_seed=42,
        )
        assert mock_run_func.call_count == 2, f"Expected 2 calls, but got {mock_run_func.call_count}"
        first_frames = extract_frame_numbers(mock_run_func.call_args[0][0])
        assert len(first_frames) == 5, f"Expected 5 frames, but got {len(first_frames)}"

        summary_log, extracted_frame_count = convert_video_to_images(
            video_path=video_path,
            image_dir=image_output_dir,
            num_frames_target=num_frames_target,
            num_downscales=num_downscales,
            crop_factor=crop_factor,
            verbose=False,
            random_seed=42,
        )

        assert mock_run_func.call_count == 4, f"Expected 4 total calls, but got {mock_run_func.call_count}"
        second_frames = extract_frame_numbers(mock_run_func.call_args[0][0])
        assert len(second_frames) == 5, f"Expected 5 frames, but got {len(first_frames)}"
        assert first_frames == second_frames

        summary_log, extracted_frame_count = convert_video_to_images(
            video_path=video_path,
            image_dir=image_output_dir,
            num_frames_target=num_frames_target,
            num_downscales=num_downscales,
            crop_factor=crop_factor,
            verbose=False,
            random_seed=52,
        )

        assert mock_run_func.call_count == 6, f"Expected 6 total calls, but got {mock_run_func.call_count}"
        third_frames = extract_frame_numbers(mock_run_func.call_args[0][0])
        assert len(third_frames) == 5, f"Expected 5 frames, but got {len(first_frames)}"
        assert first_frames != third_frames
