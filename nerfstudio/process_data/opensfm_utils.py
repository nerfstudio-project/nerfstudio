# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from rich.console import Console

from nerfstudio.process_data.colmap_utils import qvec2rotmat
from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)


def run_opensfm(
    image_dir: Path, opensfm_dir: Path, camera_model: CameraModel, opensfm_install: Path, verbose=False
) -> None:
    exe = opensfm_install / "bin" / "opensfm"
    # TODO cleanup config
    config = {
        "processes": 12,
        "matching_order_neighbors": 20,
        "feature_process_size": 1024,
        "triangulation_threshold": 0.006,
        "triangulation_type": "ROBUST",
        "min_track_length": 2,
        "retriangulation_ratio": 1.5,
        "bundle_new_points_ratio": 1.5,
    }
    with open(opensfm_dir / "config.yaml", "w") as config_file:
        yaml.dump(config, config_file)
    example_img = image_dir.glob("*").__next__()
    pil_im = Image.open(example_img)
    if camera_model == CameraModel.OPENCV:
        proj_type = "brown"
    elif camera_model == CameraModel.OPENCV_FISHEYE:
        proj_type = "fisheye_opencv"
    elif camera_model == CameraModel.EQUIRECTANGULAR:
        proj_type = "equirectangular"
    cam_overrides = {
        "all": {
            "projection_type": proj_type,
            "width": pil_im.width,
            "height": pil_im.height,
            "focal_x": 0.85,
            "focal_y": 0.85,
        }
    }
    with open(opensfm_dir / "camera_models_overrides.json", "w") as cam_file:
        json.dump(cam_overrides, cam_file)

    run_command(f"cp -r {image_dir} {opensfm_dir/'images'}")
    metadata_cmd = [f"{exe} extract_metadata", f"{opensfm_dir}"]
    metadata_cmd = " ".join(metadata_cmd)
    with status(msg="[bold yellow]Extracting OpenSfM metadata...", spinner="moon", verbose=verbose):
        run_command(metadata_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done OpenSfM metadata.")

    feature_extractor_cmd = [f"{exe} detect_features", f"{opensfm_dir}"]
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    with status(msg="[bold yellow]Running OpenSfM feature extractor...", spinner="runner", verbose=verbose):
        run_command(feature_extractor_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done extracting image features.")

    matcher_cmd = [f"{exe} match_features", f"{opensfm_dir}"]
    matcher_cmd = " ".join(matcher_cmd)
    with status(msg="[bold yellow]Running OpenSfM feature matcher...", spinner="runner", verbose=verbose):
        run_command(matcher_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done matching image features.")

    tracks_cmd = [f"{exe} create_tracks", f"{opensfm_dir}"]
    tracks_cmd = " ".join(tracks_cmd)
    with status(msg="[bold yellow]Merging OpenSfM features into tracked points...", spinner="dqpb", verbose=verbose):
        run_command(tracks_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done merging features.")

    recon_cmd = [f"{exe} reconstruct", f"{opensfm_dir}"]
    recon_cmd = " ".join(recon_cmd)
    with status(
        msg="[bold yellow]Running OpenSfM bundle adjustment... (this can take 10-15 minutes)",
        spinner="dqpb",
        verbose=verbose,
    ):
        run_command(recon_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done running bundle adjustment.")


def opensfm_to_json(reconstruction_path: Path, output_dir: Path, camera_model: CameraModel) -> int:
    sfm_out = (json.load(open(reconstruction_path, "r")))[0]
    cams = sfm_out["cameras"]
    shots = sfm_out["shots"]
    frames = []
    for shot_name, shot_info in shots.items():
        # enumerates through images in the capture
        aa_vec = np.array(shot_info["rotation"])  # 3D axis angle repr
        angle = np.linalg.norm(aa_vec)
        if angle > 1e-8:
            # normalize the axis-angle repr if angle is large enough
            aa_vec /= angle
            qx = aa_vec[0] * np.sin(angle / 2)
            qy = aa_vec[1] * np.sin(angle / 2)
            qz = aa_vec[2] * np.sin(angle / 2)
            qw = np.cos(angle / 2)
        else:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        rotation = qvec2rotmat(np.array([qw, qx, qy, qz]))
        translation = np.array(shot_info["translation"]).reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert camera +z forwards (OpenSfM) convention to -z forwards (NeRF) convention
        # Equivalent to right-multiplication by diag([1, -1, -1, 1])
        c2w[0:3, 1:3] *= -1
        # Rotation around global z-axis by -90 degrees
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = Path(f"./images/{shot_name}")

        frame = {
            "file_path": str(name),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)
    # For now just assume it's all the same camera
    cam = cams[list(cams.keys())[0]]
    out = {
        "fl_x": float(cam["focal_x"]) if "focal_x" in cam else float(cam["height"]),
        "fl_y": float(cam["focal_y"]) if "focal_y" in cam else float(cam["height"]),
        "cx": float(cam["c_x"]) if "c_x" in cam else 0.5 * cam["width"],
        "cy": float(cam["c_y"]) if "c_y" in cam else 0.5 * cam["height"],
        "w": int(cam["width"]),
        "h": int(cam["height"]),
        "camera_model": camera_model.value,
    }
    if camera_model == CameraModel.OPENCV:
        out.update(
            {
                "k1": float(cam["k1"]),
                "k2": float(cam["k2"]),
                "p1": float(cam["p1"]),
                "p2": float(cam["p2"]),
            }
        )
    if camera_model == CameraModel.OPENCV_FISHEYE:
        # TODO: find the opensfm camera model that uses these params
        out.update(
            {
                "k1": float(cam["k1"]),
                "k2": float(cam["k2"]),
                "k3": float(cam["k3"]),
                "k4": float(cam["k4"]),
            }
        )

    out["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)
