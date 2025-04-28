# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper utils for processing meshroom data into the nerfstudio format."""

import json
import math
from copy import deepcopy as dc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils.rich_utils import CONSOLE

# Rotation matrix to adjust coordinate system
ROT_MAT = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0,-1, 0, 0],
                    [0, 0, 0, 1]])

def reflect(axis, size=4):
    """Create a reflection matrix along the specified axis."""
    _diag = np.ones(size)
    _diag[axis] = -1
    refl = np.diag(_diag)
    return refl

def Mat2Nerf(mat):
    """Convert a matrix to NeRF coordinate system."""
    M = np.array(mat)
    M = ((M @ reflect(2)) @ reflect(1))
    return M

def closest_point_2_lines(oa, da, ob, db): 
    """Find the point closest to both rays of form o+t*d."""
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def central_point(out):
    """Find a central point all cameras are looking at."""
    CONSOLE.print("Computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = np.array(f["transform_matrix"])[0:3,:]
        for g in out["frames"]:
            mg = np.array(g["transform_matrix"])[0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.01:
                totp += p*w
                totw += w
                
    if len(out["frames"]) == 0:
        CONSOLE.print("[bold red]No frames found when computing center of attention[/bold red]")
        return totp

    if (totw == 0) and (not totp.any()):
        CONSOLE.print("[bold red]Center of attention is zero[/bold red]")
        return totp
    
    totp /= totw
    CONSOLE.print(f"The center of attention is: {totp}")

    return totp

def build_sensor(intrinsic):
    """Build camera intrinsics from Meshroom data."""
    out = {}
    out["w"] = float(intrinsic['width'])
    out["h"] = float(intrinsic['height'])

    # Focal length in mm
    focal = float(intrinsic['focalLength'])
    
    # Sensor width in mm
    sensor_width = float(intrinsic['sensorWidth'])
    sensor_height = float(intrinsic['sensorHeight'])

    # Focal length in pixels
    out["fl_x"] = (out["w"] * focal) / sensor_width

    # Check W/H ratio to sensor ratio
    if np.isclose((out["w"] / out["h"]), (sensor_width / sensor_height)):
        out["fl_y"] = (out["h"] * focal) / sensor_height
    else:
        CONSOLE.print("[yellow]WARNING: W/H ratio does not match sensor ratio, this is likely a bug from Meshroom. Will use fl_x to set fl_y.[/yellow]")
        out["fl_y"] = out["fl_x"]

    camera_angle_x = math.atan(out["w"] / (out['fl_x']) * 2) * 2
    camera_angle_y = math.atan(out["h"] / (out['fl_y']) * 2) * 2

    out["camera_angle_x"] = camera_angle_x
    out["camera_angle_y"] = camera_angle_y

    out["cx"] = float(intrinsic['principalPoint'][0]) + (out["w"] / 2.0)
    out["cy"] = float(intrinsic['principalPoint'][1]) + (out["h"] / 2.0)

    if intrinsic['type'] == 'radial3':
        for i, coef in enumerate(intrinsic['distortionParams']):
            out[f"k{i + 1}"] = float(coef)
    
    return out

def meshroom_to_json(
    image_filename_map: Dict[str, Path],
    json_filename: Path,
    output_dir: Path,
    ply_filename: Optional[Path] = None,
    verbose: bool = False,
) -> List[str]:
    """Convert Meshroom data into a nerfstudio dataset.

    Args:
        image_filename_map: Mapping of original image filenames to their saved locations.
        json_filename: Path to the Meshroom json file.
        output_dir: Path to the output directory.
        ply_filename: Path to the exported ply file.
        verbose: Whether to print verbose output.

    Returns:
        Summary of the conversion.
    """
    summary_log = []
    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    # Create output structure
    out = {}
    out['aabb_scale'] = 16  # Default value
    
    # Extract transforms from Meshroom data
    transforms = {}
    for pose in data.get('poses', []):
        transform = pose['pose']['transform']
        rot = np.asarray(transform['rotation'])
        rot = rot.reshape(3, 3).astype(float)

        ctr = np.asarray(transform['center'])
        ctr = ctr.astype(float)

        M = np.eye(4)
        M[:3, :3] = rot
        M[:3, 3] = ctr

        M = Mat2Nerf(M.astype(float))
        transforms[pose['poseId']] = np.dot(ROT_MAT, M)

    # Extract intrinsics from Meshroom data
    intrinsics = {}
    for intrinsic in data.get('intrinsics', []):
        intrinsics[intrinsic['intrinsicId']] = build_sensor(intrinsic)
    
    # Set camera model based on intrinsic type
    if data.get('intrinsics') and 'type' in data['intrinsics'][0]:
        intrinsic_type = data['intrinsics'][0]['type']
        if intrinsic_type in ['radial1', 'radial3']:
            out["camera_model"] = CAMERA_MODELS["perspective"].value
        elif intrinsic_type in ['fisheye', 'fisheye4']:
            out["camera_model"] = CAMERA_MODELS["fisheye"].value
        else:
            # Default to perspective
            out["camera_model"] = CAMERA_MODELS["perspective"].value
    else:
        out["camera_model"] = CAMERA_MODELS["perspective"].value
    
    # Build frames
    frames = []
    skipped_images = 0
    
    for view in data.get('views', []):
        # Get the image name from the path
        path = Path(view['path'])
        name = path.stem
        
        # Check if the image exists in our mapping
        if name not in image_filename_map:
            if verbose:
                CONSOLE.print(f"[yellow]Missing image for {name}, skipping[/yellow]")
            skipped_images += 1
            continue
        
        # Get poseId and intrinsicId
        poseId = view['poseId']
        intrinsicId = view['intrinsicId']
        
        # Check if we have the necessary data
        if poseId not in transforms:
            if verbose:
                CONSOLE.print(f"[yellow]PoseId {poseId} not found in transforms, skipping image: {name}[/yellow]")
            skipped_images += 1
            continue
        
        if intrinsicId not in intrinsics:
            if verbose:
                CONSOLE.print(f"[yellow]IntrinsicId {intrinsicId} not found, skipping image: {name}[/yellow]")
            skipped_images += 1
            continue
        
        # Create camera data
        camera = {}
        camera.update(dc(intrinsics[intrinsicId]))
        camera['transform_matrix'] = transforms[poseId]
        camera['file_path'] = image_filename_map[name].as_posix()
        
        frames.append(camera)
    
    out['frames'] = frames
    
    # Calculate center point
    center = central_point(out)
    
    # Adjust camera positions by centering
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= center
        f["transform_matrix"] = f["transform_matrix"].tolist()
    
    # Include point cloud if provided
    if ply_filename is not None:
        import open3d as o3d
        
        # Create the applied transform
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([2, 0, 1]), :]
        out["applied_transform"] = applied_transform.tolist()
        
        # Load and transform point cloud
        pc = o3d.io.read_point_cloud(str(ply_filename))
        points3D = np.asarray(pc.points)
        points3D = np.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]
        pc.points = o3d.utility.Vector3dVector(points3D)
        o3d.io.write_point_cloud(str(output_dir / "sparse_pc.ply"), pc)
        out["ply_file_path"] = "sparse_pc.ply"
        summary_log.append(f"Imported {ply_filename} as starting points")
    
    # Write output
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    
    # Add summary info
    if skipped_images == 1:
        summary_log.append(f"{skipped_images} image skipped due to missing camera pose or intrinsic data.")
    elif skipped_images > 1:
        summary_log.append(f"{skipped_images} images were skipped due to missing camera poses or intrinsic data.")
    
    summary_log.append(f"Final dataset contains {len(out['frames'])} frames.")
    
    return summary_log