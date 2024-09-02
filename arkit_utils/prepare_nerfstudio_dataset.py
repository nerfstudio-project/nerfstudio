"""
This module prepares ARKit data for use with Nerfstudio.

It provides utilities for directory management, pose alignment,
and data conversion.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from evo.core import transformations as tr
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.linalg import orthogonal_procrustes

def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): Path of the directory to create.
    """
    os.makedirs(path, exist_ok=True)

def copy_directory(src: str, dst: str, ext: str = "txt") -> None:
    """
    Copy files with a specific extension from source to destination directory.

    Args:
        src (str): Source directory path.
        dst (str): Destination directory path.
        ext (str, optional): File extension to copy. Defaults to "txt".
    """
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isfile(s) and item.endswith(f'.{ext}'):
                shutil.copy2(s, d)
    else:
        print(f"Warning: Source directory {src} does not exist. Skipping this copy operation.")

def calculate_mean_translation_difference(arkit_poses: Dict[int, np.ndarray], method_poses: Dict[int, np.ndarray]) -> Optional[float]:
    """
    Calculate the mean translation difference between ARKit and another method's poses.

    Args:
        arkit_poses (Dict[int, np.ndarray]): Dictionary of ARKit poses.
        method_poses (Dict[int, np.ndarray]): Dictionary of poses from another method.

    Returns:
        Optional[float]: Mean translation difference, or None if no common frames are found.
    """
    differences = []
    for frame in arkit_poses:
        if frame in method_poses:
            arkit_trans = arkit_poses[frame][:3, 3]
            method_trans = method_poses[frame][:3, 3]
            diff = np.linalg.norm(arkit_trans - method_trans)
            differences.append(diff)
    return np.mean(differences) if differences else None

def load_poses(pose_file: str) -> Dict[int, np.ndarray]:
    """
    Load poses from a file in COLMAP images.txt format.

    Args:
        pose_file (str): Path to the pose file.

    Returns:
        Dict[int, np.ndarray]: Dictionary of poses, keyed by frame number.
    """
    poses = {}
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        for i in range(4, len(lines), 2):  # Start from line 4, skip every other line
            parts = lines[i].split()
            frame = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            
            # Convert quaternion to rotation matrix
            R = tr.quaternion_matrix([qw, qx, qy, qz])[:3, :3]
            
            # Construct 4x4 transformation matrix
            matrix = np.eye(4)
            matrix[:3, :3] = R
            matrix[:3, 3] = [tx, ty, tz]
            
            poses[frame] = matrix
    return poses

def calculate_average_step(poses: Dict[int, np.ndarray]) -> float:
    """
    Calculate the average distance between consecutive camera positions.

    Args:
        poses (Dict[int, np.ndarray]): Dictionary of poses.

    Returns:
        float: Average step distance.
    """
    sorted_frames = sorted(poses.keys())
    steps = []
    for i in range(1, len(sorted_frames)):
        prev_pos = poses[sorted_frames[i-1]][:3, 3]
        curr_pos = poses[sorted_frames[i]][:3, 3]
        step = np.linalg.norm(curr_pos - prev_pos)
        steps.append(step)
    return np.mean(steps) if steps else 0

def calculate_relative_scale(arkit_poses: Dict[int, np.ndarray], method_poses: Dict[int, np.ndarray]) -> float:
    """
    Calculate the relative scale between ARKit poses and another method's poses.

    Args:
        arkit_poses (Dict[int, np.ndarray]): Dictionary of ARKit poses.
        method_poses (Dict[int, np.ndarray]): Dictionary of poses from another method.

    Returns:
        float: Relative scale factor (method scale / ARKit scale).
    """
    arkit_step = calculate_average_step(arkit_poses)
    method_step = calculate_average_step(method_poses)
    return method_step / arkit_step if arkit_step != 0 else 0

def calculate_oriented_bounding_box(poses):
    """
    Calculate the oriented bounding box for a set of poses.

    Args:
        poses (Dict[int, np.ndarray]): Dictionary of poses.

    Returns:
        tuple: Center, rotation matrix, and dimensions of the bounding box.
    """
    points = np.array([pose[:3, 3] for pose in poses.values()])
    hull = ConvexHull(points)
    
    # PCA to find principal axes
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort eigenvectors by eigenvalues in descending order
    order = eigenvalues.argsort()[::-1]
    rotation = eigenvectors[:, order].T
    
    # Transform points to aligned coordinate system
    transformed = np.dot(centered, rotation.T)
    
    # Calculate min and max along each axis
    min_coords = np.min(transformed, axis=0)
    max_coords = np.max(transformed, axis=0)
    
    # Calculate center and dimensions
    center = np.mean(points, axis=0)
    dimensions = max_coords - min_coords
    
    return center, rotation, dimensions

def calculate_alignment(poses1, poses2):
    """
    Calculate the alignment between two sets of poses using oriented bounding boxes.

    Args:
        poses1 (Dict[int, np.ndarray]): First set of poses.
        poses2 (Dict[int, np.ndarray]): Second set of poses.

    Returns:
        tuple: Translation vector, rotation matrix, and scale factor between the two bounding boxes.
    """
    center1, rotation1, dimensions1 = calculate_oriented_bounding_box(poses1)
    center2, rotation2, dimensions2 = calculate_oriented_bounding_box(poses2)
    
    # Calculate translation
    translation = center2 - center1
    
    # Calculate rotation
    rotation, _ = orthogonal_procrustes(rotation1, rotation2)
    
    # Calculate scale
    scale = np.linalg.norm(dimensions2) / np.linalg.norm(dimensions1)
    
    return translation, rotation, scale

def main(args: argparse.Namespace) -> None:
    """
    Main function to prepare ARKit data for Nerfstudio.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    root_path = Path(args.input_path).resolve()
    parent_name = root_path.parent.name
    output_root = root_path.parent / f"{parent_name}_nerfstudio"

    # Create new directory structure
    create_directory(output_root)
    
    colmap_dir = output_root / "colmap"
    create_directory(colmap_dir)
    
    # Copy ARKit data
    arkit_src = root_path / "post" / "sparse" / "online_loop"
    arkit_dst = colmap_dir / "arkit" / "0"
    copy_directory(arkit_src, arkit_dst)
    
    # Align and visualize poses for each method
    methods: List[str] = ["lightglue", "loftr", "colmap", "glomap"]

    # Copy data for each method
    for method in methods:
        method_src = root_path / "post" / "sparse" / "offline" / method / "final"
        if method == "glomap":
            method_src = method_src / "0"
        method_dst = colmap_dir / method / "0"
        copy_directory(method_src, method_dst)

    # Copy images
    images_src = root_path / "post" / "images"
    images_dst = output_root / "images"
    copy_directory(images_src, images_dst, ext="png")

    print(f"Conversion completed successfully. Output directory: {output_root}")

    # Load ARKit poses
    arkit_poses = load_poses(arkit_dst / "images.txt")

    # Calculate and save mean translation differences
    alignments: Dict[str, Dict[str, Union[np.ndarray, float]]] = {}
    for method in methods:
        pose_file = colmap_dir / method / "0" / "images.txt"
        if not os.path.exists(pose_file):
            continue
        method_poses = load_poses(pose_file)
        # Calculate alignment
        translation, rotation, scale = calculate_alignment(arkit_poses, method_poses)
        alignments[method] = {
            "translation": translation,
            "rotation": rotation,
            "scale": scale
        }

    # Save differences as a table
    with open(output_root / "alignment_eval.txt", "w") as f:
        f.write("Method\tTranslation (m)\tRotation (deg)\tScale\n")
        f.write("-" * 60 + "\n")
        for method, alignment in alignments.items():
            translation = alignment["translation"]
            # Format translation as a string of x, y, z components
            translation_str = f"({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})"
            
            scale = alignment["scale"]

            rotation = alignment["rotation"]
            rotation_angle = np.arccos((np.trace(rotation) - 1) / 2)
            rotation_deg = np.degrees(rotation_angle)
            f.write(f"{method}\t{translation_str}\t{rotation_deg:.2f}\t{scale:.6f}\n")

    print(f"Conversion completed successfully. Output directory: {output_root}")
    print(f"Mean translation differences saved to: {output_root}/alignment_eval.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ARKit 3DGS output for nerfstudio training.")
    parser.add_argument("--input_path", help="Path to the root directory of run_arkit_3dgs.sh output")
    args = parser.parse_args()
    
    main(args)