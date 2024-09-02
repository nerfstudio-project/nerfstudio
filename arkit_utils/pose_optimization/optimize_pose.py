import argparse
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, List
import logging
import colorlog
import time
from tabulate import tabulate

import pycolmap
from hloc.utils.read_write_model import Camera, Image, Point3D, write_model, read_model
from hloc import extract_features, match_features, pairs_from_poses, triangulation, match_dense

# Set up colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s:%(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def prepare_pose_and_intrinsic_prior(dataset_base: str) -> Path:
    """
    Prepare pose and intrinsic prior from ARKit data.

    Args:
        dataset_base (str): Path to the dataset base directory.

    Returns:
        Path: Path to the prepared COLMAP model.
    """
    dataset_dir = Path(dataset_base)
    
    # Read ARKit poses and camera parameters
    logger.info("Reading ARKit poses and camera parameters...")
    images = read_arkit_poses(dataset_base)
    cameras = read_arkit_cameras(dataset_base)
    points3D: Dict[int, Point3D] = {}

    logger.info('Writing the COLMAP model...')
    colmap_arkit_base = dataset_dir / 'post' / 'sparse' / 'offline'
    colmap_arkit = colmap_arkit_base / 'raw'
    colmap_arkit.mkdir(exist_ok=True, parents=True)
    write_model(images=images, cameras=cameras, points3D=points3D, path=str(colmap_arkit), ext='.bin')

    return colmap_arkit

def read_arkit_poses(dataset_base: str) -> Dict[int, Image]:
    """
    Read ARKit poses from the dataset.

    Args:
        dataset_base (str): Path to the dataset base directory.

    Returns:
        Dict[int, Image]: Dictionary of Image objects keyed by image ID.
    """
    images: Dict[int, Image] = {}
    logger.info(f"Reading ARKit poses from {dataset_base}/post/sparse/online_loop/images.txt")
    with open(dataset_base + "/post/sparse/online_loop/images.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                         camera_id=camera_id, name=image_name,
                                         xys=xys, point3D_ids=point3D_ids)
    logger.info(f"Read {len(images)} ARKit poses")
    return images

def read_arkit_cameras(dataset_base: str) -> Dict[int, Camera]:
    """
    Read ARKit camera parameters from the dataset.

    Args:
        dataset_base (str): Path to the dataset base directory.

    Returns:
        Dict[int, Camera]: Dictionary of Camera objects keyed by camera ID.
    """
    cameras: Dict[int, Camera] = {}
    logger.info(f"Reading ARKit camera parameters from {dataset_base}/post/sparse/online_loop/cameras.txt")
    with open(dataset_base + "/post/sparse/online_loop/cameras.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader supports other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    logger.info(f"Read {len(cameras)} ARKit camera parameters")
    return cameras

def setup_hloc(dataset_base: str, 
               colmap_arkit: Path, 
               methods: List[str], 
               n_matched: int) -> Dict[str, Tuple[Path, Path, Path, Path, Path]]:
    """
    Set up HLOC for pose optimization.

    Args:
        dataset_base (str): Path to the dataset base directory.
        colmap_arkit (Path): Path to the COLMAP ARKit model.
        methods (List[str]): List of pose optimization methods.
        n_matched (int): Number of matched images.

    Returns:
        Dict[str, Tuple[Path, Path, Path, Path, Path]]: Dictionary of paths for each method.
    """
    dataset_dir = Path(dataset_base)
    results = {}

    for method in methods:
        logger.info(f"Setting up HLOC for method: {method}")
        outputs = dataset_dir / 'post' / 'sparse' / 'offline' / method
        outputs.mkdir(exist_ok=True, parents=True)

        images = dataset_dir / 'post' / 'images'
        sfm_pairs = outputs / 'pairs-sfm.txt'
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'

        logger.info(f"Generating pairs from poses for {method}")
        pairs_from_poses.main(colmap_arkit, sfm_pairs, n_matched)

        if method in ['colmap', 'lightglue', 'glomap']:
            references = [str(p.relative_to(images)) for p in images.iterdir()]
            feature_conf = extract_features.confs['superpoint_inloc']
            logger.info(f"Extracting features for {method}")
            extract_features.main(feature_conf, images, image_list=references, feature_path=features)
            
            matcher_conf = match_features.confs['superglue'] if method == 'colmap' else match_features.confs['superpoint+lightglue']
            logger.info(f"Matching features for {method}")
            match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        elif method == 'loftr':
            matcher_conf = match_dense.confs['loftr_aachen']
            logger.info(f"Performing dense matching for {method}")
            features, matches = match_dense.main(matcher_conf, sfm_pairs, images, outputs, max_kps=8192, overwrite=False)

        results[method] = (outputs, images, sfm_pairs, features, matches)

    return results

def optimize_pose(dataset_base: str, 
                  methods: List[str], 
                  n_ba_iterations: int, 
                  n_matched: int = 10) -> None:
    """
    Optimize ARKit pose using various methods.

    Args:
        dataset_base (str): Path to the dataset base directory.
        methods (List[str]): List of pose optimization methods.
        n_ba_iterations (int): Number of bundle adjustment iterations.
        n_matched (int, optional): Number of matched images. Defaults to 10.
    """
    logger.info("Starting pose optimization process")
    colmap_arkit = prepare_pose_and_intrinsic_prior(dataset_base)
    hloc_setups = setup_hloc(dataset_base, colmap_arkit, methods, n_matched)

    processing_times = {}

    for method, (outputs, images, sfm_pairs, features, matches) in hloc_setups.items():
        logger.info(f"Optimizing pose using method: {method}")
        start_time = time.time()
        if method == 'glomap':
            optimize_pose_glomap(outputs, images, colmap_arkit, sfm_pairs, features, matches)
            output_dir = outputs / 'final' / '0'
        else:
            optimize_pose_colmap(outputs, images, colmap_arkit, sfm_pairs, features, matches, n_ba_iterations, method)
            output_dir = outputs / 'final'
        end_time = time.time()
        processing_times[method] = end_time - start_time

        # Get BA results and write to external folder
        colmap_arkit_base = output_dir
        logger.info(f"Reading optimized model for {method}")
        cameras, images, point3D = read_model(output_dir, ext=".bin")
        
        logger.info("Sorting cameras, images, and points3D")
        sorted_cameras = dict(sorted(cameras.items()))
        sorted_images = dict(sorted(images.items()))
        sorted_point3D = dict(sorted(point3D.items()))
        
        logger.info(f"Writing optimized model for {method}")
        write_model(sorted_cameras, sorted_images, sorted_point3D, colmap_arkit_base, ext=".txt")

    # Print comparison table
    table_data = [[method, f"{time:.2f} seconds"] for method, time in processing_times.items()]
    table = tabulate(table_data, headers=["Method", "Processing Time"], tablefmt="grid")
    print("\nComparison of Pose-Refining Methods:")
    print(table)

def optimize_pose_colmap(outputs: Path, 
                         images: Path, 
                         colmap_input: Path, 
                         sfm_pairs: Path, 
                         features: Path, 
                         matches: Path, 
                         n_ba_iterations: int, 
                         method: str) -> None:
    """
    Optimize pose using COLMAP.

    Args:
        outputs (Path): Path to output directory.
        images (Path): Path to images directory.
        colmap_input (Path): Path to COLMAP input.
        sfm_pairs (Path): Path to SfM pairs file.
        features (Path): Path to features file.
        matches (Path): Path to matches file.
        n_ba_iterations (int): Number of bundle adjustment iterations.
        method (str): Method to refine poses.
    """
    for i in range(n_ba_iterations):
        logger.info(f"COLMAP optimization iteration {i+1}/{n_ba_iterations}")
        colmap_sparse = outputs / 'colmap_sparse'
        colmap_sparse.mkdir(exist_ok=True, parents=True)
        logger.info("Performing triangulation")
        triangulation.main(colmap_sparse, colmap_input, images, sfm_pairs, features, matches)
        
        colmap_ba = outputs / 'final'
        colmap_ba.mkdir(exist_ok=True, parents=True)
        logger.info("Running bundle adjustment")
        # --BundleAdjustment.refine_rotation_only 1
        BA_cmd = f'colmap bundle_adjuster --BundleAdjustment.refine_focal_length 0 --BundleAdjustment.refine_principal_point 0 --BundleAdjustment.refine_extra_params 0 --input_path {colmap_sparse} --output_path {colmap_ba}'
        os.system(BA_cmd)
        
        colmap_input = colmap_ba

def optimize_pose_glomap(outputs: Path, images: Path, colmap_input: Path, sfm_pairs: Path, features: Path, matches: Path) -> None:
    """
    Optimize pose using GLOMAP.

    Args:
        outputs (Path): Path to output directory.
        images (Path): Path to images directory.
        colmap_input (Path): Path to COLMAP input.
        sfm_pairs (Path): Path to SfM pairs file.
        features (Path): Path to features file.
        matches (Path): Path to matches file.
    """
    logger.info("Starting GLOMAP optimization")
    colmap_sparse = outputs / 'colmap_sparse'
    colmap_sparse.mkdir(exist_ok=True, parents=True) 
    logger.info("Performing triangulation for GLOMAP")
    triangulation.main(colmap_sparse, colmap_input, images, sfm_pairs, features, matches)

    glomap = outputs / 'final'
    glomap.mkdir(exist_ok=True, parents=True)
    logger.info("Running GLOMAP mapper")
    glomap_cmd = f"glomap mapper --database_path {colmap_sparse}/database.db --image_path {images} --output_path {glomap}"
    exit_code = os.system(glomap_cmd)
    if exit_code != 0:
        logger.error(f"GLOMAP mapper failed with code {exit_code}")
        raise RuntimeError(f"GLOMAP mapper failed with code {exit_code}")
    logger.info("GLOMAP optimization completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ARkit pose using various methods")
    parser.add_argument("--input_database_path", type=str, default="data/arkit_pose/study_room/arkit_undis")
    parser.add_argument("--methods", nargs='+', type=str, choices=['colmap', 'loftr', 'lightglue', 'glomap'], default=['colmap'], help="Choose pose optimization methods")
    parser.add_argument("--BA_iterations", type=int, default=5)

    args = parser.parse_args()

    logger.info("Starting pose optimization script")
    optimize_pose(args.input_database_path, args.methods, args.BA_iterations)
    logger.info("Pose optimization completed")