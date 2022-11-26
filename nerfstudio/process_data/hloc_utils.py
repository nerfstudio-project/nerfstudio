from pathlib import Path
from typing_extensions import Literal

from nerfstudio.process_data.process_data_utils import CameraModel
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, pairs_from_retrieval, pairs_from_exhaustive

from hloc.triangulation import (import_features, import_matches)
from hloc.utils.io import get_keypoints, get_matches

import pycolmap

def run_hloc(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    gpu: bool = True,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    feature_type: Literal["sift", "superpoint_aachen", "superpoint_max", "superpoint_inloc", "r2d2", "d2net-ss", "sosnet", "disk"] = "superpoint_aachen",
    matcher_type: Literal["superglue", "superglue-fast", "NN-superpoint", "NN-ratio", "NN-mutual", "adalam"] = "superglue",
    num_matched: int = 25,
) -> None:
    """Runs hloc on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
    """
    outputs = colmap_dir
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sparse' / '0'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'
    
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs[feature_type]
    matcher_conf = match_features.confs[matcher_type]
    
    references = image_dir.iterdir()
    extract_features.main(feature_conf, image_dir, image_list=references, feature_path=features)
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    else:
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    match_path = match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    
    image_options=pycolmap.ImageReaderOptions(camera_model=camera_model.value)
    model = reconstruction.main(sfm_dir, image_dir, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_options=image_options, verbose=verbose)
    
