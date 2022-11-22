from pathlib import Path
from typing_extensions import Literal

from nerfstudio.process_data.process_data_utils import CameraModel
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, pairs_from_retrieval, pairs_from_exhaustive
import pycolmap

def run_hloc(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    gpu: bool = True,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree"
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
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    
    references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]
    extract_features.main(feature_conf, image_dir, image_list=references, feature_path=features)
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    else:
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=25)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    match_path = match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    
    model = reconstruction.main(sfm_dir, image_dir, sfm_pairs, features, matches, image_list=references, camera_mode=pycolmap.CameraMode.SINGLE, image_options=pycolmap.ImageReaderOptions(camera_model='OPENCV'))
