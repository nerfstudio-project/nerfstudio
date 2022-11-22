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
    
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    
    retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
    
    feature_path = extract_features.main(feature_conf, image_dir, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
    
    model = reconstruction.main(sfm_dir, image_dir, sfm_pairs, feature_path, match_path, camera_mode=pycolmap.CameraMode.SINGLE, image_options=pycolmap.ImageReaderOptions(camera_model='OPENCV'))
