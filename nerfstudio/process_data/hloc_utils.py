"""
Code that uses the hierarchical localization toolbox (hloc)
to extract and match image features, estimate camera poses,
and do sparse reconstruction.
Requires hloc module from : https://github.com/cvg/Hierarchical-Localization
"""

# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

import sys
from pathlib import Path

from rich.console import Console
from typing_extensions import Literal

from nerfstudio.process_data.process_data_utils import CameraModel

try:
    # TODO(1480) un-hide pycolmap import
    import pycolmap
    from hloc import (
        extract_features,
        match_features,
        pairs_from_exhaustive,
        pairs_from_retrieval,
        reconstruction,
    )
except ImportError:
    _HAS_HLOC = False
else:
    _HAS_HLOC = True

try:
    from pixsfm.refine_hloc import PixSfM
except ImportError:
    _HAS_PIXSFM = False
else:
    _HAS_PIXSFM = True

CONSOLE = Console(width=120)


def run_hloc(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    feature_type: Literal[
        "sift", "superpoint_aachen", "superpoint_max", "superpoint_inloc", "r2d2", "d2net-ss", "sosnet", "disk"
    ] = "superpoint_aachen",
    matcher_type: Literal[
        "superglue", "superglue-fast", "NN-superpoint", "NN-ratio", "NN-mutual", "adalam"
    ] = "superglue",
    num_matched: int = 50,
    refine_pixsfm: bool = False,
) -> None:
    """Runs hloc on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
        matching_method: Method to use for matching images.
        feature_type: Type of visual features to use.
        matcher_type: Type of feature matcher to use.
        num_matched: Number of image pairs for loc.
        refine_pixsfm: If True, refine the reconstruction using pixel-perfect-sfm.
    """
    if not _HAS_HLOC:
        CONSOLE.print(
            f"[bold red]Error: To use this set of parameters ({feature_type}/{matcher_type}/hloc), "
            "you must install hloc toolbox!!"
        )
        sys.exit(1)

    if refine_pixsfm and not _HAS_PIXSFM:
        CONSOLE.print("[bold red]Error: use refine_pixsfm, you must install pixel-perfect-sfm toolbox!!")
        sys.exit(1)

    outputs = colmap_dir
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sparse" / "0"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs[feature_type]
    matcher_conf = match_features.confs[matcher_type]

    references = [p.relative_to(image_dir).as_posix() for p in image_dir.iterdir()]
    extract_features.main(feature_conf, image_dir, image_list=references, feature_path=features)
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    else:
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)
        if num_matched >= len(references):
            num_matched = len(references)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    image_options = pycolmap.ImageReaderOptions(  # pylint: disable=c-extension-no-member
        camera_model=camera_model.value
    )
    if refine_pixsfm:
        sfm = PixSfM(
            conf={
                "dense_features": {"use_cache": True},
                "KA": {"dense_features": {"use_cache": True}, "max_kps_per_problem": 1000},
                "BA": {"strategy": "costmaps"},
            }
        )
        refined, _ = sfm.reconstruction(
            sfm_dir,
            image_dir,
            sfm_pairs,
            features,
            matches,
            image_list=references,
            camera_mode=pycolmap.CameraMode.SINGLE,  # pylint: disable=c-extension-no-member
            image_options=image_options,
            verbose=verbose,
        )
        print("Refined", refined.summary())

    else:
        reconstruction.main(
            sfm_dir,
            image_dir,
            sfm_pairs,
            features,
            matches,
            camera_mode=pycolmap.CameraMode.SINGLE,  # pylint: disable=c-extension-no-member
            image_options=image_options,
            verbose=verbose,
        )
