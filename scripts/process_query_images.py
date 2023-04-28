from pathlib import Path
import json
import argparse
import pandas as pd
import numpy as np
from typing import List, Optional, Literal
import sys
from rich.console import Console

from nerfstudio.utils.io import load_from_json
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_cameras_binary # also has txt read methods
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params, get_colmap_version, get_vocab_tree, colmap_to_json
from nerfstudio.process_data.process_data_utils import CameraModel, downscale_images
from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="data/nerfstudio/poster", help='Data path to original dataset')
parser.add_argument('--query_path', type=str, default="/home/maturk/git/nerfstudio/data/desk/query/query_images.txt", help='File to query data')
parser.add_argument('--query_dir', type=str, default="data/nerfstudio/poster/queries", help="Path to folder where query images are stored")
parser.add_argument('--bundle_adjust', type=bool, default=True, help="Refine image registration with bundle adjustment (optional)")
parser.add_argument('--verbose', type=bool, default=True, help= '')

# TODO: Add other arguments. Mainly Camera
# TODO: somehow check that query image is not named the same way as original images. Make a name converstion. 
# TODO: reconsider input/output directories to not make too many extra files
DEFAULT_COLMAP_PATH = Path("colmap/sparse/0")
CONSOLE = Console(width=120)

def register_images(
    query_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel = CAMERA_MODELS['perspective'],
    camera_mask_path: Optional[Path] = None,
    gpu: bool = False,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    colmap_cmd: str = "colmap",
    bundle_adjust: bool = True,
) -> None:
    """Runs COLMAP on query images and matches them based on existing dataset.

    Args:
        query_dir: Path to the directory containing query images.
        colmap_dir: Path to existing colmap directory containing database.db file.
        camera_model: Camera model to use.
        camera_mask_path: Path to the camera mask.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
        matching_method: Matching method to use.
        colmap_cmd: Path to the COLMAP executable.
    """

    colmap_version = get_colmap_version(colmap_cmd)

    colmap_database_path = colmap_dir / "database.db"

    # Feature extraction
    feature_extractor_cmd = [
        f"{colmap_cmd} feature_extractor",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {query_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {camera_model.value}",
        f"--SiftExtraction.use_gpu {int(gpu)}",
        "--ImageReader.existing_camera_id 1"
    ]
    if camera_mask_path is not None:
        feature_extractor_cmd.append(f"--ImageReader.camera_mask_path {camera_mask_path}")
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    with status(msg="[bold yellow]Running COLMAP feature extractor...", spinner="moon", verbose=verbose):
        run_command(feature_extractor_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done extracting COLMAP features on query image(s).")

    # Feature matching
    feature_matcher_cmd = [
        f"{colmap_cmd} {matching_method}_matcher",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--SiftMatching.use_gpu {int(gpu)}",
    ]
    if matching_method == "vocab_tree":
        vocab_tree_filename = get_vocab_tree()
        feature_matcher_cmd.append(f"--VocabTreeMatching.vocab_tree_path {vocab_tree_filename}")
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    with status(msg="[bold yellow]Running COLMAP feature matcher...", spinner="runner", verbose=verbose):
        run_command(feature_matcher_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done matching COLMAP features.")

    # Image registration
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    image_registration_cmd = [
        f"{colmap_cmd} image_registrator",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--input_path {sparse_dir}/0",
        f"--output_path {query_dir}",
    ]
    image_registration_cmd = " ".join(image_registration_cmd)
    with status(msg="[bold yellow]Running COLMAP image registration...", spinner="runner", verbose=verbose):
        run_command(image_registration_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done registering query images")

    # Bundle adjust (Optional)
    if bundle_adjust:
        bundle_adjuster_cmd = [
            f"{colmap_cmd} bundle_adjuster",
            f"--input_path {query_dir}",
            f"--output_path {query_dir}",
        ]
        bundle_adjuster_cmd = " ".join(bundle_adjuster_cmd)
        with status(msg="[bold yellow]Running COLMAP bundle adjustment on query images...", spinner="runner", verbose=verbose):
            run_command(bundle_adjuster_cmd, verbose=verbose)
        CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment on query images.")

    # TODO: check valid paths
    #if not (self.output_dir / self.colmap_model_path).exists():
    #            CONSOLE.log(
    #                f"[bold red]The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist."
    #            )
    #            sys.exit(1)
    #colmap image_registrator --database_path data/desk/colmap/database.db --input_path data/desk/colmap/sparse/0/ --output_path data/desk/query/

    # TODO: make this optional
    # Bundle adjustment
    # sparse_dir = colmap_dir / "sparse"
    # sparse_dir.mkdir(parents=True, exist_ok=True)
    # mapper_cmd = [
    #     f"{colmap_cmd} mapper",
    #     f"--database_path {colmap_dir / 'database.db'}",
    #     f"--image_path {query_dir}",
    #     f"--output_path {sparse_dir}",
    # ]
    # if colmap_version >= 3.7:
    #     mapper_cmd.append("--Mapper.ba_global_function_tolerance 1e-6")

    # mapper_cmd = " ".join(mapper_cmd)

    # with status(
    #     msg="[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
    #     spinner="circle",
    #     verbose=verbose,
    # ):
    #     run_command(mapper_cmd, verbose=verbose)
    # CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment.")
    # with status(msg="[bold yellow]Refine intrinsics...", spinner="dqpb", verbose=verbose):
    #     bundle_adjuster_cmd = [
    #         f"{colmap_cmd} bundle_adjuster",
    #         f"--input_path {sparse_dir}/0",
    #         f"--output_path {sparse_dir}/0",
    #         "--BundleAdjustment.refine_principal_point 1",
    #     ]
    #     run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
    # CONSOLE.log("[bold green]:tada: Done refining intrinsics.")



def main():
    flags = parser.parse_args()
    # pylint: disable=too-many-statements
    colmap_model_path: Path = DEFAULT_COLMAP_PATH
    require_cameras_exist = False
    colmap_model_path = opt.dataset / Path(colmap_model_path)
    if self.colmap_model_path != DEFAULT_COLMAP_PATH:
        if not self.skip_colmap:
            CONSOLE.log("[bold red]The --colmap-model-path can only be used when --skip-colmap is not set.")
            sys.exit(1)
        elif not (self.output_dir / self.colmap_model_path).exists():
            CONSOLE.log(
                f"[bold red]The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist."
            )
            sys.exit(1)
        require_cameras_exist = True
    num_downscales = 2
    #TODO: check this dir exists
    colmap_dir = Path(flags.dataset) / "colmap"
    skip_image_processing = False
    skip_colmap = False
    if not skip_image_processing:
        downscale_images(Path(flags.query_dir)/'images', num_downscales, verbose=flags.verbose)
    if not skip_colmap:
        if not (colmap_dir).exists():
            CONSOLE.log(
                f"[bold red]The colmap-dir-path {colmap_dir} does not exist.")
            sys.exit(1)
        register_images(query_dir=Path(flags.query_dir), colmap_dir=colmap_dir, bundle_adjust=flags.bundle_adjust, verbose=flags.verbose)
        #print(Path(flags.query_dir), Path(flags.dataset))
        colmap_to_json(recon_dir=Path(flags.query_dir), output_dir=Path(flags.dataset))
    #query_frames = process_query_images(Path(dataset=flags.dataset), query_path=flags.query_path)
    #append_query_images(dataset=Path(flags.dataset), query_frames=query_frames)


if __name__ == '__main__':
    main()
# test:
# assert parser.get_dataparser_outputs("test").image_filenames == [
#         mocked_dataset / "images_3/img_4.png",
#         mocked_dataset / "images_3/img_5.png",
#     ]
# process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose)
