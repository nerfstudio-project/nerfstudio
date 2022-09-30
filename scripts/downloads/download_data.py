#!/usr/bin/env python
"""Download datasets"""
import zipfile
from pathlib import Path
from typing import Literal, Optional

import dcargs
import gdown

nerfactory_file_ids = {
    "dozer": "1-OR5F_V5S4s-yzxohbwTylaXjzYLu8ZR",
    "sf_street": "1DbLyptL6my2QprEVtYuW2uzgp9JAK5Wz",
    "poster": "1dmjWGXlJnUxwosN6MVooCDQe970PkD-1",
    "lion": "1bx0aDtHoqB3xTAVHL9UIrlTL_vqC6vaH",
}


def download_friends():
    """Script ported from: https://github.com/ethanweber/sitcoms3D/blob/master/download_data.py"""
    data_filename_to_file_id = [
        ("./data/human_data.zip", "1zk12qxnZbReKqdvy-JYp1jQn2QA3oJzz"),
        ("./data/human_pairs.zip", "1927rnmJ3mWcjsj0afqawSto2IK2lLwmA"),
        ("./data/sparse_reconstruction_and_nerf_data.zip", "1RmwDUAp1T4RkwZg1S_7L9JAViWtBZfAG"),
    ]

    disk_data_filename_to_file_id = [  # pylint: disable=unused-variable
        ("./data/sparse_reconstruction_and_nerf_data/ELR-apartment-disk.zip", "1YcTwMv5PP0uqXYdN-Lp8k0WnXUdHBXrY"),
        ("./data/sparse_reconstruction_and_nerf_data/Frasier-apartment-disk.zip", "1vPqplDv5rrFu5LrxTHbET0FiePTGi8zE"),
        (
            "./data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment-disk.zip",
            "1yNpU4M44gIuWEaC5tSAOKy-A6KX8wTm8",
        ),
        (
            "./data/sparse_reconstruction_and_nerf_data/HIMYM-red_apartment-disk.zip",
            "1m76i46YYlpDKvlx-Jn7qDDAvkOVVVmqF",
        ),
        (
            "./data/sparse_reconstruction_and_nerf_data/Seinfeld-jerry_living_room-disk.zip",
            "1YFoV3Gd9asKsRarZvzRTYtri1755rBPx",
        ),
        ("./data/sparse_reconstruction_and_nerf_data/TAAHM-kitchen-disk.zip", "1QpF8kVGfgkqm_cDY5sY_E82BT8VOWL38"),
        (
            "./data/sparse_reconstruction_and_nerf_data/TBBT-big_living_room-disk.zip",
            "175BjkpxMAOt75ZVIjYpArxWpO8uRexgJ",
        ),
    ]

    # !!! Large files !!! Only uncomment if you need the disk correspondences.
    # data_filename_to_file_id += disk_data_filename_to_file_id

    for data_filename, file_id in data_filename_to_file_id:
        # Download the files
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output=data_filename, quiet=False)
        with zipfile.ZipFile(data_filename, "r") as zip_ref:
            zip_ref.extractall(data_filename.replace(".zip", "").replace("-disk", ""))
        Path(data_filename).unlink(missing_ok=True)


def download_blender():
    """Download blender dataset and format accordingly"""
    url = "https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"
    download_path = Path("data/blender_data.zip")
    gdown.download(url, output=str(download_path), quiet=False)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall("data/")
    unzip_path = Path("data/nerf_synthetic")
    final_path = Path("data/blender")
    unzip_path.rename(final_path)
    download_path.unlink(missing_ok=True)


def download_nerfactory(dataset_name: str):
    """Download a zipped nerfactory dataset"""
    url = f"https://drive.google.com/uc?id={nerfactory_file_ids[dataset_name]}"
    download_path = Path(f"data/{dataset_name}.zip")
    gdown.download(url, output=str(download_path), quiet=False)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(f"data/ours/{dataset_name}")
    download_path.unlink(missing_ok=True)


def main(
    dataset: Literal["blender", "friends", "nerfactory"],
    dataset_name: Optional[str] = None,
):
    """Main download script to download all data"""
    save_dir = Path("data/")
    save_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "blender":
        download_blender()
    if dataset == "friends":
        download_friends()
    if dataset == "nerfactory":
        if dataset_name is None or dataset_name not in nerfactory_file_ids:
            raise ValueError(
                f"must pass in dataset_name when downloading nerfactory data, options: {nerfactory_file_ids.keys()}"
            )
        download_nerfactory(dataset_name)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(main)


if __name__ == "__main__":
    entrypoint()
