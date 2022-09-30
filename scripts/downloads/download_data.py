"""Download datasets and specific captures from the datasets."""
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Literal, Optional

import dcargs
import gdown
from rich.console import Console

console = Console(width=120)


def download_blender():
    """Download the blender dataset."""

    # https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
    blender_file_id = "18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"

    url = f"https://drive.google.com/uc?id={blender_file_id}"
    download_path = Path("data/blender_data.zip")
    gdown.download(url, output=str(download_path), quiet=False)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall("data/")
    unzip_path = Path("data/nerf_synthetic")
    final_path = Path("data/blender")
    unzip_path.rename(final_path)
    download_path.unlink(missing_ok=True)


def download_friends():
    """Download the friends dataset."""

    # https://drive.google.com/file/d/1sgKr0ZO7BQC0FYinAnRSxobIWNucAST5/view?usp=sharing
    friends_file_id = "1sgKr0ZO7BQC0FYinAnRSxobIWNucAST5"

    # Download the files
    url = f"https://drive.google.com/uc?id={friends_file_id}"
    download_path = "data/friends.zip"
    gdown.download(url, output=download_path, quiet=False)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall("data/")
    os.remove(download_path)


# https://drive.google.com/drive/folders/1Wh66z3qQTZ8o2MwPPwYOrdwQtXUEQFyq?usp=sharing
nerfstudio_file_ids = {
    "dozer": "1-OR5F_V5S4s-yzxohbwTylaXjzYLu8ZR",
    "sf_street": "1DbLyptL6my2QprEVtYuW2uzgp9JAK5Wz",
    "poster": "1dmjWGXlJnUxwosN6MVooCDQe970PkD-1",
}


def download_nerfstudio(capture_name: str):
    """Download specific captures from the nerfstudio dataset."""

    url = f"https://drive.google.com/uc?id={nerfstudio_file_ids[capture_name]}"
    target_path = f"data/nerfstudio/{capture_name}"
    os.makedirs(target_path, exist_ok=True)
    download_path = Path(f"{target_path}.zip")
    gdown.download(url, output=str(download_path), quiet=False)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(f"/tmp/{target_path}")
    inner_folders = os.listdir(f"/tmp/{target_path}")
    assert len(inner_folders) == 1, "There is more than one folder inside this zip file."
    folder = os.path.join(f"/tmp/{target_path}", inner_folders[0])
    shutil.rmtree(target_path)
    shutil.move(folder, target_path)
    shutil.rmtree(f"/tmp/{target_path}")
    os.remove(download_path)


def main(
    dataset: Literal["blender", "friends", "nerfstudio"],
    capture_name: Optional[str] = None,
):
    """Main download script to download all data"""
    save_dir = Path("data/")
    save_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "blender":
        download_blender()
    if dataset == "friends":
        download_friends()
    if dataset == "nerfstudio":
        if capture_name is None:
            capture_names = sorted(nerfstudio_file_ids.keys())
            console.print(
                "[bold yellow]You must pass in --capture-name when downloading from the nerfstudio dataset."
                f" Use one of the following: \n\t {capture_names}"
            )
            sys.exit()
        if capture_name not in nerfstudio_file_ids:
            capture_names = sorted(nerfstudio_file_ids.keys())
            console.print(f"[bold yellow]Invalid --capture-name choice. Use one of the following: \n {capture_names}")
            sys.exit()
        download_nerfstudio(capture_name)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(main)


if __name__ == "__main__":
    entrypoint()
