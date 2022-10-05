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


def download_blender(save_dir: Path):
    """Download the blender dataset."""
    # TODO: give this code the same structure as download_nerfstudio

    # https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
    blender_file_id = "18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"

    final_path = save_dir / Path("blender")
    if os.path.exists(final_path):
        shutil.rmtree(str(final_path))
    url = f"https://drive.google.com/uc?id={blender_file_id}"
    download_path = save_dir / "blender_data.zip"
    gdown.download(url, output=str(download_path))
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(str(save_dir))
    unzip_path = save_dir / Path("nerf_synthetic")
    final_path = save_dir / Path("blender")
    unzip_path.rename(final_path)
    download_path.unlink(missing_ok=True)


def download_friends(save_dir: Path):
    """Download the friends dataset."""

    # https://drive.google.com/file/d/1sgKr0ZO7BQC0FYinAnRSxobIWNucAST5/view?usp=sharing
    friends_file_id = "1sgKr0ZO7BQC0FYinAnRSxobIWNucAST5"

    # Download the files
    url = f"https://drive.google.com/uc?id={friends_file_id}"
    download_path = str(save_dir / "friends.zip")
    gdown.download(url, output=download_path)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(str(save_dir))
    os.remove(download_path)


def grab_file_id(zip_url: str) -> str:
    """Get the file id from the google drive zip url."""
    s = zip_url.split("/d/")[1]
    return s.split("/")[0]


# https://drive.google.com/drive/folders/1Wh66z3qQTZ8o2MwPPwYOrdwQtXUEQFyq?usp=sharing
nerfstudio_file_ids = {
    "bunny": grab_file_id("https://drive.google.com/file/d/1oVytSgd1gQclF0CybJ6JdXDmaqKOx7Wg/view?usp=sharing"),
    "bww_entrance": grab_file_id("https://drive.google.com/file/d/1ylkRHtfB3n3IRLf2wplpfxzPTq7nES9I/view?usp=sharing"),
    "bww_tree": grab_file_id("https://drive.google.com/file/d/1N_OQejT0MblK1UP05KORxy25SXjKqZN1/view?usp=sharing"),
    "campanelle": grab_file_id("https://drive.google.com/file/d/13aOfGJRRH05pOOk9ikYGTwqFc2L1xskU/view?usp=sharing"),
    "desolation": grab_file_id("https://drive.google.com/file/d/14IzOOQm9KBJ3kPbunQbUTHPnXnmZus-f/view?usp=sharing"),
    "dozer": grab_file_id("https://drive.google.com/file/d/1-OR5F_V5S4s-yzxohbwTylaXjzYLu8ZR/view?usp=sharing"),
    "japanese_maple": grab_file_id(
        "https://drive.google.com/file/d/1ytCnaAEqm-fIziuQXbBp5yVbZ_cmkHlj/view?usp=sharing"
    ),
    "kushikatsu": grab_file_id("https://drive.google.com/file/d/1mTNbDW1EyX_fi_ffeP-6Q_0STxr3WfU6/view?usp=sharing"),
    "library": grab_file_id("https://drive.google.com/file/d/1Hjbh_-BuaWETQExn2x2qGD74UwrFugHx/view?usp=sharing"),
    "poster": grab_file_id("https://drive.google.com/file/d/1dmjWGXlJnUxwosN6MVooCDQe970PkD-1/view?usp=sharing"),
    "redwoods2": grab_file_id("https://drive.google.com/file/d/1rg-4NoXT8p6vkmbWxMOY6PSG4j3rfcJ8/view?usp=sharing"),
    "sf_street": grab_file_id("https://drive.google.com/file/d/1DbLyptL6my2QprEVtYuW2uzgp9JAK5Wz/view?usp=sharing"),
    "storefront": grab_file_id("https://drive.google.com/file/d/16b792AguPZWDA_YC4igKCwXJqW0Tb21o/view?usp=sharing"),
    "vegetation": grab_file_id("https://drive.google.com/file/d/1wBhLQ2odycrtU39y2akVurXEAt9SsVI3/view?usp=sharing"),
}


def download_nerfstudio(save_dir: Path, capture_name: str):
    """Download specific captures from the nerfstudio dataset."""

    url = f"https://drive.google.com/uc?id={nerfstudio_file_ids[capture_name]}"
    target_path = str(save_dir / f"nerfstudio/{capture_name}")
    os.makedirs(target_path, exist_ok=True)
    download_path = Path(f"{target_path}.zip")
    tmp_path = str(save_dir / ".temp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path, exist_ok=True)
    gdown.download(url, output=str(download_path))
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    inner_folders = os.listdir(tmp_path)
    assert len(inner_folders) == 1, "There is more than one folder inside this zip file."
    folder = os.path.join(tmp_path, inner_folders[0])
    shutil.rmtree(target_path)
    shutil.move(folder, target_path)
    shutil.rmtree(tmp_path)
    os.remove(download_path)


def main(
    dataset: Literal["blender", "friends", "nerfstudio"],
    capture_name: Optional[str] = None,
    save_dir: Path = Path("data/"),
):
    """Main download script to download all data.

    Args:
        dataset: The dataset to download (from).
        capture_name: The capture name to download (from the dataset).
        save_dir: The directory to save the data to.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "blender":
        download_blender(save_dir)
    if dataset == "friends":
        download_friends(save_dir)
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
        download_nerfstudio(save_dir, capture_name)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(main)


if __name__ == "__main__":
    entrypoint()
