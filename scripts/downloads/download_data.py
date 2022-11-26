"""Download datasets and specific captures from the datasets."""
from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gdown
import tyro
from rich.console import Console
from typing_extensions import Annotated

from nerfstudio.configs.base_config import PrintableConfig

CONSOLE = Console(width=120)


@dataclass
class DatasetDownload(PrintableConfig):
    """Download a dataset"""

    capture_name = None

    save_dir: Path = Path("data/")
    """The directory to save the dataset to"""

    def download(self, save_dir: Path) -> None:
        """Download the dataset"""
        raise NotImplementedError


@dataclass
class BlenderDownload(DatasetDownload):
    """Download the blender dataset."""

    def download(self, save_dir: Path):
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
        if download_path.exists():
            download_path.unlink()


@dataclass
class FriendsDownload(DatasetDownload):
    """Download the friends dataset."""

    def download(self, save_dir: Path):
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


nerfstudio_file_ids = {
    "bww_entrance": grab_file_id("https://drive.google.com/file/d/1ylkRHtfB3n3IRLf2wplpfxzPTq7nES9I/view?usp=sharing"),
    "campanile": grab_file_id("https://drive.google.com/file/d/13aOfGJRRH05pOOk9ikYGTwqFc2L1xskU/view?usp=sharing"),
    "desolation": grab_file_id("https://drive.google.com/file/d/14IzOOQm9KBJ3kPbunQbUTHPnXnmZus-f/view?usp=sharing"),
    "dozer": grab_file_id("https://drive.google.com/file/d/1-OR5F_V5S4s-yzxohbwTylaXjzYLu8ZR/view?usp=sharing"),
    "library": grab_file_id("https://drive.google.com/file/d/1Hjbh_-BuaWETQExn2x2qGD74UwrFugHx/view?usp=sharing"),
    "poster": grab_file_id("https://drive.google.com/file/d/1dmjWGXlJnUxwosN6MVooCDQe970PkD-1/view?usp=sharing"),
    "redwoods2": grab_file_id("https://drive.google.com/file/d/1rg-4NoXT8p6vkmbWxMOY6PSG4j3rfcJ8/view?usp=sharing"),
    "storefront": grab_file_id("https://drive.google.com/file/d/16b792AguPZWDA_YC4igKCwXJqW0Tb21o/view?usp=sharing"),
    "vegetation": grab_file_id("https://drive.google.com/file/d/1wBhLQ2odycrtU39y2akVurXEAt9SsVI3/view?usp=sharing"),
    "all": None,
}

NerfstudioCaptureName = tyro.extras.literal_type_from_choices(nerfstudio_file_ids.keys())


def download_capture_name(save_dir: Path, dataset_name: str, capture_name: str, capture_name_to_file_id: dict):
    """Download specific captures a given dataset and capture name."""

    url = f"https://drive.google.com/uc?id={capture_name_to_file_id[capture_name]}"
    target_path = str(save_dir / f"{dataset_name}/{capture_name}")
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


@dataclass
class NerfstudioDownload(DatasetDownload):
    """Download the nerfstudio dataset."""

    capture_name: NerfstudioCaptureName = "bww_entrance"

    def download(self, save_dir: Path):
        """Download the nerfstudio dataset."""
        if self.capture_name == "all":
            for capture_name in nerfstudio_file_ids:
                if capture_name != "all":
                    download_capture_name(save_dir, "nerfstudio", capture_name, nerfstudio_file_ids)
            return

        download_capture_name(save_dir, "nerfstudio", self.capture_name, capture_name_to_file_id=nerfstudio_file_ids)


record3d_file_ids = {
    "bear": grab_file_id("https://drive.google.com/file/d/1WRZohWMRj0nNlYFIEBwkddDoGPvLTzkR/view?usp=sharing")
}

Record3dCaptureName = tyro.extras.literal_type_from_choices(record3d_file_ids.keys())


@dataclass
class Record3dDownload(DatasetDownload):
    """Download the record3d dataset."""

    capture_name: Record3dCaptureName = "bear"

    def download(self, save_dir: Path):
        download_capture_name(save_dir, "record3d", self.capture_name, capture_name_to_file_id=record3d_file_ids)


@dataclass
class DNerfDownload(DatasetDownload):
    """Download the dnerf dataset."""

    def download(self, save_dir: Path):
        """Download the D-NeRF dataset (https://github.com/albertpumarola/D-NeRF)."""
        # TODO: give this code the same structure as download_nerfstudio

        final_path = save_dir / Path("dnerf")
        if os.path.exists(final_path):
            shutil.rmtree(str(final_path))
        download_path = save_dir / "dnerf_data.zip"
        os.system(f"curl -L https://www.dropbox.com/s/raw/0bf6fl0ye2vz3vr/data.zip > {download_path}")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(str(save_dir))
        unzip_path = save_dir / Path("data")
        final_path = save_dir / Path("dnerf")
        unzip_path.rename(final_path)
        if download_path.exists():
            download_path.unlink()


# pylint: disable=line-too-long
phototourism_downloads = {
    "brandenburg-gate": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/brandenburg_gate.tar.gz",
    "buckingham-palace": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/buckingham_palace.tar.gz",
    "colosseum-exterior": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/colosseum_exterior.tar.gz",
    "grand-palace-brussels": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/grand_place_brussels.tar.gz",
    "notre-dame-facade": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/notre_dame_front_facade.tar.gz",
    "westminster-palace": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/palace_of_westminster.tar.gz",
    "pantheon-exterior": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/pantheon_exterior.tar.gz",
    "taj-mahal": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/taj_mahal.tar.gz",
    "temple-nara": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/temple_nara_japan.tar.gz",
    "trevi-fountain": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/trevi_fountain.tar.gz",
    "all": None,
}

PhototourismCaptureName = tyro.extras.literal_type_from_choices(phototourism_downloads.keys())


@dataclass
class PhototourismDownload(DatasetDownload):
    """Download the phototourism dataset."""

    capture_name: PhototourismCaptureName = "brandenburg-gate"

    def download(self, save_dir: Path):
        """Download a PhotoTourism dataset: https://www.cs.ubc.ca/~kmyi/imw2020/data.html"""

        if self.capture_name == "all":
            for capture_name in phototourism_downloads:
                if capture_name != "all":
                    PhototourismDownload(capture_name=capture_name).download(save_dir)
            return

        assert (
            self.capture_name in phototourism_downloads
        ), f"Capture name {self.capture_name} not found in {phototourism_downloads.keys()}"
        url = phototourism_downloads[self.capture_name]
        target_path = str(save_dir / f"phototourism/{self.capture_name}")
        os.makedirs(target_path, exist_ok=True)
        download_path = Path(f"{target_path}.tar.gz")
        tmp_path = str(save_dir / ".temp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)

        os.system(f"curl -L {url} > {download_path}")

        with tarfile.open(download_path, "r:gz") as tar_ref:
            tar_ref.extractall(str(tmp_path))

        inner_folders = os.listdir(tmp_path)
        assert len(inner_folders) == 1, "There is more than one folder inside this zip file."
        folder = os.path.join(tmp_path, inner_folders[0])
        shutil.rmtree(target_path)
        shutil.move(folder, target_path)
        shutil.rmtree(tmp_path)
        os.remove(download_path)


Commands = Union[
    Annotated[BlenderDownload, tyro.conf.subcommand(name="blender")],
    Annotated[FriendsDownload, tyro.conf.subcommand(name="friends")],
    Annotated[NerfstudioDownload, tyro.conf.subcommand(name="nerfstudio")],
    Annotated[Record3dDownload, tyro.conf.subcommand(name="record3d")],
    Annotated[DNerfDownload, tyro.conf.subcommand(name="dnerf")],
    Annotated[PhototourismDownload, tyro.conf.subcommand(name="phototourism")],
]


def main(
    dataset: DatasetDownload,
):
    """Script to download existing datasets.
    We currently support the following datasets:
    - nerfstudio: Growing collection of real-world scenes. Use the `capture_name` argument to specify
        which capture to download.
    - blender: Blender synthetic scenes realeased with NeRF.
    - friends: Friends TV show scenes.
    - record3d: Record3d dataset.
    - dnerf: D-NeRF dataset.
    - phototourism: PhotoTourism dataset. Use the `capture_name` argument to specify which capture to download.

    Args:
        dataset: The dataset to download (from).
    """
    dataset.save_dir.mkdir(parents=True, exist_ok=True)

    dataset.download(dataset.save_dir)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
