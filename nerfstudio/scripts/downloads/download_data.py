# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Download datasets and specific captures from the datasets."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

import gdown
import torch
import tyro
from typing_extensions import Annotated

from nerfstudio.process_data import process_data_utils
from nerfstudio.scripts.downloads.eyeful_tower import EyefulTowerDownload
from nerfstudio.scripts.downloads.utils import DatasetDownload
from nerfstudio.utils import install_checks
from nerfstudio.utils.scripts import run_command


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
class Sitcoms3DDownload(DatasetDownload):
    """Download the sitcoms3D dataset."""

    def download(self, save_dir: Path):
        """Download the sitcoms3D dataset."""

        # https://drive.google.com/file/d/1sgKr0ZO7BQC0FYinAnRSxobIWNucAST5/view?usp=sharing
        sitcoms3d_file_id = "1sgKr0ZO7BQC0FYinAnRSxobIWNucAST5"

        # Download the files
        url = f"https://drive.google.com/uc?id={sitcoms3d_file_id}"
        download_path = str(save_dir / "sitcoms3d.zip")
        gdown.download(url, output=download_path)
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(str(save_dir))
        os.remove(download_path)
        # The folder name of the downloaded dataset is the previously using 'friends/'
        if os.path.exists(str(save_dir / "friends/")):
            os.rename(str(save_dir / "friends/"), str(save_dir / "sitcoms3d/"))


def grab_file_id(zip_url: str) -> str:
    """Get the file id from the google drive zip url."""
    s = zip_url.split("/d/")[1]
    return s.split("/")[0]


nerfstudio_dataset = [
    "Egypt",
    "person",
    "kitchen",
    "plane",
    "dozer",
    "floating-tree",
    "aspen",
    "stump",
    "sculpture",
    "Giannini-Hall",
]
nerfstudio_file_ids = {
    "bww_entrance": grab_file_id("https://drive.google.com/file/d/1ylkRHtfB3n3IRLf2wplpfxzPTq7nES9I/view?usp=sharing"),
    "campanile": grab_file_id("https://drive.google.com/file/d/13aOfGJRRH05pOOk9ikYGTwqFc2L1xskU/view?usp=sharing"),
    "desolation": grab_file_id("https://drive.google.com/file/d/14IzOOQm9KBJ3kPbunQbUTHPnXnmZus-f/view?usp=sharing"),
    "library": grab_file_id("https://drive.google.com/file/d/1Hjbh_-BuaWETQExn2x2qGD74UwrFugHx/view?usp=sharing"),
    "poster": grab_file_id("https://drive.google.com/file/d/1dmjWGXlJnUxwosN6MVooCDQe970PkD-1/view?usp=sharing"),
    "redwoods2": grab_file_id("https://drive.google.com/file/d/1rg-4NoXT8p6vkmbWxMOY6PSG4j3rfcJ8/view?usp=sharing"),
    "storefront": grab_file_id("https://drive.google.com/file/d/16b792AguPZWDA_YC4igKCwXJqW0Tb21o/view?usp=sharing"),
    "vegetation": grab_file_id("https://drive.google.com/file/d/1wBhLQ2odycrtU39y2akVurXEAt9SsVI3/view?usp=sharing"),
    "Egypt": grab_file_id("https://drive.google.com/file/d/1YktD85afw7uitC3nPamusk0vcBdAfjlF/view?view?usp=sharing"),
    "person": grab_file_id("https://drive.google.com/file/d/1HsGMwkPu-R7oU7ySMdoo6Eppq8pKhHF3/view?view?usp=sharing"),
    "kitchen": grab_file_id("https://drive.google.com/file/d/1IRmNyNZSNFidyj93Tt5DtaEU9h6eJdi1/view?view?usp=sharing"),
    "plane": grab_file_id("https://drive.google.com/file/d/1tnv2NC2Iwz4XRYNtziUWvLJjObkZNo2D/view?view?usp=sharing"),
    "dozer": grab_file_id("https://drive.google.com/file/d/1jQJPz5PhzTH--LOcCxvfzV_SDLEp1de3/view?view?usp=sharing"),
    "floating-tree": grab_file_id(
        "https://drive.google.com/file/d/1mVEHcO2ep13WPx92IPDvdQg66vLQwFSy/view?view?usp=sharing"
    ),
    "aspen": grab_file_id("https://drive.google.com/file/d/1X1PQcji_QpxGfMxbETKMeK8aOnWCkuSB/view?view?usp=sharing"),
    "stump": grab_file_id("https://drive.google.com/file/d/1yZFAAEvtw2hs4MXrrkvhVAzEliLLXPB7/view?view?usp=sharing"),
    "sculpture": grab_file_id(
        "https://drive.google.com/file/d/1CUU_k0Et2gysuBn_R5qenDMfYXEhNsd1/view?view?usp=sharing"
    ),
    "Giannini-Hall": grab_file_id(
        "https://drive.google.com/file/d/1UkjWXLN4qybq_a-j81FsTKghiXw39O8E/view?view?usp=sharing"
    ),
    "all": None,
    "nerfstudio-dataset": nerfstudio_dataset,
}

if TYPE_CHECKING:
    NerfstudioCaptureName = str
else:
    NerfstudioCaptureName = tyro.extras.literal_type_from_choices(nerfstudio_file_ids.keys())


def download_capture_name(save_dir: Path, dataset_name: str, capture_name: str, capture_name_to_file_id: dict):
    """Download specific captures a given dataset and capture name."""

    file_id_or_zip_url = capture_name_to_file_id[capture_name]
    if file_id_or_zip_url.endswith(".zip"):
        url = file_id_or_zip_url  # zip url
        target_path = str(save_dir / dataset_name / capture_name)
        os.makedirs(target_path, exist_ok=True)
        download_path = Path(f"{target_path}.zip")
        tmp_path = str(save_dir / dataset_name / f".temp_{capture_name}")
        shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)
        try:
            os.remove(download_path)
        except OSError:
            pass
        run_command(f"wget {url} -O {download_path}", verbose=True)
    else:
        url = f"https://drive.google.com/uc?id={file_id_or_zip_url}"  # file id
        target_path = str(save_dir / dataset_name / capture_name)
        os.makedirs(target_path, exist_ok=True)
        download_path = Path(f"{target_path}.zip")
        tmp_path = str(save_dir / dataset_name / f".temp_{capture_name}")
        shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)
        try:
            os.remove(download_path)
        except OSError:
            pass
        gdown.download(url, output=str(download_path))
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    inner_folders = os.listdir(tmp_path)
    assert len(inner_folders) == 1, f"There is more than one folder inside this zip file: {inner_folders}"
    folder = os.path.join(tmp_path, inner_folders[0])
    shutil.rmtree(target_path)
    shutil.move(folder, target_path)
    shutil.rmtree(tmp_path)
    os.remove(download_path)


@dataclass
class NerfstudioDownload(DatasetDownload):
    """
    Download data in the Nerfstudio format.
    If you are interested in the Nerfstudio Dataset subset from the SIGGRAPH 2023 paper,
    you can obtain that by using --capture-name nerfstudio-dataset or by visiting Google Drive directly at:
    https://drive.google.com/drive/folders/19TV6kdVGcmg3cGZ1bNIUnBBMD-iQjRbG?usp=drive_link.
    """

    capture_name: NerfstudioCaptureName = "bww_entrance"

    def download(self, save_dir: Path):
        """Download the nerfstudio dataset."""
        if self.capture_name == "all":
            for capture_name in nerfstudio_file_ids:
                if capture_name not in ("all", "nerfstudio-dataset"):
                    download_capture_name(save_dir, "nerfstudio", capture_name, nerfstudio_file_ids)
            return

        if self.capture_name == "nerfstudio-dataset":
            for capture_name in nerfstudio_dataset:
                if capture_name not in ("all", "nerfstudio-dataset"):
                    download_capture_name(save_dir, "nerfstudio", capture_name, nerfstudio_file_ids)
            return

        download_capture_name(save_dir, "nerfstudio", self.capture_name, capture_name_to_file_id=nerfstudio_file_ids)


record3d_file_ids = {
    "bear": grab_file_id("https://drive.google.com/file/d/1WRZohWMRj0nNlYFIEBwkddDoGPvLTzkR/view?usp=sharing")
}

if TYPE_CHECKING:
    Record3dCaptureName = str
else:
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

        install_checks.check_curl_installed()
        final_path = save_dir / Path("dnerf")
        if os.path.exists(final_path):
            shutil.rmtree(str(final_path))
        download_path = save_dir / "dnerf_data.zip"
        subprocess.run(
            ["curl", "-L", "https://www.dropbox.com/s/raw/0bf6fl0ye2vz3vr/data.zip", "-o", download_path], check=True
        )
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(str(save_dir))
        unzip_path = save_dir / Path("data")
        final_path = save_dir / Path("dnerf")
        unzip_path.rename(final_path)
        if download_path.exists():
            download_path.unlink()


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

if TYPE_CHECKING:
    PhototourismCaptureName = str
else:
    PhototourismCaptureName = tyro.extras.literal_type_from_choices(phototourism_downloads.keys())


@dataclass
class PhototourismDownload(DatasetDownload):
    """Download the phototourism dataset."""

    capture_name: PhototourismCaptureName = "brandenburg-gate"

    def download(self, save_dir: Path):
        """Download a PhotoTourism dataset: https://www.cs.ubc.ca/~kmyi/imw2020/data.html"""

        install_checks.check_curl_installed()
        if self.capture_name == "all":
            for capture_name in phototourism_downloads:
                if capture_name != "all":
                    PhototourismDownload(capture_name=capture_name).download(save_dir)
            return

        assert (
            self.capture_name in phototourism_downloads
        ), f"Capture name {self.capture_name} not found in {phototourism_downloads.keys()}"
        url = phototourism_downloads[self.capture_name]
        target_path = str(save_dir / "phototourism" / self.capture_name)
        os.makedirs(target_path, exist_ok=True)
        download_path = Path(f"{target_path}.tar.gz")
        tmp_path = str(save_dir / "phototourism" / f".temp_{self.capture_name}")
        shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)

        subprocess.run(["curl", "-L", url, "-o", download_path], check=True)

        with tarfile.open(download_path, "r:gz") as tar_ref:
            tar_ref.extractall(str(tmp_path))

        inner_folders = os.listdir(tmp_path)
        assert len(inner_folders) == 1, f"There is more than one folder inside this zip file: {inner_folders}"
        folder = os.path.join(tmp_path, inner_folders[0])
        shutil.rmtree(target_path)
        shutil.move(folder, target_path)
        shutil.rmtree(tmp_path)
        os.remove(download_path)


# credit to https://autonomousvision.github.io/sdfstudio/

sdfstudio_downloads = {
    "sdfstudio-demo-data": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/sdfstudio-demo-data.tar",
    "dtu": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/DTU.tar",
    "replica": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/Replica.tar",
    "scannet": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/scannet.tar",
    "tanks-and-temple": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/tnt_advanced.tar",
    "tanks-and-temple-highres": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/highresTNT.tar",
    "heritage": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/Heritage-Recon.tar",
    "neural-rgbd-data": "http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip",
    "all": None,
}

if TYPE_CHECKING:
    SDFstudioCaptureName = str
else:
    SDFstudioCaptureName = tyro.extras.literal_type_from_choices(sdfstudio_downloads.keys())


@dataclass
class SDFstudioDemoDownload(DatasetDownload):
    """Download the sdfstudio dataset."""

    dataset_name: SDFstudioCaptureName = "sdfstudio-demo-data"

    def download(self, save_dir: Path):
        """Download the sdfstudio dataset (https://autonomousvision.github.io/sdfstudio/)."""
        # TODO: give this code the same structure as download_nerfstudio

        if self.dataset_name == "all":
            for dataset_name in sdfstudio_downloads:
                if dataset_name != "all":
                    SDFstudioDemoDownload(dataset_name=dataset_name).download(save_dir)
            return

        assert (
            self.dataset_name in sdfstudio_downloads
        ), f"Capture name {self.dataset_name} not found in {sdfstudio_downloads.keys()}"

        url = sdfstudio_downloads[self.dataset_name]

        target_path = str(save_dir / "sdfstudio" / self.dataset_name)
        os.makedirs(target_path, exist_ok=True)

        file_format = url[-4:]

        download_path = Path(f"{target_path}{file_format}")
        tmp_path = str(save_dir / "sdfstudio" / f".temp_{self.dataset_name}")
        shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)

        subprocess.run(["curl", "-L", url, "-o", download_path], check=True)
        if file_format == ".tar":
            with tarfile.open(download_path, "r") as tar_ref:
                tar_ref.extractall(str(tmp_path))
        elif file_format == ".zip":
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(str(target_path))
            return
        else:
            raise NotImplementedError

        inner_folders = os.listdir(tmp_path)
        assert len(inner_folders) == 1, f"There is more than one folder inside this zip file: {inner_folders}"
        folder = os.path.join(tmp_path, inner_folders[0])
        shutil.rmtree(target_path)
        shutil.move(folder, target_path)
        shutil.rmtree(tmp_path)
        os.remove(download_path)


nerfosr_downloads = {
    "europa": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=europa&downloadStartSecret=0k2r95c1fdej",
    "lk2": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=lk2&downloadStartSecret=w8kuvjzmchc",
    "lwp": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=lwp&downloadStartSecret=gtnc4vmkcjq",
    "rathaus": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=rathaus&downloadStartSecret=7372aewy6rr",
    "schloss": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=schloss&downloadStartSecret=y8t00nqx0h",
    "st": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=st&downloadStartSecret=kl9ptuxe8v",
    "stjacob": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=stjacob&downloadStartSecret=sntsim6ebvm",
    "stjohann": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=stjohann&downloadStartSecret=g80ug1fsbmh",
    "trevi": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk/download?path=%2FData&files=trevi&downloadStartSecret=ot1483bigjm",
    "all": None,
}


if TYPE_CHECKING:
    NeRFOSRCaptureName = str
else:
    NeRFOSRCaptureName = tyro.extras.literal_type_from_choices(nerfosr_downloads.keys())


@dataclass
class NeRFOSRDownload(DatasetDownload):
    """Download the NeRF-OSR dataset."""

    capture_name: NeRFOSRCaptureName = "europa"

    def download(self, save_dir: Path):
        """Download the NeRF-OSR dataset: https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk"""

        if self.capture_name == "all":
            for capture_name in nerfosr_downloads:
                if capture_name != "all":
                    NeRFOSRDownload(capture_name=capture_name).download(save_dir)
            return

        assert (
            self.capture_name in nerfosr_downloads
        ), f"Capture name {self.capture_name} not found in {nerfosr_downloads.keys()}"
        url = nerfosr_downloads[self.capture_name]
        target_path = str(save_dir / "nerfosr" / self.capture_name)
        os.makedirs(target_path, exist_ok=True)
        download_path = Path(f"{target_path}.zip")
        tmp_path = str(save_dir / "nerfosr" / f".temp_{self.capture_name}")
        shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)

        subprocess.run(["curl", "-L", url, "-o", download_path], check=True)

        # Extract the zip file
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(tmp_path)

        inner_folders = os.listdir(tmp_path)
        assert len(inner_folders) == 1, f"There is more than one folder inside this zip file: {inner_folders}"
        folder = os.path.join(tmp_path, inner_folders[0])
        shutil.rmtree(target_path)
        shutil.move(folder, target_path)
        shutil.rmtree(tmp_path)
        os.remove(download_path)


mill19_downloads = {
    "building": "https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz",
    "rubble": "https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz",
    "all": None,
}

if TYPE_CHECKING:
    Mill19CaptureName = str
else:
    Mill19CaptureName = tyro.extras.literal_type_from_choices(mill19_downloads.keys())


@dataclass
class Mill19Download(DatasetDownload):
    """Download the Mill 19 dataset."""

    capture_name: Mill19CaptureName = "building"

    def download(self, save_dir: Path) -> None:
        """Download a Mill 19 dataset: https://meganerf.cmusatyalab.org/#data"""

        install_checks.check_curl_installed()
        if self.capture_name == "all":
            for capture_name in mill19_downloads:
                if capture_name != "all":
                    Mill19Download(capture_name=capture_name).download(save_dir)
            return

        assert (
            self.capture_name in mill19_downloads
        ), f"Capture name {self.capture_name} not found in {mill19_downloads.keys()}"
        url = mill19_downloads[self.capture_name]
        target_path = save_dir / "mill19" / self.capture_name
        target_path.mkdir(parents=True, exist_ok=True)
        download_path = Path(f"{target_path}.tgz")
        tmp_path = save_dir / "mill19" / f".temp_{self.capture_name}"
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(["curl", "-L", url, "-o", download_path], check=True)

        with tarfile.open(download_path, "r:gz") as tar_ref:
            tar_ref.extractall(tmp_path)

        inner_folders = list(tmp_path.iterdir())
        assert len(inner_folders) == 1, f"There is more than one folder inside this zip file: {inner_folders}"
        folder = inner_folders[0]
        shutil.rmtree(target_path)
        folder.rename(target_path)
        shutil.rmtree(tmp_path)
        download_path.unlink()

        # Convert data layout into what the nerfstudio dataparser expects
        meta = {"frames": []}
        for subdir, prefix in [("train", "train_"), ("val", "eval_")]:
            copied_images = process_data_utils.copy_images(
                target_path / subdir / "rgbs",
                image_dir=target_path / "images",
                image_prefix=prefix,
                num_downscales=3,
                verbose=True,
                keep_image_dir=True,
            )

            split_filepaths = []
            for image_path, new_image_path in copied_images.items():
                metadata_path = image_path.parent.parent / "metadata" / f"{image_path.stem}.pt"
                metadata = torch.load(metadata_path, map_location="cpu")
                c2w = torch.eye(4)
                c2w[:3] = metadata["c2w"]
                file_path = str(Path("images") / f"{new_image_path.name}")
                split_filepaths.append(file_path)
                meta["frames"].append(
                    {
                        "file_path": file_path,
                        "fl_x": metadata["intrinsics"][0].item(),
                        "fl_y": metadata["intrinsics"][1].item(),
                        "cx": metadata["intrinsics"][2].item(),
                        "cy": metadata["intrinsics"][3].item(),
                        "w": metadata["W"],
                        "h": metadata["H"],
                        "transform_matrix": c2w.tolist(),
                    }
                )
            meta[f"{subdir}_filenames"] = split_filepaths

        with (target_path / "transforms.json").open("w") as f:
            json.dump(meta, f, indent=4)

        shutil.rmtree(target_path / "train")
        shutil.rmtree(target_path / "val")


Commands = Union[
    Annotated[BlenderDownload, tyro.conf.subcommand(name="blender")],
    Annotated[Sitcoms3DDownload, tyro.conf.subcommand(name="sitcoms3d")],
    Annotated[NerfstudioDownload, tyro.conf.subcommand(name="nerfstudio")],
    Annotated[Record3dDownload, tyro.conf.subcommand(name="record3d")],
    Annotated[DNerfDownload, tyro.conf.subcommand(name="dnerf")],
    Annotated[PhototourismDownload, tyro.conf.subcommand(name="phototourism")],
    Annotated[SDFstudioDemoDownload, tyro.conf.subcommand(name="sdfstudio")],
    Annotated[NeRFOSRDownload, tyro.conf.subcommand(name="nerfosr")],
    Annotated[Mill19Download, tyro.conf.subcommand(name="mill19")],
    Annotated[EyefulTowerDownload, tyro.conf.subcommand(name="eyefultower")],
]


def main(
    dataset: DatasetDownload,
):
    """Script to download existing datasets.
    We currently support the datasets listed above in the Commands.

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
