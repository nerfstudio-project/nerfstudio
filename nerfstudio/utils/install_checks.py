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

"""Helpers for checking if programs are installed"""

import shutil
import sys

from rich.console import Console

CONSOLE = Console(width=120)


def check_ffmpeg_installed():
    """Checks if ffmpeg is installed."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        CONSOLE.print("[bold red]Could not find ffmpeg. Please install ffmpeg.")
        print("See https://ffmpeg.org/download.html for installation instructions.")
        print("ffmpeg is only necessary if using videos as input.")
        sys.exit(1)


def check_colmap_installed():
    """Checks if colmap is installed."""
    colmap_path = shutil.which("colmap")
    if colmap_path is None:
        CONSOLE.print("[bold red]Could not find COLMAP. Please install COLMAP.")
        print("See https://colmap.github.io/install.html for installation instructions.")
        sys.exit(1)
