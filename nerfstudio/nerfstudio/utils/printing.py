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

"""A collection of common strings and print statements used throughout the codebase."""

from math import floor, log

from nerfstudio.utils.rich_utils import CONSOLE


def print_tcnn_speed_warning(module_name: str):
    """Prints a warning about the speed of the TCNN."""
    CONSOLE.line()
    CONSOLE.print(f"[bold yellow]WARNING: Using a slow implementation for the {module_name} module. ")
    CONSOLE.print(
        "[bold yellow]:person_running: :person_running: "
        + "Install tcnn for speedups :person_running: :person_running:"
    )
    CONSOLE.print("[yellow]pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")
    CONSOLE.line()


def human_format(num):
    """Format a number in a more human readable way

    Args:
        num: number to format
    """
    units = ["", "K", "M", "B", "T", "P"]
    k = 1000.0
    magnitude = int(floor(log(num, k)))
    return f"{(num / k**magnitude):.2f} {units[magnitude]}"
