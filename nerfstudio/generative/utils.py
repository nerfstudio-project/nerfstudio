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

"""Utility helper functions for diffusion models"""

import sys
from nerfstudio.utils.rich_utils import CONSOLE


class CatchMissingPackages:
    """Class to catch missing environment packages related to diffusion models."""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
        CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
        CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
        sys.exit(1)

    def __getattr__(self, attr):
        return self.__call__
