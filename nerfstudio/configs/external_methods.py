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


"""This file contains the configuration for external methods which are not included in this repository."""

import inspect
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tyro
from rich.prompt import Confirm

from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ExternalMethod:
    """External method class. Represents a link to a nerfstudio-compatible method not included in this repository."""

    instructions: str
    """Instructions for installing the method. This will be printed to
    the console when the user tries to use the method."""
    configurations: List[Tuple[str, str]]
    """List of configurations for the method. Each configuration is a tuple of (registered slug, description)
    as it will be printed in --help."""
    pip_package: Optional[str] = None
    """Specifies a pip package if the method can be installed by running `pip install <pip_package>`."""


external_methods = []

# Instruct-NeRF2NeRF
external_methods.append(
    ExternalMethod(
        """[bold yellow]Instruct-NeRF2NeRF[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/in2n.html

To enable Instruct-NeRF2NeRF, you must install it first by running:
  [grey]pip install git+https://github.com/ayaanzhaque/instruct-nerf2nerf[/grey]""",
        configurations=[
            ("in2n", "Instruct-NeRF2NeRF. Full model, used in paper"),
            ("in2n-small", "Instruct-NeRF2NeRF. Half precision model"),
            ("in2n-tiny", "Instruct-NeRF2NeRF. Half prevision with no LPIPS"),
        ],
        pip_package="git+https://github.com/ayaanzhaque/instruct-nerf2nerf",
    )
)

# K-Planes
external_methods.append(
    ExternalMethod(
        """[bold yellow]K-Planes[/bold yellow]
For more information visit https://docs.nerf.studio/nerfology/methods/kplanes.html

To enable K-Planes, you must install it first by running:
  [grey]pip install kplanes-nerfstudio[/grey]""",
        configurations=[
            ("kplanes", "K-Planes model tuned to static blender scenes"),
            ("kplanes-dynamic", "K-Planes model tuned to dynamic DNeRF scenes"),
        ],
        pip_package="kplanes-nerfstudio",
    )
)

# LERF
external_methods.append(
    ExternalMethod(
        """[bold yellow]LERF[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/lerf.html

To enable LERF, you must install it first by running:
  [grey]pip install git+https://github.com/kerrj/lerf[/grey]""",
        configurations=[
            ("lerf-big", "LERF with OpenCLIP ViT-L/14"),
            ("lerf", "LERF with OpenCLIP ViT-B/16, used in paper"),
            ("lerf-lite", "LERF with smaller network and less LERF samples"),
        ],
        pip_package="git+https://github.com/kerrj/lerf",
    )
)

# Tetra-NeRF
external_methods.append(
    ExternalMethod(
        """[bold yellow]Tetra-NeRF[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/tetranerf.html

To enable Tetra-NeRF, you must install it first. Please follow the instructions here:
  https://github.com/jkulhanek/tetra-nerf/blob/master/README.md#installation""",
        configurations=[
            ("tetra-nerf-original", "Tetra-NeRF. Official implementation from the paper"),
            ("tetra-nerf", "Tetra-NeRF. Different sampler - faster and better"),
        ],
    )
)

# NeRFPlayer
external_methods.append(
    ExternalMethod(
        """[bold yellow]NeRFPlayer[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/nerfplayer.html

To enable NeRFPlayer, you must install it first by running:
  [grey]pip install git+https://github.com/lsongx/nerfplayer-nerfstudio[/grey]""",
        configurations=[
            ("nerfplayer-nerfacto", "NeRFPlayer with nerfacto backbone"),
            ("nerfplayer-ngp", "NeRFPlayer with instang-ngp-bounded backbone"),
        ],
    )
)

# Volinga
external_methods.append(
    ExternalMethod(
        """[bold yellow]Volinga[/bold yellow]
For more information visit: https://docs.nerf.studio/extensions/unreal_engine.html

To enable Volinga, you must install it first by running:
  [grey]pip install git+https://github.com/Volinga/volinga-model[/grey]""",
        configurations=[
            (
                "volinga",
                "Real-time rendering model from Volinga. Directly exportable to NVOL format at https://volinga.ai/",
            ),
        ],
        pip_package="git+https://github.com/Volinga/volinga-model",
    )
)

# BioNeRF
external_methods.append(
    ExternalMethod(
        """[bold yellow]BioNeRF[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/bionerf.html

To enable BioNeRF, you must install it first by running:
  [grey]pip install git+https://github.com/Leandropassosjr/ns_bionerf[/grey]""",
        configurations=[("BioNeRF", "BioNeRF. Nerfstudio implementation")],
        pip_package="git+https://github.com/Leandropassosjr/ns_bionerf",
    )
)

# Instruct-GS2GS
external_methods.append(
    ExternalMethod(
        """[bold yellow]Instruct-GS2GS[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/igs2gs.html

To enable Instruct-GS2GS, you must install it first by running:
  [grey]pip install git+https://github.com/cvachha/instruct-gs2gs[/grey]""",
        configurations=[("igs2gs", "Instruct-GS2GS. Full model, used in paper")],
        pip_package="git+https://github.com/cvachha/instruct-gs2gs",
    )
)

# PyNeRF
external_methods.append(
    ExternalMethod(
        """[bold yellow]PyNeRF[/bold yellow]
For more information visit https://docs.nerf.studio/nerfology/methods/pynerf.html

To enable PyNeRF, you must install it first by running:
  [grey]pip install git+https://github.com/hturki/pynerf[/grey]""",
        configurations=[
            ("pynerf", "PyNeRF with proposal network. The default parameters are suited for outdoor scenes."),
            (
                "pynerf-synthetic",
                "PyNeRF with proposal network. The default parameters are suited for synthetic scenes.",
            ),
            (
                "pynerf-occupancy-grid",
                "PyNeRF with occupancy grid. The default parameters are suited for synthetic scenes.",
            ),
        ],
        pip_package="git+https://github.com/hturki/pynerf",
    )
)

# SeaThru-NeRF
external_methods.append(
    ExternalMethod(
        """[bold yellow]Seathru-NeRF[/bold yellow]
For more information visit https://docs.nerf.studio/nerfology/methods/seathru_nerf.html

To enable Seathru-NeRF, you must install it first by running:
  [grey]pip install git+https://github.com/AkerBP/seathru_nerf[/grey]""",
        configurations=[
            ("seathru-nerf", "SeaThru-NeRF for underwater scenes."),
            ("seathru-nerf-lite", "SeaThru-NeRF for underwater scenes (smaller networks and batches)."),
        ],
        pip_package="git+https://github.com/AkerBP/seathru_nerf",
    )
)

# Zip-NeRF
external_methods.append(
    ExternalMethod(
        """[bold yellow]Zip-NeRF[/bold yellow]
For more information visit https://docs.nerf.studio/nerfology/methods/zipnerf.html

To enable Zip-NeRF, you must install it first by running:
  [grey]pip install git+https://github.com/SuLvXiangXin/zipnerf-pytorch#subdirectory=extensions/cuda 
  and pip install git+https://github.com/SuLvXiangXin/zipnerf-pytorch[/grey]""",
        configurations=[
            ("zipnerf", "A pytorch implementation of 'Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields'")
        ],
        pip_package="pip install git+https://github.com/SuLvXiangXin/zipnerf-pytorch",
    )
)

# SIGNeRF
external_methods.append(
    ExternalMethod(
        """[bold yellow]SIGNeRF[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/signerf.html

To enable SIGNeRF, you must install it first by running:
  [grey]pip install git+https://github.com/cgtuebingen/SIGNeRF[/grey] and install Stable Diffusion Web UI see [grey]https://github.com/cgtuebingen/SIGNeRF?tab=readme-ov-file#installation[/grey]""",
        configurations=[
            ("signerf", "SIGNeRF method (high quality) used in paper"),
            ("signerf_nerfacto", "SIGNeRF method combined with Nerfacto (faster training less quality)"),
        ],
        pip_package="git+https://github.com/cgtuebingen/SIGNeRF",
    )
)

# NeRF-SH
external_methods.append(
    ExternalMethod(
        """[bold yellow]NeRF-SH[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/nerf2gs2nerf.html

To enable NeRF-SH, you must install it first by running:
  [grey]pip install git+https://github.com/grasp-lyrl/NeRFtoGSandBack.git#subdirectory=nerfsh[/grey]""",
        configurations=[
            ("nerfsh", "NeRF-SH, used in paper"),
        ],
        pip_package="git+https://github.com/grasp-lyrl/NeRFtoGSandBack.git#subdirectory=nerfsh",
    )
)

# NeRFGS
external_methods.append(
    ExternalMethod(
        """[bold yellow]NeRFGS[/bold yellow]
For more information visit: https://docs.nerf.studio/nerfology/methods/nerf2gs2nerf.html

To enable NeRFGS, you must install it first by running:
  [grey]pip install git+https://github.com/grasp-lyrl/NeRFtoGSandBack.git#subdirectory=nerfgs[/grey]""",
        configurations=[
            ("nerfgs", "NeRFGS, used in paper"),
        ],
        pip_package="git+https://github.com/grasp-lyrl/NeRFtoGSandBack.git#subdirectory=nerfgs",
    )
)


@dataclass
class ExternalMethodDummyTrainerConfig:
    """Dummy trainer config for external methods (a) which do not have an
    implementation in this repository, and (b) are not yet installed. When this
    config is instantiated, we give the user the option to install the method.
    """

    # tyro.conf.Suppress will prevent these fields from appearing as CLI arguments.
    method_name: tyro.conf.Suppress[str]
    method: tyro.conf.Suppress[ExternalMethod]

    def __post_init__(self):
        """Offer to install an external method."""

        # Don't trigger install message from get_external_methods() below; only
        # if this dummy object is instantiated from the CLI.
        if inspect.stack()[2].function == "get_external_methods":
            return

        CONSOLE.print(self.method.instructions)
        if self.method.pip_package and Confirm.ask(
            "\nWould you like to run the install it now?", default=False, console=CONSOLE
        ):
            # Install the method
            install_command = f"{sys.executable} -m pip install {self.method.pip_package}"
            CONSOLE.print(f"Running: [cyan]{install_command}[/cyan]")
            result = subprocess.run(install_command, shell=True, check=False)
            if result.returncode != 0:
                CONSOLE.print("[bold red]Error installing method.[/bold red]")
                sys.exit(1)

        sys.exit(0)


def get_external_methods() -> Tuple[Dict[str, ExternalMethodDummyTrainerConfig], Dict[str, str]]:
    """Returns the external methods trainer configs and the descriptions."""
    method_configs: Dict[str, ExternalMethodDummyTrainerConfig] = {}
    descriptions: Dict[str, str] = {}
    for external_method in external_methods:
        for config_slug, config_description in external_method.configurations:
            method_configs[config_slug] = ExternalMethodDummyTrainerConfig(
                method_name=config_slug, method=external_method
            )
            descriptions[config_slug] = f"""[External, run 'ns-train {config_slug}' to install] {config_description}"""
    return method_configs, descriptions
