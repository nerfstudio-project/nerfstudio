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

# pylint: disable=invalid-name

"""This file contains the configuration for external methods which are not included in this repository."""
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, cast

from rich.prompt import Confirm

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ExternalMethod:
    """External method class. Represents a link to a nerfstudio-compatible method not included in this repository."""

    slug: str
    model_description: str
    instructions: str
    install_commands: Optional[str] = None


pip = f"{sys.executable} -m pip"
external_methods = []

# Instruct-NeRF2NeRF
in2n_install_instructions = """
[bold yellow]Instruct-NeRF2NeRF[/bold yellow]
For more information visit: https://docs.nerf.studio/en/latest/nerfology/methods/in2n.html

To enable Instruct-NeRF2NeRF, you must install it first by running:
  [grey]pip install git+https://github.com/ayaanzhaque/instruct-nerf2nerf[/grey]
""".strip()
in2n_install_script = f"""{pip} install git+https://github.com/ayaanzhaque/instruct-nerf2nerf"""
external_methods.extend(
    [
        ExternalMethod(
            "in2n", "Instruct-NeRF2NeRF. Full model, used in paper", in2n_install_instructions, in2n_install_script
        ),
        ExternalMethod(
            "in2n-small", "Instruct-NeRF2NeRF. Half precision model", in2n_install_instructions, in2n_install_script
        ),
        ExternalMethod(
            "in2n-tiny",
            "Instruct-NeRF2NeRF. Half prevision with no LPIPS",
            in2n_install_instructions,
            in2n_install_script,
        ),
        ExternalMethod(
            "in2n-tiny",
            "Instruct-NeRF2NeRF. Half prevision with no LPIPS",
            in2n_install_instructions,
            in2n_install_script,
        ),
    ]
)

# LERF
lerf_install_instructions = """
[bold yellow]LERF[/bold yellow]
For more information visit: https://docs.nerf.studio/en/latest/nerfology/methods/lerf.html

To enable LERF, you must install it first by running:
  [grey]pip install git+https://github.com/kerrj/lerf[/grey]
""".strip()
lerf_install_script = f"""{pip} install git+https://github.com/kerrj/lerf"""
external_methods.extend(
    [
        ExternalMethod("lerf-big", "LERF with OpenCLIP ViT-L/14", lerf_install_instructions, lerf_install_script),
        ExternalMethod(
            "lerf", "LERF with OpenCLIP ViT-B/16, used in paper", lerf_install_instructions, lerf_install_script
        ),
        ExternalMethod(
            "lerf-lite",
            "LERF with smaller network and less LERF samples",
            lerf_install_instructions,
            lerf_install_script,
        ),
    ]
)


# Tetra-NeRF
tetranerf_install_instructions = """
[bold yellow]Tetra-NeRF[/bold yellow]
For more information visit: https://docs.nerf.studio/en/latest/nerfology/methods/tetranerf.html

To enable Tetra-NeRF, you must install it first. Please follow the instructions here:
  https://github.com/jkulhanek/tetra-nerf/blob/master/README.md#installation
""".strip()
external_methods.extend(
    [
        ExternalMethod(
            "tetra-nerf-original", "Tetra-NeRF. Official implementation from the paper", tetranerf_install_instructions
        ),
        ExternalMethod(
            "tetra-nerf", "Tetra-NeRF. Different sampler - faster and better", tetranerf_install_instructions
        ),
    ]
)


@dataclass
class ExternalMethodTrainerConfig(TrainerConfig):
    """
    Trainer config for external methods which does not have an implementation in this repository.
    """

    _method: ExternalMethod = field(default=cast(ExternalMethod, None))

    def __post_init__(self):
        self.method_name = self._method.slug

    def handle_print_information(self, *_args, **_kwargs):
        """Prints the method information and exits."""
        CONSOLE.print(self._method.instructions)
        if self._method.install_commands and Confirm.ask(
            "\nWould you like to run the install it now?", default=False, console=CONSOLE
        ):
            # Install the method
            CONSOLE.print(f"Running: [cyan]{self._method.install_commands}[/cyan]")
            result = subprocess.run(self._method.install_commands, shell=True, check=False)
            if result.returncode != 0:
                CONSOLE.print("[bold red]Error installing method.[/bold red]")
                sys.exit(1)

        sys.exit(0)

    def __getattribute__(self, __name: str) -> Any:
        out = object.__getattribute__(self, __name)
        if callable(out) and __name not in {"handle_print_information"} and not __name.startswith("__"):
            # We exit early, displaying the message
            return self.handle_print_information
        return out


def get_external_methods() -> Tuple[Dict[str, TrainerConfig], Dict[str, str]]:
    """Returns the external methods trainer configs and the descriptions."""
    method_configs = {}
    descriptions = {}
    for external_method in external_methods:
        method_configs[external_method.slug] = ExternalMethodTrainerConfig(_method=external_method)
        descriptions[external_method.slug] = f"""[External] {external_method.model_description}"""
    return method_configs, descriptions
