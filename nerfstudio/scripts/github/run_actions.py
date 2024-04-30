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

#!/usr/bin/env python
"""Simple yaml debugger"""
import subprocess
import sys

import tyro
import yaml
from rich.style import Style

from nerfstudio.utils.rich_utils import CONSOLE

LOCAL_TESTS = ["Run license checks", "Run Ruff Linter", "Run Ruff Formatter", "Run Pyright", "Test with pytest"]


def run_command(command: str, continue_on_fail: bool = False) -> bool:
    """Run a command kill actions if it fails

    Args:
        command: command to run
        continue_on_fail: whether to continue running commands if the current one fails.
    """
    ret_code = subprocess.call(command, shell=True)
    if ret_code != 0:
        CONSOLE.print(f"[bold red]Error: `{command}` failed.")
        if not continue_on_fail:
            sys.exit(1)
    return ret_code == 0


def run_github_actions_file(filename: str, continue_on_fail: bool = False):
    """Run a github actions file locally.

    Args:
        filename: Which yml github actions file to run.
        continue_on_fail: Whether or not to continue running actions commands if the current one fails
    """
    with open(filename, "rb") as f:
        my_dict = yaml.safe_load(f)
    steps = my_dict["jobs"]["build"]["steps"]

    success = True

    for step in steps:
        if "name" in step and step["name"] in LOCAL_TESTS:
            curr_command = step["run"].replace("\n", ";").replace("\\", "")
            if curr_command.startswith("ruff"):
                if "ruff check" in curr_command:
                    curr_command = curr_command.replace("ruff check", "ruff check --fix")

                curr_command = curr_command.replace(" --check", "")
                curr_command = curr_command.replace(" --diff", "")
                curr_command = curr_command.replace(" --output-format=github", "")

            CONSOLE.line()
            CONSOLE.rule(f"[bold green]Running: {curr_command}")
            success = success and run_command(curr_command, continue_on_fail=continue_on_fail)
        else:
            skip_name = step["name"] if "name" in step else step["uses"]
            CONSOLE.print(f"Skipping {skip_name}")

    # Add checks for building documentation
    CONSOLE.line()
    CONSOLE.rule("[bold green]Adding notebook documentation metadata")
    success = success and run_command(
        "python nerfstudio/scripts/docs/add_nb_tags.py", continue_on_fail=continue_on_fail
    )
    CONSOLE.line()
    CONSOLE.rule("[bold green]Building Documentation")
    success = success and run_command("cd docs/; make html SPHINXOPTS='-W;'", continue_on_fail=continue_on_fail)

    if success:
        CONSOLE.line()
        CONSOLE.rule(characters="=")
        CONSOLE.print("[bold green]:TADA: :TADA: :TADA: ALL CHECKS PASSED :TADA: :TADA: :TADA:", justify="center")
        CONSOLE.rule(characters="=")
    else:
        CONSOLE.line()
        CONSOLE.rule(characters="=", style=Style(color="red"))
        CONSOLE.print("[bold red]:skull: :skull: :skull: ERRORS FOUND :skull: :skull: :skull:", justify="center")
        CONSOLE.rule(characters="=", style=Style(color="red"))


def run_code_checks(continue_on_fail: bool = False):
    """Run a github actions file locally.

    Args:
        continue_on_fail: Whether or not to continue running actions commands if the current one fails
    """
    # core code checks
    run_github_actions_file(filename=".github/workflows/core_code_checks.yml", continue_on_fail=continue_on_fail)
    # viewer build and deployment
    # run_github_actions_file(filename=".github/workflows/viewer_build_deploy.yml", continue_on_fail=continue_on_fail)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(run_code_checks)


if __name__ == "__main__":
    entrypoint()
