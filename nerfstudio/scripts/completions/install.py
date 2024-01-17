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
"""Configuration script for setting up tab completion for nerfstudio in bash and zsh."""

import concurrent.futures
import itertools
import os
import pathlib
import random
import shutil
import stat
import subprocess
import sys
from typing import List, Literal, Optional, Union, get_args as typing_get_args

import tyro
from rich.prompt import Confirm
from typing_extensions import assert_never

from nerfstudio.utils.rich_utils import CONSOLE

if sys.version_info < (3, 10):
    import importlib_metadata
else:
    from importlib import metadata as importlib_metadata

ConfigureMode = Literal["install", "uninstall"]
ShellType = Literal["zsh", "bash"]

HEADER_LINE = "# Source nerfstudio autocompletions."


def _get_all_entry_points() -> List[str]:
    # TODO: we should filter out entrypoints that are not tyro CLIs.
    entry_points = importlib_metadata.distribution("nerfstudio").entry_points
    return [x.name for x in entry_points]


def _check_tyro_cli(script_path: pathlib.Path) -> bool:
    """Check if a path points to a script containing a tyro.cli() call. Also checks
    for any permissions/shebang issues.

    Args:
        script_path: Path to prospective CLI.

    Returns:
        True if a completion is can be generated.
    """
    assert script_path.suffix == ".py"
    script_src = script_path.read_text()

    if '\nif __name__ == "__main__":\n' in script_src:
        # Check script for execute permissions. For consistency, note that we apply this
        # and the shebang check even if tyro isn't used.
        execute_flags = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        if not script_path.stat().st_mode & execute_flags and Confirm.ask(
            f"[yellow]:warning: {script_path} is not marked as executable. Fix?[/yellow]",
            default=True,
        ):
            script_path.chmod(script_path.stat().st_mode | execute_flags)

        # Check that script has a shebang.
        if not script_src.startswith("#!/") and Confirm.ask(
            f"[yellow]:warning: {script_path} is missing a shebang. Fix?[/yellow]",
            default=True,
        ):
            script_path.write_text("#!/usr/bin/env python\n" + script_src)

        # Return True only if compatible with tyro.
        return "import tyro" in script_src and "tyro.cli" in script_src
    return False


def _generate_completion(
    path_or_entrypoint: Union[pathlib.Path, str], shell: ShellType, completions_dir: pathlib.Path
) -> pathlib.Path:
    """Given a path to a tyro CLI, write a completion script to a target directory.

    Args:
        script_path: Path to Python CLI to generate completion script for.
        shell: Shell to generate completion script for.
        completions_dir: Directory to write completion script to.

    Returns:
        Success flag.
    """
    if isinstance(path_or_entrypoint, pathlib.Path):
        # Scripts.
        target_name = "_" + path_or_entrypoint.name.replace(".", "_")
        args = [sys.executable, str(path_or_entrypoint), "--tyro-print-completion", shell]
    elif isinstance(path_or_entrypoint, str):
        # Entry points.
        target_name = "_" + path_or_entrypoint
        args = [path_or_entrypoint, "--tyro-print-completion", shell]
    else:
        assert_never(path_or_entrypoint)

    target_path = completions_dir / shell / target_name

    # Generate and write the new completion.
    try:
        new = subprocess.run(
            args=args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf8",
            check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        CONSOLE.log(f":x: Completion script generation failed: {args}")
        if e.stdout is not None and len(e.stdout) > 0:
            CONSOLE.log(e.stdout)
        if e.stderr is not None and len(e.stderr) > 0:
            CONSOLE.log(e.stderr)
        raise e

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        target_path.write_text(new)
        CONSOLE.log(f":heavy_check_mark: Wrote new completion to {target_path}!")
    elif target_path.read_text().strip() != new.strip():
        target_path.write_text(new)
        CONSOLE.log(f":heavy_check_mark: Updated completion at {target_path}!")
    else:
        CONSOLE.log(f"[dim]:heavy_check_mark: Nothing to do for {target_path}[/dim].")
    return target_path


def _exclamation() -> str:
    return random.choice(["Cool", "Nice", "Neat", "Great", "Exciting", "Excellent", "Ok"]) + "!"


def _get_deactivate_script(commands: List[str], shell: Optional[ShellType], add_header=True) -> str:
    if shell is None:
        # Install the universal script
        result_script = []
        for shell_type in typing_get_args(ShellType):
            result_script.append(f'if [ -n "${shell_type.upper()}_VERSION" ]; then')
            result_script.append(_get_deactivate_script(commands, shell_type, add_header=False))
            result_script.append("fi")
        source_lines = "\n".join(result_script)

    elif shell == "zsh":
        source_lines = "\n".join([f"unset '_comps[{command}]' &> /dev/null" for command in commands])
    elif shell == "bash":
        source_lines = "\n".join([f"complete -r {command} &> /dev/null" for command in commands])
    else:
        assert_never(shell)

    if add_header:
        source_lines = f"\n{HEADER_LINE}\n{source_lines}"
    return source_lines


def _get_source_script(completions_dir: pathlib.Path, shell: Optional[ShellType], add_header=True) -> str:
    if shell is None:
        # Install the universal script
        result_script = []
        for shell_type in typing_get_args(ShellType):
            result_script.append(f'if [ -n "${shell_type.upper()}_VERSION" ]; then')
            result_script.append(_get_source_script(completions_dir, shell_type, add_header=False))
            result_script.append("fi")
        source_lines = "\n".join(result_script)

    elif shell == "zsh":
        source_lines = "\n".join(
            [
                "if ! command -v compdef &> /dev/null; then",
                "    autoload -Uz compinit",
                "    compinit",
                "fi",
                f"source {completions_dir / 'setup.zsh'}",
            ]
        )
    elif shell == "bash":
        source_lines = "\n".join(
            [
                f"source {completions_dir / 'setup.bash'}",
            ]
        )
    else:
        assert_never(shell)

    if add_header:
        source_lines = f"\n{HEADER_LINE}\n{source_lines}"
    return source_lines


def _update_rc(
    completions_dir: pathlib.Path,
    mode: ConfigureMode,
    shell: ShellType,
) -> None:
    """Try to add a `source /.../completions/setup.{shell}` line automatically to a user's zshrc or bashrc.

    Args:
        completions_dir: Path to location of this script.
        shell: Shell to install completion scripts for.
        mode: Install or uninstall completions.
    """

    # Install or uninstall `source_line`.
    source_lines = _get_source_script(completions_dir, shell)
    rc_path = pathlib.Path(os.environ["HOME"]) / f".{shell}rc"

    # Always try to uninstall previous completions.
    rc_source = rc_path.read_text()
    while HEADER_LINE in rc_source:
        before_install, _, after_install = rc_source.partition(HEADER_LINE)
        source_file, _, after_install = after_install.partition("\nsource ")[2].partition("\n")
        assert source_file.endswith(f"/completions/setup.{shell}")
        rc_source = before_install + after_install
        rc_path.write_text(rc_source)
        CONSOLE.log(f":broom: Existing completions uninstalled from {rc_path}.")

    # Install completions.
    if mode == "install":
        assert source_lines not in rc_source
        rc_path.write_text(rc_source.rstrip() + "\n" + source_lines)
        CONSOLE.log(
            f":person_gesturing_ok: Completions installed to {rc_path}. {_exclamation()} Open a new shell to try them"
            " out."
        )
    else:
        assert mode == "uninstall"


def _update_conda_scripts(
    commands: List[str],
    completions_dir: pathlib.Path,
    mode: ConfigureMode,
) -> None:
    """Try to add a `source /.../completions/setup.{shell}` line automatically to conda's activation scripts.

    Args:
        completions_dir: Path to location of this script.
        mode: Install or uninstall completions.
    """

    # Install or uninstall `source_line`.
    activate_source_lines = _get_source_script(completions_dir, None)
    deactivate_source_lines = _get_deactivate_script(commands, None)

    conda_path = pathlib.Path(os.environ["CONDA_PREFIX"])
    activate_path = conda_path / "etc/conda/activate.d/nerfstudio_activate.sh"
    deactivate_path = conda_path / "etc/conda/deactivate.d/nerfstudio_deactivate.sh"
    if mode == "uninstall":
        if activate_path.exists():
            os.remove(activate_path)
        if deactivate_path.exists():
            os.remove(deactivate_path)
        CONSOLE.log(f":broom: Existing completions uninstalled from {conda_path}.")
    elif mode == "install":
        # Install completions.
        activate_path.parent.mkdir(exist_ok=True, parents=True)
        deactivate_path.parent.mkdir(exist_ok=True, parents=True)
        with activate_path.open("w+", encoding="utf8") as f:
            f.write(activate_source_lines)
        with deactivate_path.open("w+", encoding="utf8") as f:
            f.write(deactivate_source_lines)
        CONSOLE.log(
            f":person_gesturing_ok: Completions installed to {conda_path}. {_exclamation()} Reactivate the environment"
            " to try them out."
        )
    else:
        assert_never(mode)


def _get_conda_path() -> Optional[pathlib.Path]:
    """
    Returns the path to the conda environment if
    the nerfstudio package is installed in one.
    """
    conda_path = None
    if "CONDA_PREFIX" in os.environ:
        # Conda is active, we will check if the Nerfstudio is installed in the conda env.
        distribution = importlib_metadata.distribution("nerfstudio")
        if str(distribution.locate_file("nerfstudio")).startswith(os.environ["CONDA_PREFIX"]):
            conda_path = pathlib.Path(os.environ["CONDA_PREFIX"])
    return conda_path


def _generate_completions_files(
    completions_dir: pathlib.Path,
    scripts_dir: pathlib.Path,
    shells_supported: List[ShellType],
    shells_found: List[ShellType],
) -> None:
    # Set to True to install completions for scripts as well.
    include_scripts = False

    # Find tyro CLIs.
    script_paths = list(filter(_check_tyro_cli, scripts_dir.glob("**/*.py"))) if include_scripts else []
    script_names = tuple(p.name for p in script_paths)
    assert len(set(script_names)) == len(script_names)

    # Get existing completion files.
    existing_completions = set()
    for shell in shells_supported:
        target_dir = completions_dir / shell
        if target_dir.exists():
            existing_completions |= set(target_dir.glob("*"))

    # Get all entry_points.
    entry_points = _get_all_entry_points()

    # Run generation jobs.
    concurrent_executor = concurrent.futures.ThreadPoolExecutor()
    with CONSOLE.status("[bold]:writing_hand:  Generating completions...", spinner="bouncingBall"):
        completion_paths = list(
            concurrent_executor.map(
                lambda path_or_entrypoint_and_shell: _generate_completion(
                    path_or_entrypoint_and_shell[0], path_or_entrypoint_and_shell[1], completions_dir
                ),
                itertools.product(script_paths + entry_points, shells_found),
            )
        )

    # Delete obsolete completion files.
    for unexpected_path in set(p.absolute() for p in existing_completions) - set(
        p.absolute() for p in completion_paths
    ):
        if unexpected_path.is_dir():
            shutil.rmtree(unexpected_path)
        elif unexpected_path.exists():
            unexpected_path.unlink()
        CONSOLE.log(f":broom: Deleted {unexpected_path}.")


def main(mode: ConfigureMode = "install") -> None:
    """Main script.

    Args:
        mode: Choose between installing or uninstalling completions.
        shells: Shell(s) to install or uninstall.
    """

    if "HOME" not in os.environ:
        CONSOLE.log("[bold red]$HOME is not set. Exiting.")
        return

    # Get conda path if in conda environment.
    conda_path = _get_conda_path()

    # Try to locate the user's bashrc or zshrc.
    shells_supported: List[ShellType] = list(typing_get_args(ShellType))
    if conda_path is not None:
        # Running in conda; we have to support all shells.
        shells_found = shells_supported
        CONSOLE.log(f":mag: Detected conda environment {conda_path}!")
    else:
        shells_found: List[ShellType] = []
        for shell in shells_supported:
            rc_path = pathlib.Path(os.environ["HOME"]) / f".{shell}rc"
            if not rc_path.exists():
                CONSOLE.log(f":person_shrugging: {rc_path.name} not found, skipping.")
            else:
                CONSOLE.log(f":mag: Found {rc_path.name}!")
                shells_found.append(shell)

    # Get scripts/ directory.
    completions_dir = pathlib.Path(__file__).absolute().parent
    scripts_dir = completions_dir.parent
    assert completions_dir.name == "completions"
    assert scripts_dir.name == "scripts"

    # Install mode: Generate completion for each tyro script.
    if mode == "uninstall":
        for shell in shells_supported:
            # Reset target directory for each shell type.
            target_dir = completions_dir / shell
            if target_dir.exists():
                assert target_dir.is_dir()
                shutil.rmtree(target_dir, ignore_errors=True)
                CONSOLE.log(f":broom: Deleted existing completion directory: {target_dir}.")
            else:
                CONSOLE.log(f":heavy_check_mark: No existing completions at: {target_dir}.")
    elif mode == "install":
        _generate_completions_files(completions_dir, scripts_dir, shells_supported, shells_found)
    else:
        assert_never(mode)

    if conda_path is not None:
        # In conda environment we add the completions activation scripts.
        commands = _get_all_entry_points()
        _update_conda_scripts(commands, completions_dir, mode)
    else:
        # Install or uninstall from bashrc/zshrc.
        for shell in shells_found:
            _update_rc(completions_dir, mode, shell)

    CONSOLE.print("[bold]All done![/bold]")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main, description=__doc__)


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(main)  # noqa
