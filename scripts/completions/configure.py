"""Configuration script for setting up tab completion for nerfactory in bash and zsh.

TODO:
    - updated vs unchanged messages
    - delete random files
    - remove shell CLI options
"""
import concurrent.futures
import difflib
import itertools
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from typing import List, Literal, Tuple

import dcargs
from rich.console import Console
from rich.prompt import Confirm
from typing_extensions import assert_never

# Try to import nerfactory. We don't need it, but this helps us verify that we're in the
# right virtual environment.
import nerfactory

CONSOLE = Console(width=120)


def _is_dcargs_cli(script_path: pathlib.Path) -> bool:
    """Check if a path points to a script containing a dcargs.cli() call.

    Args:
        script_path: Path to prospective CLI.

    Returns:
        True if a completion is can be generated.
    """
    assert script_path.suffix == ".py"
    script_src = script_path.read_text()
    return "import dcargs" in script_src and "dcargs.cli" in script_src and "__main__" in script_src


def _generate_completion(script_path: pathlib.Path, target_dir: pathlib.Path, shell: Literal["zsh", "bash"]) -> bool:
    """Given a path to a dcargs CLI, write a completion script to a target directory.

    Args:
        script_path: Path to Python CLI to generate completion script for.
        target_dir: Directory to write completion script to.
        shell: Shell to generate completion script for.

    Returns:
        Success flag.
    """
    # Note: the naming here is flexible because we manually source all completions, but is designed to be consistent
    # with bash/zsh standards.
    target_path = target_dir / (script_path.name if shell == "bash" else "_" + script_path.name.replace(".", "_"))
    # assert not target_path.exists()

    # Generate and write the new completion.
    new = subprocess.run(
        args=[sys.executable, str(script_path), "--dcargs-print-completion", shell],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf8",
        check=True,
    ).stdout
    if not target_path.exists():
        target_path.write_text(new)
        CONSOLE.log(f"[dim]:heavy_check_mark: Wrote new completion: {target_path}[/dim]")
    elif (current := target_path.read_text().strip()) != new.strip():
        # print(''.join(difflib.ndiff(current, new)))
        target_path.write_text(new)
        CONSOLE.log(f"[dim]:heavy_check_mark: Updated completion: {target_path}[/dim]")
    else:
        CONSOLE.log(f"[dim yellow]:heavy_check_mark: Nothing to do: {target_path}[/dim yellow]")
    return True


def _exclamation() -> str:
    return random.choice(["Cool", "Nice", "Neat", "Great", "Exciting", "Excellent"]) + "!"


def _update_rc(
    completions_dir: pathlib.Path,
    mode: Literal["install", "uninstall"],
    shell: Literal["zsh", "bash"],
) -> None:
    """Try to add a `source /.../completions/setup.{shell}` line automatically to a user's zshrc or bashrc.

    Args:
        completions_dir: Path to location of this script.
        shell: Shell to install completion scripts for.
        mode: Install or uninstall completions.
    """

    # Install or uninstall `source_line`.
    source_line = f"\nsource {completions_dir / 'setup'}.{shell}"
    rc_path = pathlib.Path(os.environ["HOME"]) / f".{shell}rc"
    if mode == "install":
        if source_line in rc_path.read_text():
            CONSOLE.log(f":call_me_hand: Completions are already installed in {rc_path}. {_exclamation()}")
            return

        if not Confirm.ask(f"[bold yellow]Install to {rc_path}?", default=True):
            CONSOLE.log(f"[bold red]Skipping install for {rc_path.name}.")
            return

        rc_path.write_text(rc_path.read_text() + source_line)
        CONSOLE.log(f":stars: Completions installed to {rc_path}. {_exclamation()} Open a new shell to try them out.")

    elif mode == "uninstall":
        if source_line not in rc_path.read_text():
            CONSOLE.log(f":heavy_check_mark: No completions to uninstall from {rc_path.name}.")
            return

        if not Confirm.ask(f"[bold yellow]Uninstall from {rc_path}?", default=True):
            CONSOLE.log(f"[dim red]Skipping uninstall for {rc_path.name}.")
            return

        rc_path.write_text(rc_path.read_text().replace(source_line, ""))
        CONSOLE.log(f":broom: Completions uninstalled from {rc_path}.")

    else:
        assert_never(mode)


def main(
    mode: Literal["install", "uninstall"],
    shells: List[Literal["zsh", "bash"]] = ["zsh", "bash"],
    /,
) -> None:
    """Main script.

    Args:
        mode: Choose between installing or uninstalling completions.
        shells: Shell(s) to install or uninstall.
    """

    if "HOME" not in os.environ:
        CONSOLE.log("[bold red]$HOME is not set. Exiting.")
        return

    # Try to locate the user's bashrc or zshrc.
    shells_found = []
    for shell in shells:
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

    # Install mode: Generate completion for each dcargs script.
    if mode == "uninstall":
        for shell in shells:
            # Reset target directory for each shell type.
            target_dir = completions_dir / shell
            if target_dir.exists():
                assert target_dir.is_dir()
                shutil.rmtree(target_dir, ignore_errors=True)
                CONSOLE.log(f":broom: Deleted existing completion directory: {target_dir}.")
    elif mode == "install":
        concurrent_executor = concurrent.futures.ThreadPoolExecutor()
        results = []
        for shell in shells_found:
            # Get + reset target directory for each shell type.
            target_dir = completions_dir / shell
            # if target_dir.exists():
            #     assert target_dir.is_dir()
            #     shutil.rmtree(target_dir, ignore_errors=True)
            #     CONSOLE.log(f":broom: Deleted existing completion directory: {target_dir}.")
            # target_dir.mkdir()

            # Find dcargs CLIs.
            script_paths = list(filter(_is_dcargs_cli, scripts_dir.glob("**/*.py")))
            script_names = tuple(p.name for p in script_paths)
            assert len(set(script_names)) == len(script_names)

            # Generate + write completion scripts.
            results.append(
                concurrent_executor.map(
                    lambda script_path: _generate_completion(script_path, target_dir, shell),
                    script_paths,
                )
            )
        # Wait for all generation jobs to finish!
        with CONSOLE.status(f"[bold]:writing_hand:  Generating completions...", spinner="bouncingBall"):
            assert all(itertools.chain(*results))
    else:
        assert_never(mode)

    # Install or uninstall from bashrc/zshrc.
    for shell in shells_found:
        _update_rc(completions_dir, mode, shell)


if __name__ == "__main__":
    dcargs.cli(main, description=__doc__)
