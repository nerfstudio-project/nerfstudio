"""Script for automatically configuring completion scripts for bash and zsh.

Generates and installs completions by default, or uninstalls them if the --uninstall
flag is passed in.
"""

import concurrent.futures
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from typing import Literal

import dcargs
from rich.console import Console
from rich.prompt import Confirm
from typing_extensions import assert_never

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
    target_path = target_dir / (script_path.name if shell == "bash" else "_" + script_path.name.replace(".", "_"))
    assert not target_path.exists()
    out = subprocess.run(
        args=[sys.executable, str(script_path), "--dcargs-print-completion", shell],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf8",
    )
    if out.returncode != 0:
        return False
    target_path.write_text(out.stdout)
    CONSOLE.log(f"[dim green]Wrote {shell} completions for {script_path.name} to {target_path}")
    return True


def _exclamation() -> str:
    return random.choice(["Cool", "Nice", "Neat", "Wow", "Great", "Exciting"]) + "!"


def _update_rc(
    completions_dir: pathlib.Path,
    shell: Literal["zsh", "bash"],
    mode: Literal["install", "uninstall"],
) -> None:
    """Try to add a `source ___/completions/setup.{shell}` line automatically to a user's zshrc or bashrc

    Args:
        completions_dir: Path to location of this script.
        shell: Shell to install completion scripts for.
    """

    # Try to locate the user's bashrc or zshrc.
    if "HOME" not in os.environ:
        CONSOLE.log(f"[bold red]No home directory found.")
        return
    rc_path = pathlib.Path(os.environ["HOME"]) / f".{shell}rc"
    if not rc_path.exists():
        CONSOLE.log(f"[bold yellow]{rc_path.name} not found, skipping.")
        return

    # Install or uninstall `source_line`.
    source_line = f"\nsource {completions_dir / 'setup'}.{shell}"

    if mode == "install":
        if source_line in rc_path.read_text():
            CONSOLE.log(f"[bold green]Completions already look installed in your {rc_path.name}. {_exclamation()}")
            return

        if not Confirm.ask(f"[bold yellow]Install to {rc_path}?", default=True):
            CONSOLE.log(f"[bold red]Skipping install for {rc_path.name}.")
            return

        rc_path.write_text(rc_path.read_text() + source_line)
        CONSOLE.log(f"[bold green]Completion installed to {rc_path}. {_exclamation()}")

    elif mode == "uninstall":
        if source_line not in rc_path.read_text():
            CONSOLE.log(f"[bold green]No completions to uninstall from {rc_path.name}.")
            return

        if not Confirm.ask(f"[bold yellow]Uninstall from {rc_path}?", default=True):
            CONSOLE.log(f"[bold red]Skipping uninstall for {rc_path.name}.")
            return

        rc_path.write_text(rc_path.read_text().replace(source_line, ""))
        CONSOLE.log(f"[bold green]Completion uninstalled from {rc_path}. {_exclamation()}")

    else:
        assert_never(mode)


def main(mode: Literal["install", "uninstall"], /):
    # Get scripts/ directory.
    completions_dir = pathlib.Path(__file__).absolute().parent
    scripts_dir = completions_dir.parent
    assert completions_dir.name == "completions"
    assert scripts_dir.name == "scripts"

    # Generate completion for each dcargs script.
    concurrent_executor = concurrent.futures.ThreadPoolExecutor()
    shell: Literal["zsh", "bash"]
    for shell in ("zsh", "bash"):

        # Get + reset target directory for each shell type.
        target_dir = completions_dir / shell
        if target_dir.exists():
            assert target_dir.is_dir()
            shutil.rmtree(target_dir, ignore_errors=True)
            CONSOLE.log(f"[bold]Deleted existing completion directory: {target_dir}.")

        # Generate completions if in install mode.
        if mode == "install":
            target_dir.mkdir()

            # Find dcargs CLIs.
            script_paths = list(filter(_is_dcargs_cli, scripts_dir.glob("**/*.py")))
            script_names = tuple(p.name for p in script_paths)
            assert len(set(script_names)) == len(script_names)

            # Generate + write completion scripts.
            with CONSOLE.status(f"[bold]Generating {shell} completions...", spinner="bouncingBall"):
                assert all(
                    concurrent_executor.map(
                        lambda script_path: _generate_completion(script_path, target_dir, shell),
                        script_paths,
                    )
                ), "One or more completion generations failed."
            CONSOLE.log(f"[bold]Finish generating {shell} completions!")

    # Install or uninstall from bashrc/zshrc.
    for shell in ("zsh", "bash"):
        _update_rc(completions_dir, shell, mode)


if __name__ == "__main__":
    dcargs.cli(main, description=__doc__)
