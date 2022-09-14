#!/usr/bin/env python
"""Configuration script for setting up tab completion for nerfactory in bash and zsh."""

import concurrent.futures
import itertools
import os
import pathlib
import random
import shutil
import stat
import subprocess
import sys
from typing import List, Literal

import dcargs
from rich.console import Console
from rich.prompt import Confirm
from typing_extensions import assert_never

ConfigureMode = Literal["install", "uninstall"]
ShellType = Literal["zsh", "bash"]

CONSOLE = Console(width=120)


def _check_dcargs_cli(script_path: pathlib.Path) -> bool:
    """Check if a path points to a script containing a dcargs.cli() call. Also checks
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
        # and the shebang check even if dcargs isn't used.
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

        # Return True only if compatible with dcargs.
        return "import dcargs" in script_src and "dcargs.cli" in script_src
    return False


def _generate_completion(script_path: pathlib.Path, shell: ShellType, completions_dir: pathlib.Path) -> pathlib.Path:
    """Given a path to a dcargs CLI, write a completion script to a target directory.

    Args:
        script_path: Path to Python CLI to generate completion script for.
        shell: Shell to generate completion script for.
        completions_dir: Directory to write completion script to.

    Returns:
        Success flag.
    """
    # Use zsh standard for naming completion scripts.
    target_path = completions_dir / shell / ("_" + script_path.name.replace(".", "_"))

    # Generate and write the new completion.
    new = subprocess.run(
        args=[sys.executable, str(script_path), "--dcargs-print-completion", shell],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf8",
        check=True,
    ).stdout

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
    source_lines = "\n".join(
        [
            "",
            "# Source nerfactory autocompletions.",
            f"source {completions_dir / 'setup'}.{shell}",
        ]
    )
    rc_path = pathlib.Path(os.environ["HOME"]) / f".{shell}rc"
    if mode == "install":
        if source_lines in rc_path.read_text():
            CONSOLE.log(f":call_me_hand: Completions are already installed in {rc_path}. {_exclamation()}")
            return

        if not Confirm.ask(f"[bold yellow]Install to {rc_path}?", default=True):
            CONSOLE.log(f"[bold red]Skipping install for {rc_path.name}.")
            return

        rc_path.write_text(rc_path.read_text() + source_lines)
        CONSOLE.log(
            f":person_gesturing_ok: Completions installed to {rc_path}. {_exclamation()} Open a new shell to try them"
            " out."
        )

    elif mode == "uninstall":
        if source_lines not in rc_path.read_text():
            CONSOLE.log(f":heavy_check_mark: No completions to uninstall from {rc_path.name}.")
            return

        if not Confirm.ask(f"[bold yellow]Uninstall from {rc_path}?", default=True):
            CONSOLE.log(f"[dim red]Skipping uninstall for {rc_path.name}.")
            return

        rc_path.write_text(rc_path.read_text().replace(source_lines, ""))
        CONSOLE.log(f":broom: Completions uninstalled from {rc_path}.")

    else:
        assert_never(mode)


def main(
    mode: ConfigureMode,
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
    shells_supported: List[ShellType] = ["zsh", "bash"]
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

    # Install mode: Generate completion for each dcargs script.
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
        # Find dcargs CLIs.
        script_paths = list(filter(_check_dcargs_cli, scripts_dir.glob("**/*.py")))
        script_names = tuple(p.name for p in script_paths)
        assert len(set(script_names)) == len(script_names)

        # Get existing completion files.
        existing_completions = set()
        for shell in shells_supported:
            target_dir = completions_dir / shell
            if target_dir.exists():
                existing_completions |= set(target_dir.glob("*"))

        # Run generation jobs.
        concurrent_executor = concurrent.futures.ThreadPoolExecutor()
        with CONSOLE.status("[bold]:writing_hand:  Generating completions...", spinner="bouncingBall"):
            completion_paths = list(
                concurrent_executor.map(
                    lambda path_and_shell: _generate_completion(path_and_shell[0], path_and_shell[1], completions_dir),
                    itertools.product(script_paths, shells_found),
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
    else:
        assert_never(mode)

    # Install or uninstall from bashrc/zshrc.
    for shell in shells_found:
        _update_rc(completions_dir, mode, shell)

    CONSOLE.print("[bold]All done![/bold]")
    if mode == "install":
        CONSOLE.print()
        CONSOLE.print("Notes:")
        CONSOLE.print(
            ":warning: In bash, completions will trigger via [dim white]./scripts/run_train.py <TAB>[/dim white], but"
            " not [dim white]python ./scripts/run_train.py <TAB>[/dim white]."
        )
        CONSOLE.print(
            ":warning: Completions are in an experimental state. If you run into any issues, please file an issue!"
        )


if __name__ == "__main__":
    dcargs.cli(main, description=__doc__)
