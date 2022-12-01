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
from typing import List, Union

import tyro
from rich.console import Console
from rich.prompt import Confirm
from typing_extensions import Literal, assert_never

ConfigureMode = Literal["install", "uninstall"]
ShellType = Literal["zsh", "bash"]

CONSOLE = Console(width=120)

ENTRYPOINTS = [
    "ns-install-cli",
    "ns-process-data",
    "ns-download-data",
    "ns-train",
    "ns-eval",
    "ns-render",
    "ns-dev-test",
]


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
    header_line = "# Source nerfstudio autocompletions."
    if shell == "zsh":
        source_lines = "\n".join(
            [
                "",
                header_line,
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
                "",
                header_line,
                f"source {completions_dir / 'setup.bash'}",
            ]
        )
    else:
        assert_never(shell)

    rc_path = pathlib.Path(os.environ["HOME"]) / f".{shell}rc"

    # Always try to uninstall previous completions.
    rc_source = rc_path.read_text()
    while header_line in rc_source:
        before_install, _, after_install = rc_source.partition(header_line)
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


def main(mode: ConfigureMode = "install") -> None:
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

        # Run generation jobs.
        concurrent_executor = concurrent.futures.ThreadPoolExecutor()
        with CONSOLE.status("[bold]:writing_hand:  Generating completions...", spinner="bouncingBall"):
            completion_paths = list(
                concurrent_executor.map(
                    lambda path_or_entrypoint_and_shell: _generate_completion(
                        path_or_entrypoint_and_shell[0], path_or_entrypoint_and_shell[1], completions_dir
                    ),
                    itertools.product(script_paths + ENTRYPOINTS, shells_found),
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


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main, description=__doc__)


if __name__ == "__main__":
    entrypoint()
