"""Detect all dcargs CLIs, and generate completion scripts (bash, zsh) for them."""

import concurrent.futures
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Literal

from rich.console import Console

CONSOLE = Console(width=120)


def _is_dcargs_cli(script_path: pathlib.Path) -> bool:
    """Check if a path points to a script containing a dcargs.cli() call.

    Args:
        script_path: Path to prospective CLI.

    Returns:
        True if a completion is can be generated.
    """
    assert script_path.suffix == ".py"
    if script_path.absolute() == pathlib.Path(__file__).absolute():
        return False
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
    CONSOLE.log(f"[green]Wrote {shell} completions for {script_path.name} to {target_path}")
    return True


def main():
    # Get scripts/ directory.
    completions_dir = pathlib.Path(__file__).absolute().parent
    scripts_dir = completions_dir.parent
    assert completions_dir.name == "completions"
    assert scripts_dir.name == "scripts"

    concurrent_executor = concurrent.futures.ThreadPoolExecutor()

    # Generate completion for each dcargs script.
    shell: Literal["zsh", "bash"]
    for shell in ("zsh", "bash"):
        CONSOLE.print(f"[bold]Generating {shell} completions!")

        # Get + reset target directory for each shell type.
        target_dir = completions_dir / shell
        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir()

        # Find all dcargs CLIs.
        script_paths = list(filter(_is_dcargs_cli, scripts_dir.glob("**/*.py")))
        script_names = tuple(p.name for p in script_paths)
        assert len(set(script_names)) == len(script_names)

        # Generate + write completion scripts!
        assert all(
            concurrent_executor.map(
                lambda script_path: _generate_completion(script_path, target_dir, shell),
                script_paths,
            )
        ), "One or more completion generations failed."


if __name__ == "__main__":
    main()
