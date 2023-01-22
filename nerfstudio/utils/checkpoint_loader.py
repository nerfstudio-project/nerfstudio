from pathlib import Path
import sys
from rich.console import Console
import os


CONSOLE = Console(width=120, no_color=True)
DEFAULT_MODEL_GLOB = "step*ckpt"
DEFAULT_MODEL_COMPARATOR = lambda x: os.path.getctime(x)


def _validate_checkpoint_path(ckpt_path: Path):
    if not ckpt_path.is_file():
        CONSOLE.rule("Error", style="red")
        CONSOLE.print(
            f"Please make sure the checkpoint exists {str(ckpt_path)} and is a file. Checkpoint should be generated periodically during training.",
            justify="center",
        )
        sys.exit(1)
    return


def find_checkpoint(load_ckpt: Path | None) -> None | Path:
    """Returns None if no checkpoints were found"""

    if load_ckpt is None:
        return None

    ckpt_path = Path(load_ckpt)
    _validate_checkpoint_path(ckpt_path)
    return ckpt_path


def _validate_load_dir(load_dir: Path):
    if not load_dir.is_dir():
        CONSOLE.rule("Error", style="red")
        CONSOLE.print(
            f"No checkpoint directory found at {str(load_dir)}, ",
            justify="center",
        )
        CONSOLE.print(
            "Please make sure the checkpoints exists, they should be generated periodically during training",
            justify="center",
        )
        sys.exit(1)
    return


def _list_checkpoints(load_dir: Path | str, glob=DEFAULT_MODEL_GLOB) -> list[Path]:
    load_dir = Path(load_dir)
    return list(load_dir.glob(glob))


def find_latest_checkpoint(
    load_dir: Path,
    glob=DEFAULT_MODEL_GLOB,
    comparator=DEFAULT_MODEL_COMPARATOR,
    reverse_comparator=True,
):
    """Returns None if no checkpoints were found"""

    assert load_dir is not None, f"{load_dir} is None"
    load_dir = Path(load_dir)

    _validate_load_dir(load_dir)
    CONSOLE.print(f"Loading latest checkpoint from {load_dir}")
    checkpoints = _list_checkpoints(load_dir, glob)
    if len(checkpoints) == 0:
        CONSOLE.print(
            f"Please make sure directory {str(load_dir)} contains checkpoints.",
            justify="center",
        )
        sys.exit(1)

    if reverse_comparator:
        load_path = max(checkpoints, key=comparator)
    else:
        load_path = min(checkpoints, key=comparator)

    assert load_path.exists(), f"Checkpoint {str(load_path)} does not exist"
    return load_path
