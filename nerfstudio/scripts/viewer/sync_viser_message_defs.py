"""Generate viser message definitions for TypeScript, by parsing Python dataclasses."""
import pathlib
import subprocess

import tyro
from viser.infra import generate_typescript_interfaces

from nerfstudio.viewer.viser import NerfstudioMessage


def main() -> None:
    """Generate viser message definitions for TypeScript, by parsing Python dataclasses."""

    # Generate typescript source.
    defs = generate_typescript_interfaces(NerfstudioMessage)

    # Write to file.
    # Three parents from nerfstudio/scripts/viewer/sync_viser_message_defs.py:
    # - nerfstudio/scripts/viewer/
    # - nerfstudio/scripts/
    # - nerfstudio/
    target_path = pathlib.Path(__file__).absolute().parent.parent.parent / pathlib.Path(
        "nerfstudio/viewer/app/src/modules/WebSocket/ViserMessages.tsx"
    )
    assert target_path.exists()
    target_path.write_text(defs, encoding="utf_8")
    print(f"Wrote to {target_path}")

    # Run prettier.
    # TODO: if this is not installed maybe we should print some error with installation
    # instructions?
    subprocess.run(args=["prettier", "-w", str(target_path)], check=False)


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    # All entrypoints must currently be tyro CLIs.
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
