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

"""Generate viser message definitions for TypeScript, by parsing Python dataclasses."""
import json
import pathlib
from datetime import datetime

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
        "viewer/app/src/modules/WebSocket/ViserMessages.tsx"
    )
    assert target_path.exists()

    old_defs = target_path.read_text(encoding="utf_8")

    if old_defs != defs:
        target_path.write_text(defs, encoding="utf_8")

        with open("nerfstudio/viewer/app/package.json", "r", encoding="utf_8") as f:
            data = json.load(f)

        now = datetime.now()
        data["version"] = now.strftime("%y-%m-%d") + "-0"

        with open("nerfstudio/viewer/app/package.json", "w", encoding="utf_8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote updates to {target_path}")
        print(f"Current viewer version is now {data['version']}")
    else:
        print("No update to messages.")


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    # All entrypoints must currently be tyro CLIs.
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
