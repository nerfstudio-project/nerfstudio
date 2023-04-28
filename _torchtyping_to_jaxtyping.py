"""Convert torchtyping annotations to jaxtyping annotations.

This script can be deleted once we're done migrating!
"""

import itertools
from pathlib import Path


def main():
    for p in itertools.chain(
        Path("nerfstudio/").glob("**/*.py"),
        Path("docs/").glob("**/*.py"),
        Path("scripts/").glob("**/*.py"),
    ):
        src = p.read_text()
        if "TensorType" not in src:
            print(f"Skipping {p}")
            continue

        src = src.replace(
            "from torchtyping import TensorType",
            "from jaxtyping import Shaped\nfrom torch import Tensor",
        )
        print(f"Processing {p}")

        out_parts = []

        for i, part in enumerate(src.split("TensorType")):
            if i == 0:
                out_parts.append(part)
            elif not part.startswith("["):
                out_parts.append("Tensor")
                out_parts.append(part)
            else:
                shape_str, _, rest = part[1:].partition("]")

                out_parts.append("Shaped[Tensor, ")

                out_shape_parts = []
                for dim in shape_str.split(","):
                    # Remove quotes + hyphens will be treated as subtractions in
                    # jaxtyping.
                    dim = dim.strip().replace('"', "").replace("-", "_")
                    if dim == "...":
                        out_shape_parts.append("*batch")
                        continue
                    elif dim.endswith(":..."):
                        out_shape_parts.append("*" + dim.replace(":...", ""))
                    else:
                        out_shape_parts.append(dim)

                out_parts.append('"' + " ".join(out_shape_parts) + '"]')
                out_parts.append(rest)

        p.write_text("".join(out_parts))


if __name__ == "__main__":
    main()
