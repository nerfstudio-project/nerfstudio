#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import traceback

import tyro
from rich.console import Console
from nerfstudio.scripts.my_utils import get_sequence_size_from_experiment, get_step_from_ckpt_path
from nerfstudio.utils.colormaps import SceneDiverged

from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120, no_color=True)


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Checkpoint path
    load_ckpt: Path
    # Name of the output file.
    output_path: Path = Path("output.json")

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = eval_setup(self.load_config, load_ckpt=self.load_ckpt)
        assert self.output_path.suffix == ".json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        step = get_step_from_ckpt_path(checkpoint_path)
        sequence_size = (
            get_sequence_size_from_experiment(config.experiment_name) if "_n_" in config.experiment_name else None
        )

        benchmark_info = {
            "experiment_name": config.experiment_name,
            "step": step,
            "sequence_size": sequence_size,
            "method_name": config.method_name,
            "checkpoint_path": str(checkpoint_path),
        }

        scene_diverged = False
        try:  # Get the output and define the names to save to
            metrics_dict = pipeline.get_average_eval_image_metrics()
            benchmark_info.update(metrics_dict)
        except SceneDiverged as e:
            traceback.print_exc()
            print(e)
            scene_diverged = True

        benchmark_info.update({"scene_diverged": scene_diverged})
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")
        del pipeline
        del config


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
