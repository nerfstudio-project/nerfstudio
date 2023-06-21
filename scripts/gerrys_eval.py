#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RaySamples

CONSOLE = Console(width=120)


@dataclass
class ComputeGerry:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.pickle")
    # Whether to run train metrics or not.
    run_train_metrics: bool = False


    def compute_eval_image(self, pipeline):
        print('getting eval image...')
        image_idx, camera_ray_bundle, batch = pipeline.datamanager.next_eval_image(9999999)
        print('computing eval image...')
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        print('returning eval image...')
        return image_idx, outputs['image'].cpu().numpy()


    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = eval_setup(self.load_config)

        idx, image = self.compute_eval_image(pipeline)
        import pickle
        with open(self.output_path, 'wb') as f:
            pickle.dump((idx, image), f)

        print(f"Saved results to: {self.output_path}")

@dataclass
class ComputeOccupancy:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.pickle")


    def compute_occupancy(self, pipeline):
        print("Entered compute_occupancy...")
        model = pipeline.model

        # positions = torch.meshgrid()  # TODO(benjamin)
        # positions = torch.zeros((10, 1, 3), device=model.device)
        # positions = torch.arange(-5, 5, 0.1, device=model.device).view(-1, 1, 1).repeat(1, 1, 3)
        # _, wavelengths = model.get_wavelengths([0, 1, 2]) # rgb uses only 0, 1, 2

        # bounds = -5, 5, 0.1
        # bounds = -2.75, 2.75, 0.1
        bounds = -0.875, 0.875, 0.01 # TODO: compute them automatically
        positions = torch.stack(torch.meshgrid(torch.arange(*bounds), torch.arange(*bounds), torch.arange(*bounds)), -1).to(model.device).reshape(-1, 1, 3)

        n_wavelengths = model.config.num_output_color_channels
        wavelengths = torch.arange(
                n_wavelengths,
                dtype=torch.float32,
                device=model.device,
                requires_grad=False,
            ) / n_wavelengths
        print(f"n_wavelengths = {n_wavelengths}")
        densities = model.field.density_fn(positions, wavelengths=wavelengths)
        import pickle

        with open('test_position.pkl', 'wb') as handle:
            pickle.dump(positions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('test_densities.pkl', 'wb') as handle:
            pickle.dump(densities, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(densities)
        return densities


    def main(self) -> None:
        """Main function."""
        print("Entered main...")
        # import time
        # time.sleep(2)
        config, pipeline, checkpoint_path = eval_setup(self.load_config)

        densities = self.compute_occupancy(pipeline)
        # import pickle
        # with open(self.output_path, 'wb') as f:
        #     pickle.dump((idx, image), f)

        # print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    # tyro.cli(ComputeGerry).main()
    tyro.cli(ComputeOccupancy).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputeGerry)  # noqa
