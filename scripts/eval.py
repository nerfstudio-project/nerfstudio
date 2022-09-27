#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, Union

import dcargs
import mediapy as media
import torch
import yaml
from rich.console import Console
from rich.progress import track
from typing_extensions import assert_never

from nerfactory.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfactory.cameras.cameras import Cameras
from nerfactory.configs import base as cfg
from nerfactory.pipelines.base import Pipeline

console = Console(width=120)

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)


def _load_checkpoint(config: cfg.TrainerConfig, pipeline: Pipeline) -> Path:
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    """
    assert config.load_dir is not None
    if config.load_step is None:
        console.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"])
    console.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_name: Name of the renderer output to use.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Number for the output video.
    """
    console.print("[bold green]Creating trajectory video")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    for camera_idx in track(range(cameras.size), description=":movie_camera: Rendering :movie_camera:"):
        camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx).to(pipeline.device)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        image = outputs[rendered_output_name].cpu().numpy()
        images.append(image)

    fps = len(images) / seconds
    with console.status("[yellow]Saving video", spinner="bouncingBall"):
        media.write_video(output_filename, images, fps=fps)
    console.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    console.print(f"[green]Saved video to {output_filename}", justify="center")


def _eval_setup(config_path: Path) -> Tuple[cfg.Config, Pipeline, Path]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.

    Returns:
        Loaded config, pipeline module, and corresponding checkpoint.
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.trainer.load_dir = config.get_checkpoint_dir()
    config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=True)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path = _load_checkpoint(config.trainer, pipeline)

    return config, pipeline, checkpoint_path


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save to a JSON."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = _eval_setup(self.load_config)

        assert self.output_path.suffix == ".json"
        metrics_dict = pipeline.get_average_eval_image_metrics()
        # save output to some file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        console.print(f"Saved results to: {self.output_path}")


@dataclasses.dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer output to use. rgb, depth, etc.
    rendered_output_name: str = "rgb"
    #  Trajectory to render.
    traj: Literal["spiral", "interp", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("output.mp4")
    # How long the video should be.
    seconds: float = 5.0

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = _eval_setup(self.load_config)

        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
            # TODO(ethan): pass in the up direction of the camera
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif self.traj == "interp":
            # cameras_a = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
            # cameras_b = pipeline.datamanager.eval_dataloader.get_camera(image_idx=10)
            # camera_path = get_interpolated_camera_path(cameras, steps=30)
            raise NotImplementedError("Interpolated camera path not implemented.")
        elif self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_name=self.rendered_output_name,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # A Union over dataclass types will create a subcommand for each type.
    #
    # TODO: it would make sense to split this script up into separate scripts.
    # - To reduce duplicate code, some "upstream" refactor could also simplify the
    #   shared `_load_checkpoint()` and `_eval_setup()` helpers. The high-level
    #   operations implemented by each seem fairly universal; ideally the checkpoint
    #   loading logic, for example, would be the same as what's used for loading a
    #   checkpoint when resuming a training run.
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(Union[ComputePSNR, RenderTrajectory]).main()


if __name__ == "__main__":
    entrypoint()
