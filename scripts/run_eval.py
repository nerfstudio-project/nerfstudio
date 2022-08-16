"""
run_eval.py
"""
import enum
import json
import os
from typing import Dict, List, Optional

import dcargs
import mediapy as media
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from tqdm import tqdm

from nerfactory.cameras.camera_paths import (
    CameraPath,
    get_interpolated_camera_path,
    get_spiral_path,
)
from nerfactory.pipelines.base import Pipeline, setup_pipeline
from nerfactory.utils import io as io_utils
from nerfactory.utils.config import setup_config
from nerfactory.utils.misc import human_format
from nerfactory.utils.writer import TimeWriter


def _update_avg(prev_avg: float, new_val: float, step: int) -> float:
    """helper to calculate the running average

    Args:
        prev_avg (float): previous average value
        new_val (float): new value to update the average with
        step (int): current step number

    Returns:
        float: new updated average
    """
    return (step * prev_avg + new_val) / (step + 1)


def _load_checkpoint(config: DictConfig, pipeline: Pipeline) -> str:
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    """
    assert config.load_dir is not None
    if config.load_step is None:
        print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = os.path.join(config.load_dir, f"step-{load_step:09d}.ckpt")
    assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"])
    print(f"done loading checkpoint from {load_path}")
    return load_path


def render_stats_dict(pipeline: Pipeline) -> Dict[str, float]:
    """Helper function to evaluate the pipeline on a dataloader.

    Args:
        pipeline (Pipeline): Pipeline to evaluate

    Returns:
        dict: returns the average psnr and average rays per second
    """
    avg_psnr = 0
    avg_rays_per_sec = 0
    avg_fps = 0
    for step, (camera_ray_bundle, batch) in tqdm(enumerate(pipeline.dataloader.eval_dataloader)):
        with TimeWriter(writer=None, name=None, write=False) as t:
            with torch.no_grad():
                image_idx = int(camera_ray_bundle.camera_indices[0, 0])
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                psnr = pipeline.model.log_test_image_outputs(image_idx, step, batch, outputs)
        avg_rays_per_sec = _update_avg(avg_rays_per_sec, camera_ray_bundle.origins.shape[0] / t.duration, step)
        avg_psnr = _update_avg(avg_psnr, psnr, step)
        avg_fps = _update_avg(avg_fps, 1 / t.duration, step)
    return {"avg psnr": avg_psnr, "avg rays per sec": avg_rays_per_sec, "avg fps": avg_fps}


def render_trajectory_video(
    pipeline: Pipeline,
    camera_path: CameraPath,
    output_filename: str,
    rendered_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    num_rays_per_chunk: int = 4096,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        camera_path: Index of the image to render.
        output_filename: Name of the output file.
        rendered_output_name: Name of the renderer output to use.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
            Defaults to 1.0.
        num_rays_per_chunk: Number of rays to use per chunk. Defaults to 4096.
    """
    print("Creating trajectory video.")
    images = []
    for camera in tqdm(camera_path.cameras):
        camera.rescale_output_resolution(rendered_resolution_scaling_factor)
        camera_ray_bundle = camera.get_camera_ray_bundle().to(pipeline.device)
        camera_ray_bundle.num_rays_per_chunk = num_rays_per_chunk
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        image = outputs[rendered_output_name].cpu().numpy()
        images.append(image)

    seconds = 5.0
    fps = len(images) / seconds
    media.write_video(output_filename, images, fps=fps)


class MethodType(enum.Enum):
    """Enum for the method type."""

    PSNR = enum.auto()
    TRAJ = enum.auto()


class TrajectoryType(enum.Enum):
    """Enum for the trajectory type."""

    SPIRAL = enum.auto()
    INTERP = enum.auto()


def main(
    config_name: str,
    checkpoint_dir: str,
    rendered_output_name: str,
    method: MethodType = MethodType.PSNR,
    traj: TrajectoryType = TrajectoryType.SPIRAL,
    output_filename: str = "output.json",
    rendered_resolution_scaling_factor: float = 1.0,
    config_overrides: Optional[List[str]] = None,
):
    """Evaluate trained model. This evaluation can either render a trajectory or compute the eval psnr.

    Args:
        config_name: Name of the config file to use.
        checkpoint_dir: Directory to load the checkpoint from.
        rendered_output_name: Name of the renderer output to use.
        method: Method to use for evaluation. PSNR computes metrics, TRAJ renders a trajectory.
        traj: Trajectory to render.
        output_filename: Name of the output file.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
            Defaults to 1.0.
        config_overrides: List of strings to override config values.
    """

    config_path = "../configs"
    initialize(version_base="1.2", config_path=config_path)
    config_overrides = config_overrides or []
    config = compose(config_name, overrides=config_overrides)

    config.trainer.resume_train.load_dir = checkpoint_dir
    config.pipeline.dataloader.eval_image_indices = None
    config = setup_config(config)  # converting to typed config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setup pipeline (which includes the dataloaders)
    pipeline = setup_pipeline(config.pipeline, device=device, test_mode=True)
    pipeline.eval()
    # load checkpointed information
    checkpoint_name = _load_checkpoint(config.trainer.resume_train, pipeline)

    if method == MethodType.PSNR:
        assert output_filename.endswith(".json")
        stats_dict = render_stats_dict(pipeline)
        avg_psnr = stats_dict["avg psnr"]
        avg_rays_per_sec = stats_dict["avg rays per sec"]
        avg_fps = stats_dict["avg fps"]
        print(f"Avg. PSNR: {avg_psnr:0.4f}")
        print(f"Avg. Rays / Sec: {human_format(avg_rays_per_sec)}")
        print(f"Avg. FPS: {avg_fps:0.4f}")
        # save output to some file
        io_utils.make_dir(output_filename)
        benchmark_info = {
            "config_name": config_name,
            "checkpoint": checkpoint_name,
            "results": stats_dict,
        }
        with open(output_filename, "w", encoding="utf8") as f:
            json.dump(benchmark_info, f, indent=2)
        print(f"Saved results to: {output_filename}")

    elif method == MethodType.TRAJ:
        assert output_filename.endswith(".mp4")
        # TODO(ethan): use camera information from parsing args
        if traj == TrajectoryType.SPIRAL:
            camera_start = pipeline.dataloader.eval_dataloader.get_camera(image_idx=0)
            # TODO(ethan): pass in the up direction of the camera
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif traj == TrajectoryType.INTERP:
            camera_start = pipeline.dataloader.eval_dataloader.get_camera(image_idx=0)
            camera_end = pipeline.dataloader.eval_dataloader.get_camera(image_idx=10)
            camera_path = get_interpolated_camera_path(camera_start, camera_end, steps=30)
        render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=output_filename,
            rendered_output_name=rendered_output_name,
            rendered_resolution_scaling_factor=rendered_resolution_scaling_factor,
            num_rays_per_chunk=pipeline.dataloader.eval_num_rays_per_chunk,
        )


if __name__ == "__main__":
    dcargs.cli(main)
