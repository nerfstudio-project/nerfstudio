"""
run_eval.py
"""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict

import mediapy as media
import torch
from tqdm import tqdm

from nerfactory.cameras.camera_paths import (
    CameraPath,
    get_interpolated_camera_path,
    get_spiral_path,
)
from nerfactory.configs import base as cfg
from nerfactory.configs.utils import cli_from_base_configs
from nerfactory.pipelines.base import Pipeline
from nerfactory.utils.misc import human_format
from nerfactory.utils.writer import TimeWriter

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)


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


def _load_checkpoint(config: cfg.TrainerConfig, pipeline: Pipeline) -> Path:
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
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
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
    output_filename: Path,
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


def main(config: cfg.Config):
    """Evaluate trained model. This evaluation can either render a trajectory or compute the eval psnr.

    Args:
        config_name: Name of the config file to use.
        checkpoint_dir: Directory to load the checkpoint from.
        rendered_output_name: Name of the renderer output to use (rgb, depth, etc.).
        method: Method to use for evaluation. PSNR computes metrics, TRAJ renders a trajectory.
        traj: Trajectory to render.
        output_filename: Name of the output file.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
            Defaults to 1.0.
        config_overrides: List of strings to override config values.
    """
    assert config.eval is not None, "Eval config needs to be set"
    assert config.eval.checkpoint_dir is not None, "Missing eval checkpoint dir: --eval.checkpoint-dir"

    config.trainer.load_dir = config.eval.checkpoint_dir
    config.pipeline.dataloader.eval_image_indices = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setup pipeline (which includes the dataloaders)
    pipeline = config.pipeline.setup(device=device, test_mode=True)
    pipeline.eval()
    # load checkpointed information
    checkpoint_name = _load_checkpoint(config.trainer, pipeline)

    if config.eval.method == cfg.MethodType.PSNR:
        assert config.eval.output_filename.suffix == ".json"
        stats_dict = render_stats_dict(pipeline)
        avg_psnr = stats_dict["avg psnr"]
        avg_rays_per_sec = stats_dict["avg rays per sec"]
        avg_fps = stats_dict["avg fps"]
        print(f"Avg. PSNR: {avg_psnr:0.4f}")
        print(f"Avg. Rays / Sec: {human_format(avg_rays_per_sec)}")
        print(f"Avg. FPS: {avg_fps:0.4f}")
        # save output to some file
        config.eval.output_filename.parent.mkdir(parents=True, exist_ok=True)
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_name),
            "results": stats_dict,
        }
        with open(config.eval.output_filename, "w", encoding="utf8") as output_f:
            json.dump(benchmark_info, output_f, indent=2)
        print(f"Saved results to: {config.eval.output_filename}")

    elif config.eval.method == cfg.MethodType.TRAJ:
        assert config.eval.output_filename.suffix == ".mp4"
        assert config.eval.rendered_output_name is not None, "Missing eval output type: --eval.rendered-output-name"

        # TODO(ethan): use camera information from parsing args
        if config.eval.traj == cfg.TrajectoryType.SPIRAL:
            camera_start = pipeline.dataloader.eval_dataloader.get_camera(image_idx=0)
            # TODO(ethan): pass in the up direction of the camera
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif config.eval.traj == cfg.TrajectoryType.INTERP:
            camera_start = pipeline.dataloader.eval_dataloader.get_camera(image_idx=0)
            camera_end = pipeline.dataloader.eval_dataloader.get_camera(image_idx=10)
            camera_path = get_interpolated_camera_path(camera_start, camera_end, steps=30)
        else:
            raise NotImplementedError

        render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=config.eval.output_filename,
            rendered_output_name=config.eval.rendered_output_name,
            rendered_resolution_scaling_factor=config.eval.rendered_resolution_scaling_factor,
            num_rays_per_chunk=pipeline.dataloader.eval_num_rays_per_chunk,
        )


if __name__ == "__main__":
    from nerfactory.configs.base_configs import base_configs

    instantiated_config = cli_from_base_configs(base_configs, eval_mode=True)
    if instantiated_config.eval.load_config:
        logging.info(f"Loading pre-set config to: {instantiated_config.eval.load_config}")
        with open(instantiated_config.eval.load_config, "rb") as config_f:
            train_instantiated_config = pickle.load(config_f)
        train_instantiated_config.eval = instantiated_config.eval
        instantiated_config = train_instantiated_config
    logging.info("Printing current config setup")
    print(instantiated_config)
    main(instantiated_config)
