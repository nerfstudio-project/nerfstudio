"""
run_eval.py
"""
import enum
import os
from typing import Dict, List, Optional

import dcargs
import mediapy as media
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from tqdm import tqdm

from pyrad.cameras.camera_paths import CameraPath, get_interpolated_camera_path, get_spiral_path
from pyrad.data.dataloader import EvalDataloader, setup_dataset_eval, setup_dataset_train
from pyrad.graphs.base import Graph, setup_graph
from pyrad.utils.writer import TimeWriter


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


def _load_checkpoint(config: DictConfig, graph: Graph) -> None:
    """Helper function to load checkpointed graph

    Args:
        config (DictConfig): Configuration of graph to load
        graph (Graph): Graph instance of which to load weights
    """
    assert config.load_dir is not None
    if config.load_step is None:
        print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        load_step = sorted([int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir)])[-1]
    else:
        load_step = config.load_step
    load_path = os.path.join(config.load_dir, f"step-{load_step:09d}.ckpt")
    assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    graph.load_graph(loaded_state)
    print(f"done loading checkpoint from {load_path}")


def run_inference_from_config(config: DictConfig) -> Dict[str, float]:
    """helper function to run inference given config specifications (also used in benchmarking)

    Args:
        config (DictConfig): Configuration for loading the evaluation.

    Returns:
        Tuple[float, float]: returns both the avg psnr and avg rays per second
    """
    print("Running inference.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setup graph and dataset
    dataset_inputs_train, _ = setup_dataset_train(config.data, device=device)
    _, dataloader_eval = setup_dataset_eval(config.data, test_mode=True, device=device)
    graph = setup_graph(config.graph, dataset_inputs_train, device=device)
    graph.eval()

    # load checkpointed information
    _load_checkpoint(config.trainer.resume_train, graph)

    # calculate average psnr across test dataset
    return render_stats_dict(graph, dataloader_eval)


def render_stats_dict(graph: Graph, dataloader_eval: EvalDataloader) -> Dict[str, float]:
    """Helper function to evaluate the graph on a dataloader.

    Args:
        graph (Graph): Graph to evaluate
        dataloader_eval (EvalDataloader): Dataloader to evaluate on

    Returns:
        dict: returns the average psnr and average rays per second
    """
    avg_psnr = 0
    avg_rays_per_sec = 0
    avg_fps = 0
    for step, (camera_ray_bundle, batch) in tqdm(enumerate(dataloader_eval)):
        with TimeWriter(writer=None, name=None, write=False) as t:
            with torch.no_grad():
                image_idx = int(camera_ray_bundle.camera_indices[0, 0])
                outputs = graph.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                psnr = graph.log_test_image_outputs(image_idx, step, batch, outputs)
        avg_rays_per_sec = _update_avg(avg_rays_per_sec, camera_ray_bundle.origins.shape[0] / t.duration, step)
        avg_psnr = _update_avg(avg_psnr, psnr, step)
        avg_fps = _update_avg(avg_fps, 1 / t.duration, step)
    return {"avg psnr": avg_psnr, "avg rays per sec": avg_rays_per_sec, "avg fps": avg_fps}


def render_trajectory_video(
    graph: Graph,
    camera_path: CameraPath,
    output_filename: Optional[str] = None,
    rendered_output_name: Optional[str] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    num_rays_per_chunk: int = 4096,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        graph: Graph to evaluate with.
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
        camera_ray_bundle = camera.get_camera_ray_bundle().to(graph.device)
        camera_ray_bundle.num_rays_per_chunk = num_rays_per_chunk
        with torch.no_grad():
            outputs = graph.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        # TODO: don't hardcode the key! this will break for some nerf Graphs
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
    output_filename: str = "output.mp4",
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
    # parser = my_func_that_returns_a_parser()
    # args = parser.parse_args()

    config_path = "../configs"
    initialize(version_base="1.2", config_path=config_path)
    config_overrides = config_overrides or []
    config = compose(config_name, overrides=config_overrides)

    config.trainer.resume_train.load_dir = checkpoint_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setup graph and dataset
    dataset_inputs_train, _ = setup_dataset_train(config.data, device=device)
    _, dataloader_eval = setup_dataset_eval(config.data, test_mode=True, device=device)
    graph = setup_graph(config.graph, dataset_inputs_train, device=device)
    graph.eval()
    # load checkpointed information
    _load_checkpoint(config.trainer.resume_train, graph)

    if method == MethodType.PSNR:
        stats_dict = render_stats_dict(graph, dataloader_eval)
        avg_psnr = stats_dict["avg psnr"]
        avg_rays_per_sec = stats_dict["avg rays per sec"]
        avg_fps = stats_dict["avg fps"]
        print(f"Avg. PSNR: {avg_psnr:0.4f}")
        print(f"Avg. Rays per sec: {avg_rays_per_sec:0.4f}")
        print(f"Avg. FPS: {avg_fps:0.4f}")
    elif method == MethodType.TRAJ:
        # TODO(ethan): pass in camera information into argparse parser
        if traj == TrajectoryType.SPIRAL:
            camera_start = dataloader_eval.get_camera(image_idx=0)
            # TODO(ethan): pass in the up direction of the camera
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif traj == TrajectoryType.INTERP:
            camera_start = dataloader_eval.get_camera(image_idx=0)
            camera_end = dataloader_eval.get_camera(image_idx=10)
            camera_path = get_interpolated_camera_path(camera_start, camera_end, steps=30)
        render_trajectory_video(
            graph,
            camera_path,
            output_filename=output_filename,
            rendered_output_name=rendered_output_name,
            rendered_resolution_scaling_factor=rendered_resolution_scaling_factor,
            num_rays_per_chunk=config.data.dataloader_eval.num_rays_per_chunk,
        )


if __name__ == "__main__":
    dcargs.cli(main)
