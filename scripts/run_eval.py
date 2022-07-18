"""
run_eval.py
"""
import argparse
import os
from typing import Dict

import mediapy as media
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from tqdm import tqdm

from pyrad.cameras.camera_paths import get_spiral_path
from pyrad.data.dataloader import setup_dataset_eval, setup_dataset_train
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


def run_inference(config: DictConfig, local_rank: int = 0, world_size: int = 1) -> Dict[str, float]:
    """helper function to run inference given config specifications (also used in benchmarking)

    Args:
        config (DictConfig): Configuration for loading the evaluation.

    Returns:
        Tuple[float, float]: returns both the avg psnr and avg rays per second
    """
    device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
    # setup graph and dataset
    dataset_inputs_train, _ = setup_dataset_train(config.data, device=device)
    _, dataloader_eval = setup_dataset_eval(config.data, test_mode=True, device=device)
    graph = setup_graph(config.graph, dataset_inputs_train, device=device)
    graph.eval()

    # load checkpointed information
    _load_checkpoint(config.trainer.resume_train, graph)

    # calculate average psnr across test dataset
    # TODO(ethan): trajector specification
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


def create_spiral_video(
    config: DictConfig, local_rank: int = 0, world_size: int = 1, output_filename: str = None
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        config (DictConfig): Configuration for loading the evaluation.
        local_rank (int): Local rank of the process.
        world_size (int): Total number of GPUs.
        output_filename (str): Name of the output file.
    """
    print("Creating spiral video")
    device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
    # setup graph and dataset
    dataset_inputs_train, _ = setup_dataset_train(config.data, device=device)
    _, dataloader_eval = setup_dataset_eval(config.data, test_mode=True, device=device)
    graph = setup_graph(config.graph, dataset_inputs_train, device=device)

    # load checkpointed information
    _load_checkpoint(config.trainer.resume_train, graph)

    # get a trajecory
    start_camera = dataloader_eval.get_camera(image_idx=0)
    # TODO(ethan): replace with radius with radiuses, based on camera pose percentiles
    # see original nerf paper code for details
    camera_path = get_spiral_path(start_camera, steps=60, radius=0.5)

    images = []
    for camera in tqdm(camera_path.cameras):
        camera.cx /= 4
        camera.cy /= 4
        camera.fx /= 4
        camera.fy /= 4
        camera_ray_bundle = camera.get_camera_ray_bundle().to(device)
        camera_ray_bundle.num_rays_per_chunk = 4096
        outputs = graph.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        # TODO: don't hardcode the key! this will break for some nerf Graphs
        image = outputs["rgb"].cpu().numpy()
        images.append(image)

    seconds = 5.0
    fps = len(images) / seconds
    media.write_video(output_filename, images, fps=fps)


def main():
    """Main function."""

    parser = argparse.ArgumentParser(description="Run the evaluation of a model.")
    parser.add_argument(
        "--method",
        type=str,
        default="psnr",
        choices=["psnr", "traj"],
        help="Specify which type of evaluation method to run.",
    )
    parser.add_argument("--traj", type=str, default="spiral", choices=["spiral", "interp"])
    parser.add_argument("--output-filename", type=str, default="output.mp4")
    parser.add_argument("--config-name", type=str, default="graph_default.yaml")
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    config_path = "../configs"
    initialize(version_base="1.2", config_path=config_path)
    config = compose(args.config_name, overrides=args.overrides)

    assert config.trainer.resume_train.load_dir, "Please specify checkpoint load path"
    assert args.traj != "interp", "Camera pose interpolation trajectory isn't yet implemented."

    if args.method == "psnr":
        stats_dict = run_inference(config)
        avg_psnr = stats_dict["avg psnr"]
        avg_rays_per_sec = stats_dict["avg rays per sec"]
        avg_fps = stats_dict["avg fps"]
        print(f"Avg. PSNR: {avg_psnr:0.4f}")
        print(f"Avg. Rays per sec: {avg_rays_per_sec:0.4f}")
        print(f"Avg. FPS: {avg_fps:0.4f}")
    elif args.method == "traj":
        create_spiral_video(config, output_filename=args.output_filename)


if __name__ == "__main__":
    main()
