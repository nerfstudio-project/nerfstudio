"""
run_eval.py
"""
import logging
import os
from typing import Tuple
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

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
    load_path = os.path.join(config.load_dir, f"step-{config.load_step:09d}.ckpt")
    assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    graph.load_checkpoint(loaded_state)
    logging.info("done loading checkpoint from %s", load_path)


def run_inference(config: DictConfig, local_rank: int = 0, world_size: int = 1) -> Tuple[float, float]:
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

    # load checkpointed information
    _load_checkpoint(config.graph.resume_train, graph)

    # calculate average psnr across test dataset
    # TODO(ethan): trajector specification
    avg_psnr = 0
    avg_rays_per_sec = 0
    for step, (camera_ray_bundle, batch) in tqdm(enumerate(dataloader_eval)):
        with TimeWriter(writer=None, name=None, write=False) as t:
            with torch.no_grad():
                image_idx = int(camera_ray_bundle.camera_indices[0, 0])
                outputs = graph.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                psnr = graph.log_test_image_outputs(image_idx, step, batch, outputs)
        avg_rays_per_sec = _update_avg(avg_rays_per_sec, camera_ray_bundle.origins.shape[0] / t.duration, step)
        avg_psnr = _update_avg(avg_psnr, psnr, step)
    return avg_psnr, avg_rays_per_sec


@hydra.main(config_path="../configs", config_name="graph_default.yaml")
def main(config: DictConfig):
    """Main function."""
    assert config.graph.resume_train.load_dir, "Please specify checkpoint load path"
    assert config.graph.resume_train.load_step, "Please specify checkpoint step to load"

    avg_psnr, avg_rays_per_sec = run_inference(config)

    print(f"Avg. PSNR: {avg_psnr:0.4f}")
    print(f"Avg. Rays per sec: {avg_rays_per_sec:0.4f}")


if __name__ == "__main__":
    main()
