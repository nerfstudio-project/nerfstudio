"""
run_eval.py
"""
from typing import Tuple
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from pyrad.data.dataloader import setup_dataset_eval, setup_dataset_train
from pyrad.graphs.base import setup_graph
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


def run_inference(config: DictConfig) -> Tuple[float, float]:
    """helper function to run inference given config specifications (also used in benchmarking)

    Args:
        config (DictConfig): Configuration for loading the evaluation.

    Returns:
        Tuple[float, float]: returns both the avg psnr and avg rays per second
    """
    # setup graph and dataset
    _, dataloader_eval = setup_dataset_eval(config.data, test_mode=True)
    dataset_inputs_train, _ = setup_dataset_train(config.data)
    graph, _, _ = setup_graph(config.graph, dataset_inputs_train)
    graph.eval()

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
