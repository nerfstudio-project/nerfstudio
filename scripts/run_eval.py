"""
run_eval.py
"""
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from pyrad.data.dataloader import setup_dataset_eval
from pyrad.graphs.base import setup_graph


@hydra.main(config_path="../configs", config_name="graph_default.yaml")
def main(config: DictConfig):
    """Main function."""
    # setup graph and dataset
    dataset_inputs_eval, dataloader_eval = setup_dataset_eval(config.data, test_mode=True)
    graph, _, _ = setup_graph(config.graph, dataset_inputs_eval)
    graph.eval()

    # calculate average psnr across test dataset
    # TODO(ethan): trajector specification
    avg_psnr = 0
    for step, (camera_ray_bundle, batch) in tqdm(enumerate(dataloader_eval)):
        with torch.no_grad():
            image_idx = int(camera_ray_bundle.camera_indices[0, 0])
            outputs = graph.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            psnr = graph.log_test_image_outputs(image_idx, step, batch, outputs)
        avg_psnr = (step * avg_psnr + psnr) / (step + 1)


if __name__ == "__main__":
    main()
