"""
run_eval.py
"""
import json
import logging
import os
from datetime import date

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from tqdm import tqdm

from pyrad.engine.trainer import Trainer
from pyrad.cameras.cameras import Camera

BENCH = {
    "method": "vanilla_nerf",
    "hydra_base_dir": "outputs/",
    "benchmark_date": "05-26-2022",
    "object_list": ["mic", "ficus", "chair", "hotdog", "materials", "drums", "ship", "lego"],
}


def load_best_ckpt(hydra_dir: str, config: DictConfig) -> str:
    """helper function to update config with latest checkpoint in specified hydra_dir"""
    model_dir = os.path.join(hydra_dir, config.graph.model_dir)
    latest_ckpt = os.listdir(model_dir)
    latest_ckpt.sort()
    latest_ckpt = latest_ckpt[-1]
    step = os.path.splitext(latest_ckpt)[0].split("-")[-1]
    config.graph.resume_train.load_dir = model_dir
    config.graph.resume_train.load_step = int(step)
    config.logging.writer.TensorboardWriter.save_dir = "/tmp/"
    return os.path.join(model_dir, latest_ckpt)


def main():
    """Main function."""
    # benchmarks = {}
    # hydra_base_dir = BENCH["hydra_base_dir"]
    # method = BENCH["method"]
    # benchmark_date = BENCH["benchmark_date"]
    # for dataset in tqdm(BENCH["object_list"]):

    # hydra_dir = f"{hydra_base_dir}/blender_{dataset}_{benchmark_date}/{method}/"
    # basename = os.listdir(hydra_dir)[0]
    # hydra_dir = f"{hydra_dir}/{basename}"
    # initialize(config_path=os.path.join("../../", hydra_dir, ".hydra/"))
    # config = compose("configs/flat.yaml")

    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name="flat.yaml")
    trainer = Trainer(config, local_rank=0, world_size=1)
    trainer.setup(test_mode=False)

    # create a camera trajectory
    print("Running eval.")
    # camera = Camera()
    # trainer.graph.eval()
    # camera_ray_bundle = camera.get_camera_ray_bundle(device=trainer.device)
    image_idx = 0
    camera_ray_bundle, batch = trainer.dataloader_eval.get_data_from_image_idx(image_idx)
    image_height, image_width = camera_ray_bundle.origins.shape[:2]
    print(image_height, image_width)
    outputs = trainer.graph.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    
    # print(outputs)

    # avg_psnr = 0
    # for step, image_idx in enumerate(config.data.val_image_indices):
    #     with torch.no_grad():
    #         psnr = trainer.test_image(image_idx=image_idx, step=step)
    #     avg_psnr = (step * avg_psnr + psnr) / (step + 1)
    # benchmarks[dataset] = (avg_psnr, ckpt)
    # GlobalHydra.instance().clear()

    # benchmark_info = {"bench": BENCH, "results": benchmarks}
    # timestamp = date.today().strftime("%b-%d-%Y")
    # json_file = os.path.join(BENCH["hydra_base_dir"], f"{timestamp}.json")
    # with open(json_file, "w", encoding="utf8") as f:
    #     json.dump(benchmark_info, f, indent=2)
    # logging.info("saved benchmark results to %s", json_file)


if __name__ == "__main__":
    main()
