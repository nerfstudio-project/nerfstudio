"""
run_benchmark.py
"""
import json
import os

import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from mattport.nerf.trainer import Trainer

BENCH = {
    "mic": "outputs/blender_mic/vanilla_nerf/2022-05-11_192109/",
    "ficus": "outputs/blender_ficus/vanilla_nerf/2022-05-11_192109/",
    "chair": "outputs/blender_chair/vanilla_nerf/2022-05-11_192109/",
    "hotdog": "outputs/blender_hotdog/vanilla_nerf/2022-05-11_192109/",
    "materials": "outputs/blender_materials/vanilla_nerf/2022-05-11_192109/",
    "drums": "outputs/blender_drums/vanilla_nerf/2022-05-11_192109/",
    "ship": "outputs/blender_ship/vanilla_nerf/2022-05-11_192109/",
    "lego": "outputs/blender_lego/vanilla_nerf/2022-05-11_192109/",
}


def load_best_ckpt(hydra_dir: str, config: DictConfig):
    """helper function to update config with latest checkpoint in specified hydra_dir"""
    model_dir = os.path.join(hydra_dir, config.graph.model_dir)
    latest_ckpt = os.listdir(model_dir)
    latest_ckpt.sort()
    latest_ckpt = latest_ckpt[-1]
    step = os.path.splitext(latest_ckpt)[0].split("-")[-1]
    config.graph.resume_train.load_dir = model_dir
    config.graph.resume_train.load_step = int(step)
    config.logging.writer.TensorboardWriter.save_dir = "/tmp/"


def main():
    """Main function."""
    benchmarks = {}
    for dataset, hydra_dir in BENCH.items():
        initialize(config_path=os.path.join("../../", hydra_dir, ".hydra/"))
        config = compose("config.yaml")
        load_best_ckpt(hydra_dir, config)
        trainer = Trainer(config, local_rank=0, world_size=1)
        trainer.setup(test_mode=True)
        avg_psnr = 0
        for step, image_idx in enumerate(config.data.validation_image_indices):
            with torch.no_grad():
                psnr = trainer.test_image(image_idx=image_idx, step=step)
            avg_psnr = (step * avg_psnr + psnr) / (step + 1)
        benchmarks[dataset] = avg_psnr

    benchmark_info = {"bench": BENCH, "results": benchmarks}
    # TODO(more sophisticated saving to output directory)
    with open("/tmp/benchmark.json", "w", encoding="utf8") as f:
        json.dump(benchmark_info, f)


if __name__ == "__main__":
    main()
