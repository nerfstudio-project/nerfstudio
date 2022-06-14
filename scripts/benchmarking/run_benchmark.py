"""
run_benchmark.py
"""
import json
import logging
import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from tqdm import tqdm

from scripts.run_eval import run_inference

BENCH = {
    "method": "vanilla_nerf",
    "hydra_base_dir": "outputs/",
    "benchmark_date": "05-26-2022",
    "object_list": ["mic", "ficus", "chair", "hotdog", "materials", "drums", "ship", "lego"],
}


def _load_best_ckpt(hydra_dir: str, config: DictConfig) -> str:
    """helper function to update config with latest checkpoint in specified hydra_dir

    Args:
        hydra_dir (str): base directory for the specified run (where the model directories are stored)
        config (DictConfig): configuration for the specified run.

    Returns:
        str: path to the most recent checkpoint sorted by timestamps and step number
    """
    model_dir = os.path.join(hydra_dir, config.graph.model_dir)
    latest_ckpt = os.listdir(model_dir)
    latest_ckpt.sort()
    latest_ckpt = latest_ckpt[-1]
    step = os.path.splitext(latest_ckpt)[0].split("-")[-1]
    config.graph.resume_train.load_dir = model_dir
    config.graph.resume_train.load_step = int(step)
    config.logging.writer.TensorboardWriter.save_dir = "/tmp/"
    return os.path.join(model_dir, latest_ckpt)


def _load_hydra_config(hydra_dir: str) -> DictConfig:
    """helper function to load the specified hydra config from the specified directory

    Args:
        hydra_dir (str): base directory for the specified run (where the flattened config.yaml is stored)

    Returns:
        DictConfig: returns the loaded hydra dictionary config
    """
    basename = os.listdir(hydra_dir)[0]
    hydra_dir = f"{hydra_dir}/{basename}"
    initialize(config_path=os.path.join("../../", hydra_dir, ".hydra/"))
    config = compose("config.yaml")
    return config


def main():
    """Main function."""
    benchmarks = {}
    hydra_base_dir = BENCH["hydra_base_dir"]
    method = BENCH["method"]
    benchmark_date = BENCH["benchmark_date"]
    for dataset in tqdm(BENCH["object_list"]):
        # set up trainer, config, and checkpoint loading
        hydra_dir = f"{hydra_base_dir}/blender_{dataset}_{benchmark_date}/{method}/"
        config = _load_hydra_config(hydra_dir)
        ckpt = _load_best_ckpt(hydra_dir, config)

        # run evaluation
        avg_psnr, avg_rays_per_sec = run_inference(config)
        benchmarks[dataset] = {"avg psnr": avg_psnr, "avg rays/s": avg_rays_per_sec, "checkpoint": ckpt}

        # reset hydra config
        GlobalHydra.instance().clear()

    # output benchmark statistics to a json file
    benchmark_info = {"bench": BENCH, "results": benchmarks}
    timestamp = BENCH["benchmark_date"]
    json_file = os.path.join(BENCH["hydra_base_dir"], f"{timestamp}.json")
    with open(json_file, "w", encoding="utf8") as f:
        json.dump(benchmark_info, f, indent=2)
    logging.info("saved benchmark results to %s", json_file)


if __name__ == "__main__":
    main()
