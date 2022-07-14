"""
run_benchmark.py
"""
import argparse
import json
import logging
import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from tqdm import tqdm

from scripts.run_eval import run_inference


OBJECT_LIST = ["mic", "ficus", "chair", "hotdog", "materials", "drums", "ship", "lego"]


def _load_best_ckpt(hydra_dir: str, config: DictConfig) -> str:
    """helper function to update config with latest checkpoint in specified hydra_dir

    Args:
        hydra_dir (str): base directory for the specified run (where the model directories are stored)
        config (DictConfig): configuration for the specified run.

    Returns:
        str: path to the most recent checkpoint sorted by timestamps and step number
    """
    # get the most recent run in the model directory
    latest_run = os.listdir(hydra_dir)
    latest_run.sort()
    latest_run = latest_run[-1]
    # get the latest checkpoint name
    model_dir = os.path.join(hydra_dir, latest_run, config.model_dir)
    latest_ckpt = os.listdir(model_dir)
    latest_ckpt.sort()
    latest_ckpt = latest_ckpt[-1]
    step = os.path.splitext(latest_ckpt)[0].split("-")[-1]
    config.resume_train.load_dir = model_dir
    config.resume_train.load_step = int(step)
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
    initialize(version_base="1.2", config_path=os.path.join("../../", hydra_dir, ".hydra/"))
    config = compose("config.yaml")
    return config


def main(args):
    """Main function."""
    benchmarks = {}
    for dataset in tqdm(OBJECT_LIST):
        # set up trainer, config, and checkpoint loading
        hydra_dir = f"{args.hydra_base_dir}/blender_{dataset}_{args.benchmark_date}/{args.graph}/"
        config = _load_hydra_config(hydra_dir)
        ckpt = _load_best_ckpt(hydra_dir, config.trainer)

        # run evaluation
        avg_psnr, avg_rays_per_sec = run_inference(config)
        benchmarks[dataset] = {"avg psnr": avg_psnr, "avg rays/s": avg_rays_per_sec, "checkpoint": ckpt}

        # reset hydra config
        GlobalHydra.instance().clear()

    # output benchmark statistics to a json file
    benchmark_info = {
        "meta info": {"graph": args.graph, "benchmark_date": args.benchmark_date, "hydra": args.hydra_base_dir},
        "results": benchmarks,
    }
    json_file = os.path.join(args.hydra_base_dir, f"{args.benchmark_date}.json")
    with open(json_file, "w", encoding="utf8") as f:
        json.dump(benchmark_info, f, indent=2)
    logging.info("saved benchmark results to %s", json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", type=str, required=True, help="name of nerf graph to benchmark")
    parser.add_argument("-d", "--benchmark_date", type=str, required=True, help="timestamp of run to benchmark")
    parser.add_argument("-o", "--hydra_base_dir", type=str, default="outputs/", help="hydra base output path")
    parsed_args = parser.parse_args()
    main(parsed_args)
