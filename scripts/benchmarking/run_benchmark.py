"""
run_benchmark.py
"""
import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig

from scripts.run_eval import run_inference_from_config


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


def _load_hydra_config(hydra_dir: str, benchmark_time=None, benchmark_date=None) -> Tuple[DictConfig, str]:
    """helper function to load the specified hydra config from the specified directory

    Args:
        hydra_dir (str): base directory for the specified run (where the flattened config.yaml is stored)

    Returns:
        DictConfig: returns the loaded hydra dictionary config
    """
    if benchmark_time:
        assert benchmark_date is not None
        reformat_date = datetime.strptime(benchmark_date, "%m-%d-%Y").strftime("%Y-%m-%d")
        basename = f"{reformat_date}_{benchmark_time}"
    else:
        basename = sorted(os.listdir(hydra_dir))[-1]
    hydra_dir = f"{hydra_dir}/{basename}"
    initialize(version_base="1.2", config_path=os.path.join("../../", hydra_dir, ".hydra/"))
    config = compose("config.yaml")
    return config, basename


def _calc_avg(stat_name: str, benchmark: Dict[str, Any]):
    """helper to calculate the average across all objects in dataset"""
    stats = []
    for _, object_stats in benchmark.items():
        stats.append(object_stats[stat_name])
    return np.mean(stats)


def main(args):
    """Main function."""
    benchmarks = {}
    # set up trainer, config, and checkpoint loading
    hydra_dir = f"{args.hydra_base_dir}/blender_{args.item_name}_{args.benchmark_date}/{args.graph}/"
    config, basename = _load_hydra_config(
        hydra_dir, benchmark_time=args.benchmark_time, benchmark_date=args.benchmark_date
    )
    # config.data.dataloader_eval.image_indices = None
    ckpt = _load_best_ckpt(hydra_dir, config.trainer)

    # run evaluation
    stats_dict = run_inference_from_config(config)
    stats_dict["checkpoint"] = ckpt
    benchmarks[args.item_name] = stats_dict

    avg_rays_per_sec = _calc_avg("avg rays per sec", benchmarks)
    avg_fps = _calc_avg("avg fps", benchmarks)

    # output benchmark statistics to a json file
    benchmark_info = {
        "meta info": {"graph": args.graph, "benchmark_date": args.benchmark_date, "hydra": args.hydra_base_dir},
        "avg rays per sec": avg_rays_per_sec,
        "avg fps": avg_fps,
        "results": benchmarks,
    }
    json_dir = os.path.join(args.hydra_base_dir, args.graph)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json_file = os.path.join(json_dir, f"{basename}_{args.item_name}.json")

    # if benchmark file already exists, update the results dictionary by overriding existing info or appending.
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf8") as f:
            existing_data = json.load(f)
        existing_benchmarks = existing_data["results"]
        existing_benchmarks.update(benchmarks)
        existing_data["results"] = existing_benchmarks
        benchmark_info = existing_data

    with open(json_file, "w", encoding="utf8") as f:
        json.dump(benchmark_info, f, indent=2)
    logging.info("saved benchmark results to %s", json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--graph",
        type=str,
        required=True,
        help="name of nerf graph to benchmark. Note: name without the 'graph_' prefix",
    )
    parser.add_argument(
        "-d",
        "--benchmark_date",
        type=str,
        required=True,
        help="date of run to benchmark provided in '%%m-%%d-%%Y' format.",
    )
    parser.add_argument(
        "-i",
        "--item_name",
        type=str,
        required=True,
        choices=["mic", "ficus", "chair", "hotdog", "materials", "drums", "ship", "lego"],
        help="name of item in blender dataset to benchmark",
    )
    parser.add_argument(
        "-t",
        "--benchmark_time",
        type=str,
        default=None,
        help="timestamp of the run to benchmark; if None, will default to running most recent timestamp",
    )
    parser.add_argument("-o", "--hydra_base_dir", type=str, default="outputs/", help="hydra base output path")
    parsed_args = parser.parse_args()
    main(parsed_args)
