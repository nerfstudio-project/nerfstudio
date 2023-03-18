#!/usr/bin/env python

import glob
import os
import sys
from contextlib import ContextDecorator
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import cv2
import tyro
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)

terminal = open("logs/experiment.log", "a")
writer = {"terminal": terminal, "log": sys.stdout}


@dataclass
class Args:
    model: str
    input_data_dir: Path
    eval_output_dir: Path


class my_timer(ContextDecorator):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = timer()

    def __exit__(self, *args):
        self.end = timer()
        msg = f"[time] {self.name}: {self.end - self.start}\n"
        writer["terminal"].write(msg)
        writer["log"].write(msg)
        writer["terminal"].flush()
        writer["log"].flush()


class ExperimentPipeline:
    def __init__(self, args: Args, writer):
        self.args = args
        self.writer = writer
        self.terminal = sys.stdout
        self.log = open("experiment_log.txt", "a")

    def run(self):
        self.train()
        train_output_dir, experiment_name = self.find_evaluate_paths()
        self.eval(train_output_dir, experiment_name)

    def write(self, text: str):
        self.writer["terminal"].write(text)
        self.writer["log"].write(text)
        self.writer["terminal"].flush()
        self.writer["log"].flush()

    def find_evaluate_paths(self):
        experiment_name = "-".join(str(self.args.eval_output_dir).split("/"))
        train_output_dir = f"outputs/{experiment_name}/{self.args.model}"
        latest_changed_dir = max(glob.glob(f"{train_output_dir}/*"), key=os.path.getmtime).split("/")[-1]
        train_output_dir = os.path.join(train_output_dir, latest_changed_dir, "config.yml")
        return train_output_dir, experiment_name

    # ns-train instant-ngp --data data/videos/tier2 --trainer.load_dir $output_path --viewer.start-train False
    @my_timer("Train")
    def train(self):
        input_data_dir = self.args.input_data_dir
        CONSOLE.print(f"Training model\nModel: {self.args.model}\nInput dir: {input_data_dir}")
        cmd = f"ns-train {self.args.model} --data {input_data_dir} --vis wandb --viewer.quit-on-train-completion True"

        if self.args.model == "mipnerf":
            cmd += " nerfstudio-data"

        run_command(cmd, verbose=True)

    @my_timer("Evaluate")
    def eval(self, config: str, output_name: str):
        CONSOLE.print("Evaluating model")
        cmd = f"ns-eval --load-config {config} --output-path evals/{output_name}.json"
        run_command(cmd, verbose=True)


if __name__ == "__main__":
    """
    Run this script to process input-data, train and evaluate a model.
    Example: ./nerf_carla_pipeline.py --model nerfacto --input_data ../carlo/runs/exp_capacity_1 --eval_output_dir ../carlo/runs/exp_capacity_1
    Old Example: ./nerf_pipeline.py --model nerfacto --data_source images --input_data data/videos/hovedbygget/images_old --output_dir data/videos/hovedbygget
    """
    args = tyro.cli(Args)
    print(args)

    # Run pipeline in sequence
    input_data_dir = Path(args.input_data_dir)
    args = Args(model="nerfacto", input_data_dir=input_data_dir, eval_output_dir=input_data_dir)
    for run_dir in input_data_dir.iterdir():
        if run_dir.is_dir():
            args.input_data_dir = run_dir
            pipeline = ExperimentPipeline(args, writer)
            pipeline.run()
    
    terminal.close()
