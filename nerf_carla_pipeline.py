#!/usr/bin/env python

import glob
import os
import sys
from contextlib import ContextDecorator
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime

import tyro
from rich.console import Console

from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)

terminal = open("logs/experiment.log", "a")
writer = {"terminal": terminal, "log": sys.stdout}


@dataclass
class Args:
    model: str
    input_data_dir: Path
    output_dir: Path


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
    def __init__(self, args: Args, writer, experiment_name: str):
        self.args = args
        self.writer = writer
        self.terminal = sys.stdout
        self.log = open("experiment_log.txt", "a")

        self.input_data_dir = args.input_data_dir
        self.output_dir = args.output_dir
        self.model = args.model
        self.experiment_name = experiment_name

    def run(self):
        self.train()
        train_output_dir = self.find_evaluate_paths()
        self.eval(train_output_dir)

    def write(self, text: str):
        self.writer["terminal"].write(text)
        self.writer["log"].write(text)
        self.writer["terminal"].flush()
        self.writer["log"].flush()

    def find_evaluate_paths(self):
        train_output_dir = self.output_dir / self.experiment_name / self.model
        latest_changed_dir = max(glob.glob(f"{train_output_dir}/*"), key=os.path.getmtime).split("/")[-1]
        train_output_dir = os.path.join(train_output_dir, latest_changed_dir, "config.yml")
        return train_output_dir

    # ns-train instant-ngp --data data/videos/tier2 --trainer.load_dir $output_path --viewer.start-train False
    @my_timer("Train")
    def train(self):
        CONSOLE.print(f"Training model\nModel: {self.model}\nInput dir: {self.input_data_dir}")
        cmd = f"ns-train {self.model} --data {self.input_data_dir} --output-dir {self.output_dir} --experiment-name {self.experiment_name} --trainer.max-num-iterations 15000 --vis wandb --viewer.quit-on-train-completion True"

        if self.model == "mipnerf":
            cmd += " nerfstudio-data"

        run_command(cmd, verbose=True)

    @my_timer("Evaluate")
    def eval(self, config: str):
        CONSOLE.print("Evaluating model")
        output_name = f"{self.model}-{self.experiment_name}"
        cmd = f"ns-eval --load-config {config} --output-path {self.output_dir}/{output_name}.json"
        run_command(cmd, verbose=True)
    
    @my_timer("Render")
    def render(self, config_path: Path):
        CONSOLE.print("Rendering model")
        output_name = f"{self.model}-{self.experiment_name}"

        experiment_output_path = self.output_dir / self.experiment_name / self.model
        latest_changed_dir = max(glob.glob(f"{experiment_output_path}/*"), key=os.path.getmtime).split("/")[-1]
        config_path = experiment_output_path / latest_changed_dir / "config.yml"

        render_path = self.output_dir / "renders" / output_name
        # ns-render --load-config outputs/data-images-exp_combined_baseline_2/nerfacto/2023-03-28_112618/config.yml --traj filename --camera-path-filename data/images/exp_combined_baseline_2/camera_paths/2023-03-28_112618.json --output-path renders/data/images/exp_combined_baseline_2/2023-03-28_112618.mp4
        cmd = f"ns-render --load-config {config_path} --output-path {self.output_dir}/{output_name}.json"
        run_command(cmd, verbose=True)


if __name__ == "__main__":
    """
    Run this script to process input-data, train and evaluate a model.
    Example: ./nerf_carla_pipeline.py --model nerfacto --input_data ../carlo/runs/exp_capacity_1 --output_dir ../carlo/runs/exp_capacity_1
    Old Example: ./nerf_pipeline.py --model nerfacto --data_source images --input_data data/videos/hovedbygget/images_old --output_dir data/videos/hovedbygget
    """
    args = tyro.cli(Args)
    print(args)

    # Run pipeline in sequence
    input_data_dir = Path(args.input_data_dir)
    
    args = Args(
        model="nerfacto",
        input_data_dir=input_data_dir,
        output_dir=input_data_dir,
    )
    for run_dir in input_data_dir.iterdir():
        if run_dir.is_dir():
            new_args = Args(
                model=args.model,
                input_data_dir=run_dir,
                output_dir=run_dir,
            )

            experiment_name = "-".join(str(run_dir).split("/")[-2:])
            pipeline = ExperimentPipeline(new_args, writer, experiment_name)
            pipeline.run()

    terminal.close()
