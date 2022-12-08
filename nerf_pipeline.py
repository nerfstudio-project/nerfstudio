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
writer = {
    "terminal": terminal,
    "log": sys.stdout
}

@dataclass
class Args:
    model: str
    data_source: Literal["video", "images", "polycam"]
    input_data_dir: Path
    output_dir: Path
    percent_frames: float = None

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
        self.write(f"\n{int(self.args.percent_frames * 100)}%\n")
        self.process_data()
        self.train()
        train_output_dir, experiment_name = self.find_evaluate_paths()
        self.eval(train_output_dir, experiment_name)
            
    def write(self, text: str):
        self.writer["terminal"].write(text)
        self.writer["log"].write(text)
        self.writer["terminal"].flush()
        self.writer["log"].flush()

    def find_evaluate_paths(self):
        experiment_name = '-'.join(str(self.args.output_dir).split('/'))
        train_output_dir = f"outputs/{experiment_name}/{self.args.model}"
        latest_changed_dir = max(glob.glob(f"{train_output_dir}/*"), key=os.path.getmtime).split("/")[-1]
        train_output_dir = os.path.join(train_output_dir, latest_changed_dir, "config.yml")
        return train_output_dir, experiment_name

    @my_timer("Process")
    def process_data(self):
        CONSOLE.print("Processing data")
        cmd = f"ns-process-data {self.args.data_source} --data {self.args.input_data_dir} --output-dir {self.args.output_dir} --verbose"

        if self.args.data_source == "video" and self.args.percent_frames is not None:
            video = cv2.VideoCapture(str(self.args.input_data_dir))
            total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames = int(total * self.args.percent_frames)
            self.write(f"Num frames: {num_frames}/{total}\n")
            cmd = f"ns-process-data {self.args.data_source} --data {self.args.input_data_dir} --output-dir {self.args.output_dir} --num-frames-target {num_frames} --matching-method sequential --verbose"

        if os.environ["SLURM_JOB"]:
            cmd += f" --colmap-cmd 'singularity exec --nv  /cluster/apps/dev/colmap/colmap_latest.sif colmap'"

        print(cmd)
        run_command(cmd, verbose=True)

    # ns-train instant-ngp --data data/videos/tier2 --trainer.load_dir $output_path --viewer.start-train False
    @my_timer("Train")
    def train(self):
        input_data_dir = self.args.output_dir
        CONSOLE.print(f"Training model\nModel: {self.args.model}\nInput dir: {input_data_dir}")
        cmd = f"ns-train {self.args.model} --data {input_data_dir} --vis wandb --viewer.quit-on-train-completion True"
        
        if self.args.model == "mipnerf":
            cmd += " nerfstudio-data"

        run_command(cmd, verbose=True)
        
    @my_timer("Evaluate")
    def eval(self, config: Path, output_name: Path):
        CONSOLE.print("Evaluating model")
        cmd = f"ns-eval --load-config {config} --output-path evals/{output_name}.json"
        run_command(cmd, verbose=True)

if __name__ == '__main__':
    """
    Run this script to process input-data, train and evaluate a model.
    Example: ./nerf_pipeline.py --model nerfacto --data_source images --input_data data/videos/hovedbygget/images_old --output_dir data/videos/hovedbygget
    """
    args = tyro.cli(Args)
    print(args)

    pipeline = ExperimentPipeline(args, writer)
    pipeline.run()

    terminal.close()
