#!/usr/bin/env python

import glob
import os
import sys
from contextlib import ContextDecorator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import tyro
from rich.console import Console

import block_nerf.block_nerf as block_nerf
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)

terminal = open("logs/experiment.log", "a")
writer = {"terminal": terminal, "log": sys.stdout}


@dataclass
class Args:
    model: str
    input_data_dir: Path
    output_dir: Path
    use_camera_optimizer: bool = True


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
        self.use_camera_optimizer = args.use_camera_optimizer

    def run(self):
        self.train()
        train_output_dir = self.find_evaluate_paths()
        self.eval(train_output_dir)
        self.render(interpolate=True)
        self.render(interpolate=False)

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
        cmd = f"ns-train {self.model} --data {self.input_data_dir} --output-dir {self.output_dir} --experiment-name {self.experiment_name} --max-num-iterations 2000 --vis viewer --viewer.quit-on-train-completion True"

        if not self.use_camera_optimizer:
            cmd += " --pipeline.datamanager.camera-optimizer.mode off"

        if self.model == "mipnerf":
            cmd += " nerfstudio-data"

        run_command(cmd, verbose=True)

    @my_timer("Evaluate")
    def eval(self, config: str):
        CONSOLE.print("Evaluating model")
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        output_name = f"{self.model}-{self.experiment_name}-{timestamp}"
        cmd = f"ns-eval --load-config {config} --output-path {self.output_dir}/{output_name}.json"
        run_command(cmd, verbose=True)
    
    @my_timer("Render")
    def render(self, interpolate: bool = False):
        CONSOLE.print("Rendering model")
        output_name = f"{self.model}-{self.experiment_name}"

        experiment_output_path = self.output_dir / self.experiment_name / self.model
        latest_changed_dir = max(glob.glob(f"{experiment_output_path}/*"), key=os.path.getmtime).split("/")[-1]
        config_path = experiment_output_path / latest_changed_dir / "config.yml"
        
        render_dir = self.output_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        
        # ns-render --load-config outputs/data-images-exp_combined_baseline_2/nerfacto/2023-03-28_112618/config.yml --traj filename --camera-path-filename data/images/exp_combined_baseline_2/camera_paths/2023-03-28_112618.json --output-path renders/data/images/exp_combined_baseline_2/2023-03-28_112618.mp4
        cmd = f"ns-render --load-config {config_path}"
        
        if (interpolate):
            render_path = render_dir / f"{output_name}_interpolate.mp4"
            cmd += f" --traj interpolate --output-path {render_path}"
        else:
            render_path = render_dir / f"{output_name}.mp4"
            camera_path_path = "./camera_paths/camera_path_one_lap.json"
            cmd += f" --traj filename --camera-path-filename {camera_path_path} --output-path {render_path}"
            
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

    # Create the blocks
    # new_transforms, image_indexes = block_nerf.split_transforms(input_data_dir / "transforms.json", 4)
    # block_nerf.write_transforms(new_transforms, image_indexes, input_data_dir)

    
    args = Args(
        model="nerfacto",
        input_data_dir=input_data_dir,
        output_dir=input_data_dir,
        use_camera_optimizer=args.use_camera_optimizer,
    )
    for run_dir in input_data_dir.iterdir():
        if run_dir.is_dir() and run_dir.name != "images":
            new_args = Args(
                model=args.model,
                input_data_dir=run_dir,
                output_dir=run_dir,
                use_camera_optimizer=args.use_camera_optimizer,
            )

            experiment_name = "-".join(str(run_dir).split("/")[-2:])
            pipeline = ExperimentPipeline(new_args, writer, experiment_name)
            pipeline.run()

    terminal.close()
