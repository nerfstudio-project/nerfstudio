#!/usr/bin/env python

import glob
import os
import sys
from contextlib import ContextDecorator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Optional

import tyro
from rich.console import Console

import block_nerf.block_nerf as block_nerf
from block_nerf.block_nerf import transform_camera_path
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.scripts import run_command
from scripts.eval import ComputePSNR
from scripts.render import BlockNerfRenderTrajectory, RenderTrajectory
from scripts.train import launch, train_loop

CONSOLE = Console(width=120)

terminal = open("logs/experiment.log", "a")
writer = {"terminal": terminal, "log": sys.stdout}


@dataclass
class Args:
    model: str
    input_data_dir: Path
    output_dir: Path
    use_camera_optimizer: bool = True
    block_segments: Optional[int] = None


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
        self.block_segments = args.block_segments

        self.trainer_config: TrainerConfig = self.setup()
        self.trainer_config.setup()
        self.checkpoint_dir = self.trainer_config.get_checkpoint_dir()
        self.model_dir = self.checkpoint_dir.parent

    def run(self):
        self.train()
        self.eval()
        self.render(interpolate=True)
        # self.render(interpolate=False)

    def setup(self) -> TrainerConfig:
        config = method_configs[self.model]
        config.set_timestamp()
        config.data = self.input_data_dir
        config.output_dir = self.output_dir
        config.experiment_name = self.experiment_name
        config.vis = "viewer"
        config.viewer.quit_on_train_completion = True
        config.max_num_iterations = 500

        if not self.use_camera_optimizer:
            config.pipeline.datamanager.camera_optimizer.mode = "off"

        if config.data:
            CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
            config.pipeline.datamanager.data = config.data

        # if config.load_config:
        #     CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        #     config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

        config.print_to_terminal()
        config.save_config()
        return config

    def write(self, text: str):
        self.writer["terminal"].write(text)
        self.writer["log"].write(text)
        self.writer["terminal"].flush()
        self.writer["log"].flush()

    @my_timer("Train")
    def train(self):
        CONSOLE.print(f"Training model\nModel: {self.model}\nInput dir: {self.input_data_dir}")

        launch(
            main_func=train_loop,
            num_gpus_per_machine=self.trainer_config.machine.num_gpus,
            num_machines=self.trainer_config.machine.num_machines,
            machine_rank=self.trainer_config.machine.machine_rank,
            dist_url=self.trainer_config.machine.dist_url,
            config=self.trainer_config,
        )

    @my_timer("Evaluate")
    def eval(self):
        CONSOLE.print("Evaluating model")

        eval_config = ComputePSNR(
            load_config=self.model_dir / "config.yml",
            output_path=self.model_dir / "eval.json",
        )

        eval_config.main()

    @my_timer("Render")
    def render(self, interpolate: bool = False):
        CONSOLE.print("Rendering model")
        output_name = "render"
        config_path = self.model_dir / "config.yml"
        render_dir = self.model_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)

        render_config = RenderTrajectory(
            load_config=config_path,
        )

        if interpolate:
            render_path = render_dir / f"{output_name}_interpolate.mp4"
            render_config.traj = "interpolate"
            render_config.output_path = render_path
        else:
            render_path = render_dir / f"{output_name}.mp4"
            camera_path_path = self.get_camera_path(self.model_dir)
            render_config.traj = "filename"
            render_config.camera_path_filename = camera_path_path
            render_config.output_path = render_path

        render_config.main()

    def render_blocknerf(self, block_paths: List[Path]):
        CONSOLE.print("Block NeRF Rendering model")
        output_name = f"blocknerf-{self.experiment_name}"

        # Assumes that there're only one config file and one dataparser_transforms file in each block
        config_files = {
            block_path.name: Path(glob.glob(f"{block_path}/**/config.yml", recursive=True)[0])
            for block_path in block_paths
        }
        dataparser_transforms_paths = {
            block_path.name: Path(glob.glob(f"{block_path}/**/dataparser_transforms.json", recursive=True)[0])
            for block_path in block_paths
        }
        render_dir = self.output_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_path = render_dir / f"{output_name}.mp4"
        block_lookup = block_nerf.get_block_lookup(self.output_dir, block_paths)

        combined_transformed_camera_path = block_nerf.transform_to_single_camera_path(
            camera_path_path=Path(
                "camera_paths/camera_path_transformed_original.json"
            ),  # This is the original camera path transformed to the original CARLA coordinate system
            block_lookup=block_lookup,
            dataparser_transform_paths=dataparser_transforms_paths,
            export_dir=self.output_dir,
        )

        # Use this class to run the render script programmatically
        BlockNerfRenderTrajectory(
            config_files=config_files,
            traj="filename",
            camera_path_filename=combined_transformed_camera_path,
            output_path=render_path,
            eval_num_rays_per_chunk=1 << 15,
            block_lookup=block_lookup,
        ).main()

    def get_camera_path(self, model_path: Path) -> Path:
        if not self.block_segments:
            return Path("camera_paths/camera_path_one_lap_final.json")

        export_path = self.output_dir / "camera_path_transformed.json"
        if export_path.exists():
            return export_path

        original_camera_path_path = Path("camera_paths/camera_path_transformed_original.json")
        target_dataparser_transforms_path = model_path / "dataparser_transforms.json"

        return transform_camera_path(
            original_camera_path_path, target_dataparser_transforms_path, export_path=export_path
        )


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
    block_paths = []
    if args.block_segments:
        new_transforms, image_indexes = block_nerf.split_transforms(
            input_data_dir / "transforms.json", args.block_segments
        )
        block_paths = block_nerf.write_transforms(new_transforms, image_indexes, input_data_dir)

    args = Args(
        model="nerfacto",
        input_data_dir=input_data_dir,
        output_dir=input_data_dir,
        use_camera_optimizer=args.use_camera_optimizer,
        block_segments=args.block_segments,
    )
    for run_dir in input_data_dir.iterdir():
        if run_dir.is_dir() and run_dir.name != "images":
            new_args = Args(
                model=args.model,
                input_data_dir=run_dir,
                output_dir=run_dir,
                use_camera_optimizer=args.use_camera_optimizer,
                block_segments=args.block_segments,
            )

            experiment_name = "-".join(str(run_dir).split("/")[-2:])
            pipeline = ExperimentPipeline(new_args, writer, experiment_name)
            pipeline.run()

    # Run the render_blocknerf after all blocks have been trained
    if args.block_segments:
        args = Args(
            model=args.model,
            input_data_dir=args.input_data_dir,
            output_dir=args.output_dir,
            block_segments=args.block_segments,
        )
        experiment_name = "-".join(str(args.input_data_dir).split("/")[-2:])
        pipeline = ExperimentPipeline(args, writer, experiment_name)
        pipeline.render_blocknerf(block_paths=block_paths)

    terminal.close()
