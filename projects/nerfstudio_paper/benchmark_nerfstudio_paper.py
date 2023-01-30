"""
Benchmarking script for nerfstudio paper.

- nerfacto and instant-ngp methods on mipnerf360 data
- nerfacto ablations
"""

import threading
import time
from pathlib import Path
from typing import Union
from dataclasses import dataclass

import tyro
from typing_extensions import Annotated

import GPUtil

from nerfstudio.configs.base_config import PrintableConfig
from nerfstudio.utils.scripts import run_command

# for the mipnerf360 experiments
mipnerf360_capture_names = ["bicycle", "garden", "stump", "room", "counter", "kitchen", "bonsai"]  # 7 splits
mipnerf360_table_rows = [
    # nerfacto method
    (
        "nerfacto-w/o-pose-app",
        "nerfacto",
        "--pipeline.eval_optimize_cameras False --pipeline.eval_optimize_appearance False --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-appearance-embedding False nerfstudio-data --downscale-factor 4 --train-split-percentage 0.875 mipnerf360-data",
    ),
]

# for the ablation experiments
ablations_capture_names = [
    "Egypt",
    "person",
    "kitchen",
    "plane",
    "dozer",
    "floating-tree",
    "aspen",
    "stump",
    "sculpture",
    "Giannini-Hall",
]

ablations_table_rows = [
    ("nerfacto", "nerfacto", "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True"),
    (
        "w/o-pose",
        "nerfacto",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.datamanager.camera-optimizer.mode off",
    ),
    (
        "w/o-app",
        "nerfacto",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance False --pipeline.model.use-appearance-embedding False",
    ),
    (
        "w/o-pose-app",
        "nerfacto",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance False --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-appearance-embedding False",
    ),
    (
        "1-prop-network",
        "nerfacto",
        '--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.num-proposal-samples-per-ray "256" --pipeline.model.num_proposal_iterations 1',
    ),
    (
        "l2-contraction",
        "nerfacto",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.scene-contraction-norm l2",
    ),
    (
        "shared-prop-network",
        "nerfacto",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.use-same-proposal-network True",
    ),
    (
        "random-background-color",
        "nerfacto",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.background-color random",
    ),
    (
        "no-contraction",
        "nerfacto",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.use-bounded True --pipeline.model.use-scene-contraction False nerfstudio-data --scale_factor 0.125",
    ),
    (
        "synthetic-on-real",
        "nerfacto",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance False --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-appearance-embedding False --pipeline.model.use-bounded True --pipeline.model.use-scene-contraction False nerfstudio-data --scale_factor 0.125",
    ),
]


def launch_experiments(capture_names, table_rows, data_path: Path = Path("data/nerfstudio"), dry_run: bool = False):
    """Launch the experiments."""

    # make a list of all the jobs that need to be fun
    jobs = []
    for capture_name in capture_names:

        for table_row_name, method, table_row_command in table_rows:
            command = " ".join(
                (
                    f"ns-train {method}",
                    "--vis wandb",
                    f"--data { data_path / capture_name}",
                    "--output-dir outputs/nerfacto-ablations",
                    "--steps-per-eval-batch 0 --steps-per-eval-image 0",
                    "--steps-per-eval-all-images 5000 --max-num-iterations 30001",
                    f"--wandb-name {capture_name}_{table_row_name}",
                    f"--experiment-name {capture_name}_{table_row_name}",
                    # extra_string,
                    table_row_command,
                )
            )
            jobs.append(command)

    while jobs:
        # get GPUs that capacity to run these jobs
        gpu_devices_available = GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1)

        print("Available GPUs: ", gpu_devices_available)

        # thread list
        threads = []
        while gpu_devices_available and jobs:
            gpu = gpu_devices_available.pop(0)
            command = f"CUDA_VISIBLE_DEVICES={gpu} " + jobs.pop(0)

            def task():
                print("Starting command: ", command)
                if not dry_run:
                    _ = run_command(command, verbose=False)
                print("Finished command: ", command)

            threads.append(threading.Thread(target=task))
            threads[-1].start()

            # NOTE(ethan): here we need a delay, otherwise the wandb/tensorboard naming is messed up... not sure why
            if not dry_run:
                time.sleep(5)

        # wait for all threads to finish
        for t in threads:
            t.join()

        print("Finished all threads")


@dataclass
class Benchmark(PrintableConfig):
    """Benchmark code."""

    dry_run: bool = False

    def main(self) -> None:
        """Run the code."""
        raise NotImplementedError


@dataclass
class BenchmarkMipNeRF360(Benchmark):
    """Benchmark MipNeRF-360."""

    def main(self, dry_run: bool = False):
        launch_experiments(
            mipnerf360_capture_names, mipnerf360_table_rows, data_path=Path("data/nerfstudio-data-mipnerf360", dry_run=dry_run)
        )


@dataclass
class BenchmarkAblations(Benchmark):
    """Benchmark ablations."""

    def main(self, dry_run: bool = False):
        launch_experiments(ablations_capture_names, ablations_table_rows, dry_run=dry_run)


Commands = Union[
    Annotated[BenchmarkMipNeRF360, tyro.conf.subcommand(name="nerfacto-on-mipnerf360")],
    Annotated[BenchmarkAblations, tyro.conf.subcommand(name="nerfacto-ablations")],
]


def main(
    benchmark: Benchmark,
):
    """Script to run the benchmark experiments for the Nerfstudio paper.
    - nerfacto-on-mipnerf360: The MipNeRF-360 experiments on the MipNeRF-360 Dataset.
    - nerfacto-ablations: The Nerfacto ablations on the Nerfstudio Dataset.

    Args:
        benchmark: The benchmark to run.
    """

    benchmark.main(dry_run=benchmark.dry_run)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
