from __future__ import annotations

import os

from nerfstudio.nerfstudio.utils.checkpoint_loader import find_latest_checkpoint

os.environ["TERM"] = "dumb"
import argparse
import glob
import json
from copy import deepcopy
from pathlib import Path

import scripts.train as train
import tyro
from scripts.my_utils import *

from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion

SEGMENTATION_DIRECTORY_DELIMITER = ":"


def dynamically_override_config(config):
    timestamp = get_timestamp()
    data_name = Path(config.data).name
    experiment_name = get_experiment_name(timestamp=timestamp)
    config.experiment_name = f"{experiment_name}-{data_name}-{config.method_name}"
    config.relative_model_dir = "."  # don't save to nerfstudio_models
    return config


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--models", nargs="+")
    args.add_argument("--datasets", nargs="+")
    args.add_argument("--seg-iter", type=int, required=False)
    args.add_argument("--seg-auto", type=bool, default=False)
    args.add_argument("--train-test-split", type=float, default=0.5)

    return args.parse_known_args()


def split_to_model_and_data(
    nerfstudio_args, num_models, num_datasets, dataset_at_index
) -> tuple[list[str], list[str]]:
    left = dataset_at_index - 1 - 1 - num_models
    right = left + num_datasets - 1

    return nerfstudio_args[:left], nerfstudio_args[right:]


def train_and_time(config):
    time_start = datetime.now()
    train.main(config)
    time_end = datetime.now()
    time_total_in_minutes = (time_end - time_start).total_seconds() / 60.0
    return time_total_in_minutes


def stdout_to_log(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_to_file(Path(log_dir, "log.txt"))
    return


def main():

    script_args, nerfstudio_args = parse_args()

    train_test_split = script_args.train_test_split
    # TODO: split
    model_args, data_args = split_to_model_and_data(
        nerfstudio_args,
        len(script_args.models),
        len(script_args.datasets),
        sys.argv.index("--datasets"),
    )

    for model in script_args.models:
        for dataset_and_segmentations in script_args.datasets:

            dataset_and_segmentations = dataset_and_segmentations.split(
                SEGMENTATION_DIRECTORY_DELIMITER
            )
            dataset_path = dataset_and_segmentations[0]
            segmentation_sequences = dataset_and_segmentations[1:]

            timestamp = get_timestamp()
            data_name = Path(dataset_path).name
            experiment_name = get_experiment_name(timestamp=timestamp)

            # Build args
            prepare_args = []
            prepare_args.insert(0, model)
            prepare_args.extend(model_args)
            prepare_args.extend(["--data", dataset_path])
            prepare_args.extend(data_args)

            # Parse and adjust
            config = tyro.cli(
                deepcopy(AnnotatedBaseConfigUnion),
                args=prepare_args,
            )
                        
            config.experiment_name = (
                f"{experiment_name}-{data_name}-{config.method_name}"
            )
            config.relative_model_dir = Path(".")  # don't save to nerfstudio_models

            # Log to file
            # rgb_out_dir = Path(config.output_dir, config.experiment_name)  # type: ignore
            rgb_out_dir = config.get_base_dir()
            stdout_to_log(rgb_out_dir)

            base_config = deepcopy(config)
            time_total_in_minutes = train_and_time(config)

            stats = {"train_time_total_in_minutes": time_total_in_minutes}
            with open(Path(rgb_out_dir, "stats.json"), "w") as fp:
                json.dump(stats, fp)

            if script_args.seg_auto:
                segmentation_sequences = list(dataset_path.glob("*mask*"))

            # Train segmentations
            for seg_path in segmentation_sequences:

                reset_sockets()

                config = tyro.cli(
                    deepcopy(AnnotatedBaseConfigUnion),
                    args=prepare_args,
                )

                config.data = Path(seg_path)
                data_name = Path(seg_path).name
                print("segpat", seg_path)
                print("data_name", data_name)
                config.experiment_name += f"_{data_name}"  # type: ignore # append segmnetation name
                config.relative_model_dir = Path(".")  # don't save to nerfstudio_models

                baseline_ckpt_path = find_latest_checkpoint(rgb_out_dir)
                config.load_ckpt = baseline_ckpt_path

                if script_args.seg_iter:
                    config.max_num_iterations = script_args.seg_iter

                out_dir = config.get_base_dir()
                stdout_to_log(out_dir)

                time_total_in_minutes = train_and_time(config)
                stats = {"train_time_total_in_minutes": time_total_in_minutes}
                with open(Path(out_dir, "stats.json"), "w") as fp:
                    json.dump(stats, fp)


if __name__ == "__main__":
    main()
