from __future__ import annotations
import os

os.environ["TERM"] = "dumb"
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
import scripts.train as train
import tyro
from pathlib import Path
from scripts.my_utils import *
import tyro
from copy import deepcopy
import argparse
import json
import glob

OUTPUT_DIRECTORY = "outputs"
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
    args.add_argument("--seg_iter", nargs="+", default=30_000)
    args.add_argument("--seg_auto", default=False)

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


def main():

    script_args, nerfstudio_args = parse_args()

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
            dataset = dataset_and_segmentations[0]
            segmentation_sequences = dataset_and_segmentations[1:]

            timestamp = get_timestamp()
            data_name = Path(dataset).name
            experiment_name = get_experiment_name(timestamp=timestamp)

            # Build args
            prepare_args = []
            prepare_args.insert(0, model)
            prepare_args.extend(model_args)
            prepare_args.extend(["--data", dataset])
            prepare_args.extend(data_args)

            # Parse and adjust
            config = tyro.cli(
                deepcopy(AnnotatedBaseConfigUnion),
                args=prepare_args,
            )
            config.experiment_name = (
                f"{experiment_name}-{data_name}-{config.method_name}"
            )
            config.relative_model_dir = "."  # don't save to nerfstudio_models

            # Log to file
            rgb_out_dir = Path(config.output_dir, config.experiment_name)  # type: ignore
            rgb_out_dir.mkdir(parents=True, exist_ok=True)
            stdout_to_file(Path(rgb_out_dir, "log.txt"))

            time_total_in_minutes = train_and_time(config)

            stats = {"train_time_total_in_minutes": time_total_in_minutes}
            with open(Path(rgb_out_dir, "stats.json"), "w") as fp:
                json.dump(stats, fp)

            if script_args.seg_auto:
                # TODO: finish this
                for seg_mask in dataset.glob("*mask*"):
                    print(seg_mask)
                    segmentation_sequences.append(seg_mask)

            # Train segmentations
            for seg_path in segmentation_sequences:
                reset_sockets()

                config.data = Path(seg_path)
                data_name = Path(seg_path).name
                config.experiment_name += f"_{data_name}"  # append segmnetation name
                config.load_dir = rgb_out_dir
                config.max_num_iterations = script_args.seg_iter
                out_dir = Path(rgb_out_dir, data_name)
                out_dir.mkdir(parents=True, exist_ok=True)
                stdout_to_file(Path(out_dir, "log.txt"))

                time_total_in_minutes = train_and_time(config)

                stats = {"train_time_total_in_minutes": time_total_in_minutes}
                with open(Path(rgb_out_dir, "stats.json"), "w") as fp:
                    json.dump(stats, fp)


if __name__ == "__main__":
    main()
