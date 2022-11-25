#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

import tyro
from typing_extensions import Literal

from nerfstudio.utils.scripts import run_command


@dataclass
class Args:
    model: str
    data_source: Literal["video", "images", "polycam"]
    input_data_dir: Path
    output_dir: Path
    percent_frames_list: list = [1.0]
    

if __name__ == "main":
    args = tyro.cli(Args)
    print(args)

    for percent in args.percent_frames_list:
        cmd = f"./nerf_pipeline.py --model {args.model} --data_source {args.data_source} --input_data {args.input_data_dir} --output_dir {args.output_dir}/{str(percent * 100)} --percent_frames {percent}"
        run_command(cmd)