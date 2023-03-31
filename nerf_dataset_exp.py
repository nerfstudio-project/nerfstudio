#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import tyro
from typing_extensions import Literal

from nerfstudio.utils.scripts import run_command


@dataclass
class Args:
    model: str
    data_source: Literal["video", "images", "polycam"]
    input_data_dir: Path
    output_dir: Path
    percent_frames_list: str
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args)
    
    percent_frames_list = [float(x) for x in args.percent_frames_list.split(",")]
    for percent in percent_frames_list:
        cmd = f"./nerf_pipeline.py --model {args.model} --data_source {args.data_source} --input_data {args.input_data_dir} --output_dir {args.output_dir}/{str(int(percent * 100))} --percent_frames {percent}"
        print(cmd)
        start = timer()
        run_command(cmd, verbose=True)
        end = timer()
        print(f"[time] finish {percent}: {end-start}\n")
