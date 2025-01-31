# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Literal

from test_nerfacto_integration import run_command_with_console_output, run_ns_download_data


def run_ns_train_splatfacto(scene: Literal["poster", "dozer", "desolation"]):
    dataset_path = f"data/nerfstudio/{scene}"
    command = f"ns-train splatfacto --data {dataset_path}"
    run_command_with_console_output(command, stop_on_output="Checkpoint Directory")


def run_ns_eval(scene: Literal["poster", "dozer", "desolation"]):
    timestamp = sorted(os.listdir(f"outputs/{scene}/splatfacto/"))[-1]
    config_filename = f"outputs/{scene}/splatfacto/{timestamp}/config.yml"
    command = f"ns-eval --load-config {config_filename} --output-path splatfacto_integration_eval.json"
    run_command_with_console_output(command)

    with open("splatfacto_integration_eval.json", "r") as f:
        results = json.load(f)

    assert results["results"]["psnr"] > 20.0, "PSNR was lower than 20"
    assert results["results"]["ssim"] > 0.7, "SSIM was lower than 0.7"


def main():
    scene = "dozer"  # You can change this to "poster" or "desolation"

    print("Starting data download...")
    run_ns_download_data(scene)

    print("\nStarting training...")
    run_ns_train_splatfacto(scene)

    print("\nStarting evaluation...")
    run_ns_eval(scene)


if __name__ == "__main__":
    main()
