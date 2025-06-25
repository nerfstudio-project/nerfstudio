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
import subprocess
import sys
from typing import Literal


def run_command_with_console_output(command, stop_on_output=None):
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )

        # Read and print output in real-time
        for line in process.stdout:
            print(line, end="")
            sys.stdout.flush()  # Ensure output is printed immediately

            if stop_on_output and stop_on_output in line:
                print(f"\nDetected '{stop_on_output}'. Stopping the process.")
                process.terminate()
                break

        return_code = process.wait()
        if return_code != 0:
            print(f"Command failed with return code {return_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def run_ns_download_data(scene: Literal["poster", "dozer", "desolation"]):
    command = f"ns-download-data nerfstudio --capture-name={scene}"
    run_command_with_console_output(command)


def run_ns_train_nerfacto(scene: Literal["poster", "dozer", "desolation"]):
    dataset_path = f"data/nerfstudio/{scene}"
    command = f"ns-train nerfacto --data {dataset_path}"
    run_command_with_console_output(command, stop_on_output="Checkpoint Directory")


def run_ns_eval(scene: Literal["poster", "dozer", "desolation"]):
    timestamp = sorted(os.listdir(f"outputs/{scene}/nerfacto/"))[-1]
    config_filename = f"outputs/{scene}/nerfacto/{timestamp}/config.yml"
    command = f"ns-eval --load-config {config_filename} --output-path nerfacto_integration_eval.json"
    run_command_with_console_output(command)

    with open("nerfacto_integration_eval.json", "r") as f:
        results = json.load(f)

    assert results["results"]["psnr"] > 20.0, "PSNR was lower than 20"
    assert results["results"]["ssim"] > 0.7, "SSIM was lower than 0.7"


def main():
    scene = "dozer"  # You can change this to "poster" or "desolation"

    print("Starting data download...")
    run_ns_download_data(scene)

    print("\nStarting training...")
    run_ns_train_nerfacto(scene)

    print("\nStarting evaluation...")
    run_ns_eval(scene)


if __name__ == "__main__":
    main()
