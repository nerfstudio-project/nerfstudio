"""Simple yaml debugger"""
import subprocess

import yaml

LOCAL_TESTS = ["Run Black", "Python Pylint", "Test with pytest"]

if __name__ == "__main__":
    with open(".github/workflows/code_checks.yml", "rb") as f:
        my_dict = yaml.safe_load(f)
    steps = my_dict["jobs"]["build"]["steps"]

    for step in steps:
        if "name" in step and step["name"] in LOCAL_TESTS:
            compressed = step["run"].replace("\n", ";").replace("\\", "")
            curr_command = f"{compressed}"

            print("*" * 100)
            print(f"Running {curr_command}")
            subprocess.call(curr_command, shell=True)
        else:
            print("*" * 100)
            print(f"Skipping {step}")
