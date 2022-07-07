"""Simple yaml debugger"""
import subprocess
import sys

import yaml

LOCAL_TESTS = ["Run Black", "Python Pylint", "Test with pytest"]


def run_command(command):
    """Run a command kill actions if it fails"""
    ret_code = subprocess.call(command, shell=True)
    if ret_code != 0:
        print(f"\033[31mError: `{curr_command}` failed. Exiting...\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    with open(".github/workflows/code_checks.yml", "rb") as f:
        my_dict = yaml.safe_load(f)
    steps = my_dict["jobs"]["build"]["steps"]

    print_green = lambda x: print(f"\033[32m{x}\033[0m")

    for step in steps:
        if "name" in step and step["name"] in LOCAL_TESTS:
            compressed = step["run"].replace("\n", ";").replace("\\", "")
            curr_command = f"{compressed}"

            print("*" * 100)
            print_green(f"Running: {curr_command}")
            run_command(curr_command)
        else:
            print("*" * 100)
            print(f"Skipping {step}")

    # Add checks for building documentation
    print_green("Building Documentation")
    run_command("cd docs/; make html; cd ../")

    # Add licensing to all pyrad files
    print_green("Add licensing headers")
    run_command("./scripts/licensing/license_headers.sh")

    print("\n")
    print_green("=" * 100)
    print_green("ALL CHECKS PASSED")
    print_green("=" * 100)
