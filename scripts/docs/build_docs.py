"""Simple yaml debugger"""
import subprocess
import sys

import dcargs

LOCAL_TESTS = ["Run license checks", "Run Black", "Python Pylint", "Test with pytest"]


def run_command(command: str) -> None:
    """Run a command kill actions if it fails

    Args:
        command: command to run
    """
    ret_code = subprocess.call(command, shell=True)
    if ret_code != 0:
        print(f"\033[31mError: `{command}` failed. Exiting...\033[0m")
        sys.exit(1)


def main():
    """Run the github actions locally."""

    print_green = lambda x: print(f"\033[32m{x}\033[0m")

    print_green("Adding notebook documentation metadata")
    run_command("python scripts/docs/add_nb_tags.py")

    # Add checks for building documentation
    print_green("Building Documentation")
    run_command("cd docs/; make html SPHINXOPTS='-W;'")

    print("\n")
    print_green("=" * 100)
    print_green("Done")
    print_green("=" * 100)


if __name__ == "__main__":
    dcargs.cli(main)
