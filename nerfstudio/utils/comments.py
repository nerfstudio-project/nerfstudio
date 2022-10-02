"""A collection of common strings and print statements used throughout the codebase."""

from rich.console import Console

console = Console(width=120)


def print_tcnn_speed_warning():
    """Prints a warning about the speed of the TCNN."""
    console.line()
    console.print(
        "[bold yellow]WARNING: :person_running: :person_running: "
        + "Install tcnn for speedups :person_running: :person_running:"
    )
    console.print("[yellow]pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")
    console.line()
