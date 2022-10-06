"""Helpers for checking if programs are installed"""

import shutil


def check_ffmpeg_installed():
    """Checks if ffmpeg is installed."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        CONSOLE.print("[bold red]Could not find ffmpeg. Please install ffmpeg.")
        print("See https://ffmpeg.org/download.html for installation instructions.")
        print("ffmpeg is only necessary if using videos as input.")
        sys.exit(1)


def check_colmap_installed():
    """Checks if colmap is installed."""
    colmap_path = shutil.which("colmap")
    if colmap_path is None:
        CONSOLE.print("[bold red]Could not find COLMAP. Please install COLMAP.")
        print("See https://colmap.github.io/install.html for installation instructions.")
        sys.exit(1)

def get_colmap_version(default_version=3.8) -> float:
    """Returns the version of COLMAP.
    This code assumes that colmap returns a version string of the form
    "COLMAP 3.8 ..." which may not be true for all versions of COLMAP.

    Args:
        default_version: Default version to return if COLMAP version can't be determined.
    Returns:
        The version of COLMAP.
    """
    output = run_command("colmap", verbose=False)
    assert output is not None
    for line in output.split("\n"):
        if line.startswith("COLMAP"):
            return float(line.split(" ")[1])
    CONSOLE.print(f"[bold red]Could not find COLMAP version. Using default {default_version}")
    return default_version
