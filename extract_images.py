#!/usr/bin/env python

import os
from pathlib import Path

from nerfstudio.utils.scripts import run_command

if __name__ == "__main__":
    root_path = Path(os.curdir)
    images_dir = root_path / "images"
    videos_dir = root_path / "videos"
    images_dir.mkdir(exist_ok=True, parents=True)
    num_images = len(os.listdir(images_dir))

    for video in os.listdir(videos_dir):
        video_path = videos_dir / video
        cmd = f"ffmpeg -i {video_path} -start_number {num_images} -vf thumbnail=4,setpts=N/TB -r 1 {images_dir}/frame_%05d.png"
        run_command(cmd, verbose=True)
        num_images = len(os.listdir(images_dir))
