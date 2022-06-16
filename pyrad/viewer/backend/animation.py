# Copyright 2022 The Plenoptix Team. All rights reserved.
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

from __future__ import absolute_import, division, print_function

import tempfile
import tarfile
import sys
import os.path
import subprocess

if sys.version_info >= (3, 0):
    unicode = str

import bisect
from . import transformations as tf


class AnimationTrack(object):
    __slots__ = ["name", "jstype", "frames", "values"]

    def __init__(self, name, jstype, frames=None, values=None):
        self.name = name
        self.jstype = jstype
        if frames is None:
            self.frames = []
        else:
            self.frames = frames
        if values is None:
            self.values = []
        else:
            self.values = values

    def set_property(self, frame, value):
        i = bisect.bisect(self.frames, frame)
        self.frames.insert(i, frame)
        self.values.insert(i, value)

    def lower(self):
        return {
            "name": unicode("." + self.name),
            "type": unicode(self.jstype),
            "keys": [{"time": self.frames[i], "value": self.values[i]} for i in range(len(self.frames))],
        }


class AnimationClip(object):
    __slots__ = ["tracks", "fps", "name"]

    def __init__(self, tracks=None, fps=30, name="default"):
        if tracks is None:
            self.tracks = {}
        else:
            self.tracks = tracks
        self.fps = fps
        self.name = name

    def set_property(self, frame, property, jstype, value):
        if property not in self.tracks:
            self.tracks[property] = AnimationTrack(property, jstype)
        track = self.tracks[property]
        track.set_property(frame, value)

    def lower(self):
        return {"fps": self.fps, "name": unicode(self.name), "tracks": [t.lower() for t in self.tracks.values()]}


class Animation(object):
    __slots__ = ["clips", "default_framerate"]

    def __init__(self, clips=None, default_framerate=30):
        if clips is None:
            self.clips = {}
        else:
            self.clips = clips
        self.default_framerate = default_framerate

    def lower(self):
        return [{"path": path.lower(), "clip": clip.lower()} for (path, clip) in self.clips.items()]

    def at_frame(self, visualizer, frame):
        return AnimationFrameVisualizer(self, visualizer.path, frame)


def js_position(matrix):
    return list(matrix[:3, 3])


def js_quaternion(matrix):
    quat = tf.quaternion_from_matrix(matrix)
    return [quat[1], quat[2], quat[3], quat[0]]


class AnimationFrameVisualizer(object):
    __slots__ = ["animation", "path", "current_frame"]

    def __init__(self, animation, path, current_frame):
        self.animation = animation
        self.path = path
        self.current_frame = current_frame

    def get_clip(self):
        if self.path not in self.animation.clips:
            self.animation.clips[self.path] = AnimationClip(fps=self.animation.default_framerate)
        return self.animation.clips[self.path]

    def set_transform(self, matrix):
        assert matrix.shape == (4, 4)
        clip = self.get_clip()
        clip.set_property(self.current_frame, "position", "vector3", js_position(matrix))
        clip.set_property(self.current_frame, "quaternion", "quaternion", js_quaternion(matrix))

    def set_property(self, prop, jstype, value):
        clip = self.get_clip()
        clip.set_property(self.current_frame, prop, jstype, value)

    def __getitem__(self, path):
        return AnimationFrameVisualizer(self.animation, self.path.append(path), self.current_frame)

    def __enter__(self):
        return self

    def __exit__(self, *arg):
        pass


def convert_frames_to_video(tar_file_path, output_path="output.mp4", framerate=60, overwrite=False):
    """
    Try to convert a tar file containing a sequence of frames saved by the
    pyrad viewer into a single video file.

    This relies on having `ffmpeg` installed on your system.
    """
    output_path = os.path.abspath(output_path)
    if os.path.isfile(output_path) and not overwrite:
        raise ValueError(
            "The output path {:s} already exists. To overwrite that file, you can pass overwrite=True to this function.".format(
                output_path
            )
        )
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(tar_file_path) as tar:
            tar.extractall(tmp_dir)
        args = [
            "ffmpeg",
            "-r",
            str(framerate),
            "-i",
            r"%07d.png",
            "-vcodec",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
        ]
        if overwrite:
            args.append("-y")
        args.append(output_path)
        try:
            subprocess.check_call(args, cwd=tmp_dir)
        except subprocess.CalledProcessError as e:
            print(
                """
Could not call `ffmpeg` to convert your frames into a video.
If you want to convert the frames manually, you can extract the
.tar archive into a directory, cd to that directory, and run:
ffmpeg -r 60 -i %07d.png \\\n\t -vcodec libx264 \\\n\t -preset slow \\\n\t -crf 18 \\\n\t output.mp4
                """
            )
            raise
    print("Saved output as {:s}".format(output_path))
    return output_path
