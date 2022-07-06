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

import asyncio
import queue
import math

import cv2
import numpy
from aiortc import VideoStreamTrack
from av import VideoFrame


class FlagVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self):
        super().__init__()  # don't forget this!
        self.counter = 0
        height, width = 480, 640

        # generate flag
        data_bgr = numpy.hstack(
            [
                self._create_rectangle(width=213, height=480, color=(255, 0, 0)),  # blue
                self._create_rectangle(width=214, height=480, color=(255, 255, 255)),  # white
                self._create_rectangle(width=213, height=480, color=(0, 0, 255)),  # red
            ]
        )

        # shrink and center it
        M = numpy.float32([[0.5, 0, width / 4], [0, 0.5, height / 4]])
        data_bgr = cv2.warpAffine(data_bgr, M, (width, height))

        # compute animation
        omega = 2 * math.pi / height
        id_x = numpy.tile(numpy.array(range(width), dtype=numpy.float32), (height, 1))
        id_y = numpy.tile(numpy.array(range(height), dtype=numpy.float32), (width, 1)).transpose()

        self.frames = []
        for k in range(30):
            phase = 2 * k * math.pi / 30
            map_x = id_x + 10 * numpy.cos(omega * id_x + phase)
            map_y = id_y + 10 * numpy.sin(omega * id_x + phase)
            frame = cv2.remap(data_bgr, map_x, map_y, cv2.INTER_LINEAR)
            self.frames.append(VideoFrame.from_ndarray(frame, format="bgr24"))

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = self.frames[self.counter % 30]
        frame.pts = pts
        frame.time_base = time_base
        self.counter += 1
        return frame

    def _create_rectangle(self, width, height, color):
        data_bgr = numpy.zeros((height, width, 3), numpy.uint8)
        data_bgr[:, :] = color
        return data_bgr


class SingleFrameStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.background_frame = numpy.zeros((480, 640, 3), dtype="uint8") * 255
        self.frame = None
        self.put_frame(self.background_frame)

    def put_frame(self, frame):
        self.frame = VideoFrame.from_ndarray(frame)

    async def recv(self):

        pts, time_base = await self.next_timestamp()

        frame = self.frame
        frame.pts = pts
        frame.time_base = time_base
        return frame
