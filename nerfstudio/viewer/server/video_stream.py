# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Video Stream objects for WebRTC"""

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame


class SingleFrameStreamTrack(VideoStreamTrack):
    """Single Frame stream class: pushes single frames to a stream"""

    def __init__(self):
        super().__init__()
        self.background_frame = np.ones((480, 640, 3), dtype="uint8") * 100  # gray background
        self.frame = None
        self.put_frame(self.background_frame)

    def put_frame(self, frame: np.ndarray) -> None:
        """Sets the current viewing frame

        Args:
            frame: image to be viewed
        """
        self.frame = VideoFrame.from_ndarray(frame)

    async def recv(self):
        """Async method to grab and wait on frame"""
        pts, time_base = await self.next_timestamp()

        frame = self.frame
        frame.pts = pts
        frame.time_base = time_base
        return frame
