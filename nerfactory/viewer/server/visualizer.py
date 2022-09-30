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

"""Code to connect and send commands to the viewer.
"""

import logging
import signal
import sys
from typing import Dict, Optional, Union

import msgpack
import msgpack_numpy
import umsgpack
import zmq

from nerfactory.viewer.server.path import Path


class ViewerWindow:
    """The viewer window has the ZMQ connection to the viewer bridge server."""

    context = zmq.Context()

    def __init__(self, zmq_port):
        self.zmq_port = zmq_port
        self.client = self.context.socket(zmq.REQ)
        zmq_url = f"tcp://127.0.0.1:{self.zmq_port}"
        self.client.connect(zmq_url)
        self.assert_connected()

    def send(self, command):
        """Sends a command to the viewer bridge server."""
        self.client.send_multipart(
            [
                command["type"].encode("utf-8"),
                command["path"].encode("utf-8"),
                umsgpack.packb(command),
            ]
        )
        return umsgpack.unpackb(self.client.recv())

    def send_ping(self):
        """Tries to contact the viewer bridge server."""
        type_ = "ping"
        path = ""
        data = umsgpack.packb({"type": type_, "path": path})
        self.client.send_multipart(
            [
                type_.encode("utf-8"),
                path.encode("utf-8"),
                data,
            ]
        )
        return umsgpack.unpackb(self.client.recv())

    def assert_connected(self, timeout_in_sec: int = 5):
        """Check if the connection was established properly within some time.

        Args:
            timeout_in_sec (int): The maximum time to wait for the connection to be established.
        """

        def timeout_handler(signum, frame):
            raise Exception(f"Couldn't connect to the viewer Bridge Server in {timeout_in_sec} seconds. Exiting.")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_in_sec)
        try:
            logging.info("Sending ping to the viewer Bridge Server...")
            _ = self.send_ping()
            logging.info("Successfully connected.")
            signal.alarm(0)  # cancel the alarm
        except Exception as e:  # pylint: disable=broad-except
            logging.info(e)
            sys.exit()


class Viewer:
    """Viewer class for connecting to the bridge server.

    Args:
        zmq_port: Where to connect with ZMQ.
        window: An already existing ViewerWindow.
    """

    def __init__(self, zmq_port: Optional[int] = None, window: Optional[ViewerWindow] = None):
        if zmq_port is None and window is None:
            raise ValueError("Must specify either zmq_port or window.")
        if window is None:
            self.window = ViewerWindow(zmq_port=zmq_port)
        else:
            self.window = window
        self.path = Path(())

    @staticmethod
    def view_into(window: ViewerWindow, path: Path):
        """Returns a new Viewer but keeping the same ViewerWindow."""
        vis = Viewer(window=window)
        vis.path = path
        return vis

    def __getitem__(self, path):
        return Viewer.view_into(self.window, self.path.append(path))

    def __repr__(self):
        return f"<Viewer using: {self.window} at path: {self.path}>"

    def write(self, data: Union[Dict, str, None] = None):
        """Write data."""
        path = self.path.lower()
        return self.window.send({"type": "write", "path": path, "data": data})

    def read(self):
        """Read data."""
        path = self.path.lower()
        return self.window.send({"type": "read", "path": path})

    def delete(self):
        """Delete data."""
        return self.write(data=None)

    def set_image(self, image):
        """Sends an image to the viewer with WebRTC."""
        type_ = "set_image"
        path = self.path.lower()
        data = msgpack.packb(image, default=msgpack_numpy.encode, use_bin_type=True)
        self.window.client.send_multipart(
            [
                type_.encode("utf-8"),
                path.encode("utf-8"),
                data,
            ]
        )
        return self.window.client.recv()
