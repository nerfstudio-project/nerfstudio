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

"""Code to connect and send commands to the viewer.
"""

import sys
from threading import Thread
from typing import Dict, Optional, Union

import umsgpack
import zmq
from rich.console import Console

from nerfstudio.viewer.server.path import Path

CONSOLE = Console(width=120)


class ViewerWindow:
    """The viewer window has the ZMQ connection to the viewer bridge server.

    Args:
        zmq_port: Where to connect with ZMQ.
        ip_address: The ip address of the bridge server.
    """

    context = zmq.Context()  # pylint: disable=abstract-class-instantiated

    def __init__(self, zmq_port, ip_address="127.0.0.1"):
        self.zmq_port = zmq_port
        self.client = self.context.socket(zmq.REQ)
        zmq_url = f"tcp://{ip_address}:{self.zmq_port}"
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

    def timeout_ping(self, timeout_in_sec: int = 15):
        """Timeout if ping fails to complete in timeout_in_secs seconds"""

        res = [Exception(f"Couldn't connect to the viewer Bridge Server in {timeout_in_sec} seconds. Exiting.")]

        def wrapper_func():
            res[0] = self.send_ping()

        t = Thread(target=wrapper_func)
        t.daemon = True
        try:
            t.start()
            t.join(timeout_in_sec)
        except Exception as je:
            CONSOLE.log("Error starting thread")
            raise je
        ret = res[0]
        if isinstance(ret, BaseException):
            raise ret
        return ret

    def assert_connected(self, timeout_in_sec: int = 15):
        """Check if the connection was established properly within some time.

        Args:
            timeout_in_sec (int): The maximum time to wait for the connection to be established.
        """
        try:
            CONSOLE.print("Sending ping to the viewer Bridge Server...")
            _ = self.timeout_ping(timeout_in_sec)
            CONSOLE.print("Successfully connected.")

        except Exception as e:  # pylint: disable=broad-except
            CONSOLE.log(e)
            sys.exit()


class Viewer:
    """Viewer class for connecting to the bridge server.

    Args:
        zmq_port: Where to connect with ZMQ.
        window: An already existing ViewerWindow.
        ip_address: The ip address of the bridge server.
    """

    def __init__(
        self, zmq_port: Optional[int] = None, window: Optional[ViewerWindow] = None, ip_address: str = "127.0.0.1"
    ):
        if zmq_port is None and window is None:
            raise ValueError("Must specify either zmq_port or window.")
        if window is None:
            self.window = ViewerWindow(zmq_port=zmq_port, ip_address=ip_address)
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
