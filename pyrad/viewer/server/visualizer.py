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

import logging
import signal
import sys

import msgpack
import msgpack_numpy
import numpy as np
import umsgpack
import zmq

from pyrad.viewer.server.commands import (
    Delete,
    GetObject,
    SetObject,
    SetOutputOptions,
    SetProperty,
    SetTransform,
)
from pyrad.viewer.server.path import Path


class ViewerWindow:
    context = zmq.Context()

    def __init__(self, zmq_url):
        self.zmq_url = zmq_url
        self.client = self.context.socket(zmq.REQ)
        self.client.connect(self.zmq_url)
        self.assert_connected()

    def send(self, command):
        cmd_data = command.lower()
        self.client.send_multipart(
            [
                cmd_data["type"].encode("utf-8"),
                cmd_data["path"].encode("utf-8"),
                umsgpack.packb(cmd_data),
            ]
        )
        return self.client.recv()

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
        return self.client.recv()

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
        except Exception as e:
            logging.info(e)
            sys.exit()


class Viewer:
    """Visualizer class for connecting to the bridge server.

    Args:
        zmq_url (str, optional): _description_. Defaults to None.
        window (ViewerWindow, optional): _description_. Defaults to None.
        timeout (str, optional): _description_. Defaults to None.
    """

    def __init__(self, zmq_url: str = None, window: ViewerWindow = None):
        if zmq_url is None and window is None:
            raise ValueError("Must specify either zmq_url or window.")
        if window is None:
            self.window = ViewerWindow(zmq_url=zmq_url)
        else:
            self.window = window
        self.path = Path(("pyrad",))

    @staticmethod
    def view_into(window: ViewerWindow, path: Path):
        """Returns a new Viewer but keeping the same ViewerWindow."""
        vis = Viewer(window=window)
        vis.path = path
        return vis

    def __getitem__(self, path):
        return Viewer.view_into(self.window, self.path.append(path))

    def set_object(self, geometry, material=None):
        """Set the object at the current path
        Args:
            geometry (_type_): _description_
            material (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self.window.send(SetObject(geometry, material, self.path))

    def get_object(self):
        """Get the object at the current path."""
        data = self.window.send(GetObject(self.path))
        data = umsgpack.unpackb(data)
        if isinstance(data, str) and data.find("error") == 0:
            # some error meaning that the object does not exist
            return None
        return data

    def set_image(self, image):
        """Set the image"""
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

    def set_transform(self, matrix=np.eye(4)):
        """Set the transform"""
        assert matrix.shape == (4, 4)
        return self.window.send(SetTransform(matrix, self.path))

    def set_property(self, key, value):
        """Set the property"""
        return self.window.send(SetProperty(key, value, self.path))

    def set_output_options(self, options):
        """Set the output options"""
        return self.window.send(SetOutputOptions(options, self.path))

    def delete(self):
        """Delete the contents of the window"""
        return self.window.send(Delete(self.path))

    def __repr__(self):
        return f"<Viewer using: {self.window} at path: {self.path}>"
