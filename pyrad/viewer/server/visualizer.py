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

import numpy as np
import umsgpack
import zmq

from .commands import Delete, SetObject, GetObject, SetImage, SetProperty, SetTransform
from .path import Path


class ViewerWindow(object):
    context = zmq.Context()

    def __init__(self, zmq_url):
        self.zmq_url = zmq_url
        self.client = self.context.socket(zmq.REQ)
        self.client.connect(self.zmq_url)

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


class Viewer(object):
    """Visualizer class for connecting to the bridge server."""

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
        return self.window.send(SetObject(geometry, material, self.path))

    def get_object(self):
        return self.window.send(GetObject(self.path))

    def set_image(self, image):
        return self.window.send(SetImage(image, self.path))

    def set_transform(self, matrix=np.eye(4)):
        assert matrix.shape == (4, 4)
        return self.window.send(SetTransform(matrix, self.path))

    def set_property(self, key, value):
        return self.window.send(SetProperty(key, value, self.path))

    def delete(self):
        return self.window.send(Delete(self.path))

    def __repr__(self):
        return "<Viewer using: {window} at path: {path}>".format(window=self.window, path=self.path)
