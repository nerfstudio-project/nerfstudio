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

import sys
import time

import numpy as np
import umsgpack
import zmq

from .commands import Delete, SetAnimation, SetImage, SetObject, SetProperty, SetTransform
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
        self.client.recv()


class Visualizer(object):
    __slots__ = ["window", "path"]

    def __init__(self, window):
        self.window = window
        self.path = Path(("pyrad",))  # TODO(ethan): change this

    @staticmethod
    def view_into(window, path):
        vis = Visualizer(window=window)
        vis.path = path
        return vis

    def __getitem__(self, path):
        return Visualizer.view_into(self.window, self.path.append(path))

    def set_object(self, geometry, material=None):
        return self.window.send(SetObject(geometry, material, self.path))

    def set_transform(self, matrix=np.eye(4)):
        assert matrix.shape == (4, 4)
        return self.window.send(SetTransform(matrix, self.path))

    def set_property(self, key, value):
        return self.window.send(SetProperty(key, value, self.path))

    def set_animation(self, animation, play=True, repetitions=1):
        return self.window.send(SetAnimation(animation, play=play, repetitions=repetitions))

    def set_image(self, image):
        return self.window.send(SetImage(image, self.path))

    def delete(self):
        return self.window.send(Delete(self.path))

    def __repr__(self):
        return "<Visualizer using: {window} at path: {path}>".format(window=self.window, path=self.path)
