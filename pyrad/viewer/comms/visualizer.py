from __future__ import absolute_import, division, print_function

import sys
import webbrowser
import socketio

import umsgpack
import numpy as np
import zmq
from IPython.display import HTML

from .path import Path
from .commands import SetObject, SetTransform, Delete, SetProperty, SetAnimation, SetCamera
from .geometry import MeshPhongMaterial


class ViewerWindow(object):
    context = zmq.Context()

    def __init__(self, zmq_url="tcp://0.0.0.0:6000"):
        # self.client = socketio.Client()
        # self.client.connect('https://recon.ethanweber.me')
        self.zmq_url = zmq_url
        self.client = self.context.socket(zmq.REQ)
        self.client.connect(self.zmq_url)

    def send(self, command):
        cmd_data = command.lower()
        self.client.send_multipart([
            cmd_data["type"].encode("utf-8"),
            cmd_data["path"].encode("utf-8"),
            umsgpack.packb(cmd_data)
        ])
        self.client.recv()


class Visualizer(object):
    __slots__ = ["window", "path"]

    def __init__(self,
                 window=None):
        if window is None:
            self.window = ViewerWindow()
        else:
            self.window = window
        self.path = Path(("meshcat",))
        # self.path = Path(())

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

    def set_camera(self, path):
        return self.window.send(SetCamera(self.path))

    def delete(self):
        return self.window.send(Delete(self.path))

    def __repr__(self):
        return "<Visualizer using: {window} at path: {path}>".format(window=self.window, path=self.path)


if __name__ == '__main__':
    import time
    import sys

    args = []
    if len(sys.argv) > 1:
        zmq_url = sys.argv[1]
        if len(sys.argv) > 2:
            args = sys.argv[2:]
    else:
        zmq_url = None

    window = ViewerWindow(zmq_url, zmq_url is None, True, args)

    while True:
        time.sleep(100)
