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

import base64
import re
import sys

if sys.version_info >= (3, 0):
    ADDRESS_IN_USE_ERROR = OSError
else:
    import socket

    ADDRESS_IN_USE_ERROR = socket.error

import tornado.gen
import tornado.ioloop
import tornado.web
import tornado.websocket
import zmq
import zmq.eventloop.ioloop
from pyrad.viewer.backend.tree import SceneTree, find_node, walk
from zmq.eventloop.zmqstream import ZMQStream


def capture(pattern, s):
    match = re.match(pattern, s)
    if not match:
        raise ValueError("Could not match {:s} with pattern {:s}".format(s, pattern))
    else:
        return match.groups()[0]


def match_zmq_url(line):
    return capture(r"^zmq_url=(.*)$", line)


def _zmq_install_ioloop():
    # For pyzmq<17, install ioloop instead of a tornado ioloop
    # http://zeromq.github.com/pyzmq/eventloop.html
    try:
        pyzmq_major = int(zmq.__version__.split(".")[0])
    except ValueError:
        # Development version?
        return
    if pyzmq_major < 17:
        zmq.eventloop.ioloop.install()


_zmq_install_ioloop()


MAX_ATTEMPTS = 1000
DEFAULT_ZMQ_METHOD = "tcp"
DEFAULT_ZMQ_PORT = 6000
DEFAULT_WEBSOCKET_PORT = 8051
MESHCAT_COMMANDS = ["set_transform", "set_object", "delete", "set_property", "set_animation", "set_image"]


def find_available_port(func, default_port, max_attempts=MAX_ATTEMPTS, **kwargs):
    for i in range(max_attempts):
        port = default_port + i
        try:
            return func(port, **kwargs), port
        except (ADDRESS_IN_USE_ERROR, zmq.error.ZMQError):
            print("Port: {:d} in use, trying another...".format(port), file=sys.stderr)
        except Exception as e:
            print(type(e))
            raise
    else:
        raise (
            Exception(
                "Could not find an available port in the range: [{:d}, {:d})".format(
                    default_port, max_attempts + default_port
                )
            )
        )


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        self.bridge = kwargs.pop("bridge")
        super(WebSocketHandler, self).__init__(*args, **kwargs)

    # this disables CORS
    def check_origin(self, origin):
        return True

    def open(self):
        self.bridge.websocket_pool.add(self)
        print("opened:", self, file=sys.stderr)
        self.bridge.send_scene(self)

    def on_message(self, message):
        print(message)

    def on_close(self):
        self.bridge.websocket_pool.remove(self)
        print("closed:", self, file=sys.stderr)


def create_command(data):
    """Encode the drawing command into a Javascript fetch() command for display."""
    return """
fetch("data:application/octet-binary;base64,{}")
    .then(res => res.arrayBuffer())
    .then(buffer => viewer.handle_command_bytearray(new Uint8Array(buffer)));
    """.format(
        base64.b64encode(data).decode("utf-8")
    )


class ZMQWebSocketBridge(object):
    context = zmq.Context()

    def __init__(self, zmq_url=None, host="127.0.0.1", websocket_port=None):
        self.host = host
        self.websocket_pool = set()
        self.app = self.make_app()
        self.ioloop = tornado.ioloop.IOLoop.current()

        if zmq_url is None:

            def f(port):
                return self.setup_zmq("{:s}://{:s}:{:d}".format(DEFAULT_ZMQ_METHOD, self.host, port))

            (self.zmq_socket, self.zmq_stream, self.zmq_url), _ = find_available_port(f, DEFAULT_ZMQ_PORT)
        else:
            self.zmq_socket, self.zmq_stream, self.zmq_url = self.setup_zmq(zmq_url)

        listen_kwargs = {}

        if websocket_port is None:
            _, self.websocket_port = find_available_port(self.app.listen, DEFAULT_WEBSOCKET_PORT, **listen_kwargs)
        else:
            self.app.listen(websocket_port, **listen_kwargs)
            self.websocket_port = websocket_port

        self.tree = SceneTree()

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name} using zmq_url={self.zmq_url} and websocket_port={self.websocket_port}"

    def make_app(self):
        return tornado.web.Application([(r"/", WebSocketHandler, {"bridge": self})])

    def handle_zmq(self, frames):
        cmd = frames[0].decode("utf-8")
        if cmd not in MESHCAT_COMMANDS:
            print(f"{cmd} not in MESHCAT_COMMANDS! Proceeding anyways.")
            # self.zmq_socket.send(b"error: unrecognized comand")
        if len(frames) != 3:
            self.zmq_socket.send(b"error: expected 3 frames")
            return
        path = list(filter(lambda x: len(x) > 0, frames[1].decode("utf-8").split("/")))
        data = frames[2]
        self.forward_to_websockets(frames)
        if cmd == "set_transform":
            find_node(self.tree, path).transform = data
        elif cmd == "set_object":
            find_node(self.tree, path).object = data
            find_node(self.tree, path).properties = []
        elif cmd == "set_property":
            find_node(self.tree, path).properties.append(data)
        elif cmd == "set_animation":
            find_node(self.tree, path).animation = data
        elif cmd == "delete":
            if len(path) > 0:
                parent = find_node(self.tree, path[:-1])
                child = path[-1]
                if child in parent:
                    del parent[child]
            else:
                self.tree = SceneTree()
        self.zmq_socket.send(b"ok")

    def forward_to_websockets(self, frames):
        cmd, path, data = frames
        for websocket in self.websocket_pool:
            websocket.write_message(data, binary=True)

    def setup_zmq(self, url):
        zmq_socket = self.context.socket(zmq.REP)
        zmq_socket.bind(url)
        zmq_stream = ZMQStream(zmq_socket)
        zmq_stream.on_recv(self.handle_zmq)
        return zmq_socket, zmq_stream, url

    def send_scene(self, websocket):
        for node in walk(self.tree):
            if node.object is not None:
                websocket.write_message(node.object, binary=True)
            for p in node.properties:
                websocket.write_message(p, binary=True)
            if node.transform is not None:
                websocket.write_message(node.transform, binary=True)
            if node.animation is not None:
                websocket.write_message(node.animation, binary=True)

    def run(self):
        self.ioloop.start()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Serve the MeshCat HTML files and listen for ZeroMQ commands")
    parser.add_argument("--zmq-url", "-z", type=str, nargs="?", default=None)
    parser.add_argument("--websocket-port", "-wp", type=str, nargs="?", default=None)
    args = parser.parse_args()
    bridge = ZMQWebSocketBridge(zmq_url=args.zmq_url, websocket_port=args.websocket_port)
    print(bridge)
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
