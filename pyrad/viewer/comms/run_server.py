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
from pyrad.viewer.comms.tree import SceneTree, find_node, walk
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
DEFAULT_PORT = 8051

MESHCAT_COMMANDS = ["set_transform", "set_object", "delete", "set_property", "set_animation"]


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
        # self.bridge.send_scene(self)

    def on_message(self, message):
        pass

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


class StaticFileHandlerNoCache(tornado.web.StaticFileHandler):
    """Ensures static files do not get cached.

    Taken from: https://stackoverflow.com/a/18879658/7829525
    """

    def set_extra_headers(self, path):
        # Disable cache
        self.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")


class ZMQWebSocketBridge(object):
    context = zmq.Context()

    def __init__(self, zmq_url=None, host="0.0.0.0", zmq_port=None, port=None):
        self.host = host
        self.websocket_pool = set()
        self.app = self.make_app()
        self.ioloop = tornado.ioloop.IOLoop.current()

        if zmq_url is None:

            def f(port):
                return self.setup_zmq("{:s}://{:s}:{:d}".format(DEFAULT_ZMQ_METHOD, self.host, port))

            # TODO(ethan): handle the port setting better
            (self.zmq_socket, self.zmq_stream, self.zmq_url), _ = find_available_port(
                f, zmq_port if zmq_port is not None else DEFAULT_ZMQ_PORT
            )
        else:
            self.zmq_socket, self.zmq_stream, self.zmq_url = self.setup_zmq(zmq_url)

        listen_kwargs = {}
        self.app.listen(DEFAULT_PORT, **listen_kwargs)
        print("using {}".format(DEFAULT_PORT))
        self.tree = SceneTree()

    def make_app(self):
        return tornado.web.Application([(r"/", WebSocketHandler, {"bridge": self})])

    def handle_zmq(self, frames):
        # print(frames)
        self.forward_to_websockets(frames)
        self.zmq_socket.send(b"ok")
        # print(frames)
        # cmd = frames[0].decode("utf-8")
        # if cmd in MESHCAT_COMMANDS:
        #     if len(frames) != 3:
        #         self.zmq_socket.send(b"error: expected 3 frames")
        #         return
        #     path = list(filter(lambda x: len(x) > 0, frames[1].decode("utf-8").split("/")))
        #     data = frames[2]
        #     self.forward_to_websockets(frames)
        #     if cmd == "set_transform":
        #         find_node(self.tree, path).transform = data
        #     elif cmd == "set_object":
        #         find_node(self.tree, path).object = data
        #         find_node(self.tree, path).properties = []
        #     elif cmd == "set_property":
        #         find_node(self.tree, path).properties.append(data)
        #     elif cmd == "set_animation":
        #         find_node(self.tree, path).animation = data
        #     elif cmd == "delete":
        #         if len(path) > 0:
        #             parent = find_node(self.tree, path[:-1])
        #             child = path[-1]
        #             if child in parent:
        #                 del parent[child]
        #         else:
        #             self.tree = SceneTree()
        #     self.zmq_socket.send(b"ok")
        # else:
        #     self.zmq_socket.send(b"error: unrecognized comand")

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
    import sys
    import webbrowser

    parser = argparse.ArgumentParser(description="Serve the MeshCat HTML files and listen for ZeroMQ commands")
    parser.add_argument("--zmq-url", "-z", type=str, nargs="?", default=None)
    parser.add_argument("--open", "-o", action="store_true")
    parser.add_argument("--certfile", type=str, default=None)
    parser.add_argument("--keyfile", type=str, default=None)
    parser.add_argument(
        "--ngrok_http_tunnel",
        action="store_true",
        help="""    
ngrok is a service for creating a public URL from your local machine, which 
is very useful if you would like to make your meshcat server public.""",
    )
    results = parser.parse_args()
    bridge = ZMQWebSocketBridge(zmq_url=results.zmq_url)
    print("zmq_url={:s}".format(bridge.zmq_url))
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
