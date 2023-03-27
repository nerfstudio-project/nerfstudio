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

"""Server bridge to facilitate interactions between python backend and javascript front end"""

import sys
from typing import List, Optional, Tuple

import tornado.gen
import tornado.ioloop
import tornado.web
import tornado.websocket
import tyro
import umsgpack
import zmq
import zmq.eventloop.ioloop
from pyngrok import ngrok
from zmq.eventloop.zmqstream import ZMQStream

from nerfstudio.viewer.server.state.node import find_node, get_tree, walk
from nerfstudio.viewer.server.state.state_node import StateNode


class WebSocketHandler(tornado.websocket.WebSocketHandler):  # pylint: disable=abstract-method
    """Tornado websocket handler for receiving and sending commands from/to the viewer."""

    def __init__(self, *args, **kwargs):
        self.bridge = kwargs.pop("bridge")
        super().__init__(*args, **kwargs)

    def check_origin(self, origin):
        """This disables CORS."""
        return True

    def open(self, *args: str, **kwargs: str):
        """open websocket bridge"""
        self.bridge.websocket_pool.add(self)
        print("opened:", self, file=sys.stderr)
        self.bridge.send_scene(self)

    async def on_message(self, message: bytearray):  # pylint: disable=invalid-overridden-method
        """On reception of message from the websocket,
        parses the message and calls the appropriate function based on the type of command

        Args:
            message: byte message to parse
        """
        data = message
        m = umsgpack.unpackb(message)
        type_ = m["type"]
        path = list(filter(lambda x: len(x) > 0, m["path"].split("/")))

        if type_ == "write":
            # writes the data coming from the websocket
            find_node(self.bridge.state_tree, path).data = m["data"]
            command = {"type": "write", "path": m["path"], "data": m["data"]}
            packed_data = umsgpack.packb(command)
            frames = ["write".encode("utf-8"), m["path"].encode("utf-8"), packed_data]
            self.bridge.forward_to_websockets(frames, websocket_to_skip=self)
        elif type_ == "read":
            # reads and returns the data
            data = find_node(self.bridge.state_tree, path).data
            self.write_message(data, binary=True)
        else:
            cmd_data = {
                "type": "error",
                "path": "",
                "data": {"error": "Unknown command type: " + type_},
            }
            data = umsgpack.packb(cmd_data)
            self.write_message(data, binary=True)

    def on_close(self):
        self.bridge.websocket_pool.remove(self)
        print("closed:", self, file=sys.stderr)


class ZMQWebSocketBridge:
    """ZMQ web socket bridge class

    Args:
        zmq_port: zmq port to connect to. Defaults to None.
        websocket_port: websocket port to connect to. Defaults to None.
    """

    context = zmq.Context()  # pylint: disable=abstract-class-instantiated

    def __init__(self, zmq_port: int, websocket_port: int, ip_address: str):
        self.zmq_port = zmq_port
        self.websocket_pool = set()
        self.app = self.make_app()
        self.ioloop = tornado.ioloop.IOLoop.current()

        # zmq
        zmq_url = f"tcp://{ip_address}:{self.zmq_port:d}"
        self.zmq_socket, self.zmq_stream, self.zmq_url = self.setup_zmq(zmq_url)

        # websocket
        listen_kwargs = {"address": "0.0.0.0"}
        self.app.listen(websocket_port, **listen_kwargs)
        self.websocket_port = websocket_port
        self.websocket_url = f"0.0.0.0:{self.websocket_port}"

        # state tree
        self.state_tree = get_tree(StateNode)

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name} using zmq_port="{self.zmq_port}" and websocket_port="{self.websocket_port}"'

    def make_app(self):
        """Create a tornado application for the websocket server."""
        return tornado.web.Application([(r"/", WebSocketHandler, {"bridge": self})])

    def handle_zmq(self, frames: List[bytes]):
        """Switch function that places commands in tree based on websocket command

        Args:
            frames: the list containing command + object to be placed in tree
        """
        if len(frames) != 3:
            self.zmq_socket.send(b"error: expected 3 frames")
            return
        type_ = frames[0].decode("utf-8")
        path = list(filter(lambda x: len(x) > 0, frames[1].decode("utf-8").split("/")))
        data = frames[2]

        if type_ == "write":
            # TODO: use state_tree name
            unpacked_data = umsgpack.unpackb(data)
            find_node(self.state_tree, path).data = unpacked_data["data"]
            self.forward_to_websockets(frames)
            self.zmq_socket.send(umsgpack.packb(b"ok"))
        elif type_ == "read":
            # walk the node from the specified to get the full state dictionary
            # TODO(ethan): handle the "data" key...
            read_data = find_node(self.state_tree, path).data
            self.zmq_socket.send(umsgpack.packb(read_data))
        else:
            self.zmq_socket.send(umsgpack.packb(b"error: unknown command"))

    def forward_to_websockets(
        self, frames: Tuple[str, str, bytes], websocket_to_skip: Optional[WebSocketHandler] = None
    ):
        """Forward a zmq message to all websockets.

        Args:
            frames: byte messages to be sent over
        """
        _, _, data = frames  # cmd, path, data
        for websocket in self.websocket_pool:
            if websocket_to_skip and websocket == websocket_to_skip:
                pass
            else:
                websocket.write_message(data, binary=True)

    def setup_zmq(self, url: str):
        """Setup a zmq socket and connect it to the given url.

        Args:
            url: point of connection
        """
        zmq_socket = self.context.socket(zmq.REP)  # pylint: disable=no-member
        zmq_socket.bind(url)
        zmq_stream = ZMQStream(zmq_socket)
        zmq_stream.on_recv(self.handle_zmq)
        return zmq_socket, zmq_stream, url

    def send_scene(self, websocket: WebSocketHandler):
        """Sends entire tree of information over the specified websocket

        Args:
            websocket: websocket to send information over
        """
        print("Sending entire scene state due to websocket connection established.")
        for path, node in walk("", self.state_tree):
            if node.data is not None:
                command = {"type": "write", "path": path, "data": node.data}
                websocket.write_message(umsgpack.packb(command), binary=True)

    def run(self):
        """starts and runs the websocket bridge"""
        self.ioloop.start()


def run_viewer_bridge_server(
    zmq_port: int = 6000, websocket_port: int = 7007, ip_address: str = "127.0.0.1", use_ngrok: bool = False
):
    """Run the viewer bridge server.

    Args:
        zmq_port: port to use for zmq
        websocket_port: port to use for websocket
        ip_address: host to connect to
        use_ngrok: whether to use ngrok to expose the zmq port
    """

    # whether to launch pyngrok or not
    if use_ngrok:
        # Open a HTTP tunnel on the default port 80
        # <NgrokTunnel: "http://<public_sub>.ngrok.io" -> "http://localhost:80">
        http_tunnel = ngrok.connect(addr=str(zmq_port), proto="tcp")
        print(http_tunnel)

    bridge = ZMQWebSocketBridge(zmq_port=zmq_port, websocket_port=websocket_port, ip_address=ip_address)
    print(bridge)
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass


def entrypoint():
    """The main entrypoint."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(run_viewer_bridge_server)


if __name__ == "__main__":
    entrypoint()
