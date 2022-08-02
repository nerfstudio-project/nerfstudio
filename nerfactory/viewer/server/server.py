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

"""Server bridge to faciliate interactions between python backend and javascript front end"""


import sys
from typing import Callable, List, Optional, Tuple

import msgpack
import msgpack_numpy
import tornado.gen
import tornado.ioloop
import tornado.web
import tornado.websocket
import umsgpack
import zmq
import zmq.eventloop.ioloop
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from zmq.eventloop.zmqstream import ZMQStream

from nerfactory.viewer.server.state.node import find_node, get_tree, walk
from nerfactory.viewer.server.state.state_node import StateNode
from nerfactory.viewer.server.video_stream import SingleFrameStreamTrack

MAX_ATTEMPTS = 1000
DEFAULT_ZMQ_METHOD = "tcp"
DEFAULT_ZMQ_PORT = 6000
DEFAULT_WEBSOCKET_PORT = 8051


def find_available_port(func: Callable, default_port: int, max_attempts: int = MAX_ATTEMPTS, **kwargs) -> None:
    """Finds and attempts to connect to a port

    Args:
        func: function used on connecting to port
        default_port: the default port
        max_attempts: max number of attempts to try connection. Defaults to MAX_ATTEMPTS.
    """
    for i in range(max_attempts):
        port = default_port + i
        try:
            return func(port, **kwargs), port
        except (OSError, zmq.error.ZMQError):
            print(f"Port: {port:d} in use, trying another...", file=sys.stderr)
        except Exception as e:
            print(type(e))
            raise
    raise (
        Exception(f"Could not find an available port in the range: [{default_port:d}, {max_attempts + default_port:d})")
    )


def force_codec(pc: RTCPeerConnection, sender: RTCRtpSender, forced_codec: str) -> None:
    """Sets the codec preferences on a connection between sender and reciever

    Args:
        pc: peer connection point
        sender: sender that will send to connection point
        forced_codec: codec to set
    """
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences([codec for codec in codecs if codec.mimeType == forced_codec])


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
            if m["path"] != "renderingState/camera":
                # dispatch to the websockets if not a camera update!
                # TODO(ethan): handle the camera properly!
                # TODO: but don't update the current websocket
                command = {"type": "write", "path": m["path"], "data": m["data"]}
                packed_data = umsgpack.packb(command)
                frames = ["write".encode("utf-8"), m["path"].encode("utf-8"), packed_data]
                self.bridge.forward_to_websockets(frames, websocket_to_skip=self)
        elif type_ == "read":
            # reads and returns the data
            data = find_node(self.bridge.state_tree, path).data
            self.write_message(data, binary=True)
        elif type_ == "offer":
            # returns the description to for WebRTC to the specific websocket connection
            offer = RTCSessionDescription(m["data"]["sdp"], m["data"]["type"])

            pc = RTCPeerConnection()
            self.bridge.pcs.add(pc)

            video = SingleFrameStreamTrack()
            self.bridge.video_tracks.add(video)
            _ = pc.addTrack(video)
            # TODO: do something with the codec
            # video_sender = pc.addTrack(video)
            # force_codec(pc, video_sender, video_codec)

            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            cmd_data = {
                "type": "answer",
                "path": "",
                "data": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            }
            data = umsgpack.packb(cmd_data)
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
        zmq_url: zmq url to connect to. Defaults to None.
        host: host of server. Defaults to "127.0.0.1".
        websocket_port: websocket port to connect to. Defaults to None.
    """

    context = zmq.Context()

    def __init__(self, zmq_url: Optional[str] = None, host: str = "127.0.0.1", websocket_port: Optional[int] = None):
        self.host = host
        self.websocket_pool = set()
        self.app = self.make_app()
        self.ioloop = tornado.ioloop.IOLoop.current()
        self.pcs = set()
        self.video_tracks = set()

        if zmq_url is None:

            def f(port):
                return self.setup_zmq(f"{DEFAULT_ZMQ_METHOD:s}://{self.host:s}:{port:d}")

            (self.zmq_socket, self.zmq_stream, self.zmq_url), _ = find_available_port(f, DEFAULT_ZMQ_PORT)
        else:
            self.zmq_socket, self.zmq_stream, self.zmq_url = self.setup_zmq(zmq_url)

        listen_kwargs = {}

        if websocket_port is None:
            _, self.websocket_port = find_available_port(self.app.listen, DEFAULT_WEBSOCKET_PORT, **listen_kwargs)
        else:
            self.app.listen(websocket_port, **listen_kwargs)
            self.websocket_port = websocket_port

        self.state_tree = get_tree(StateNode)

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name} using zmq_url={self.zmq_url} and websocket_port={self.websocket_port}"

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
        elif type_ == "set_image":
            image = msgpack.unpackb(
                data, object_hook=msgpack_numpy.decode, use_list=False, max_bin_len=50000000, raw=False
            )
            for video_track in self.video_tracks:
                video_track.put_frame(image)
            self.zmq_socket.send(umsgpack.packb(b"ok"))
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
        print("Sending entire scene state on websocket connect (i.e,. a page refresh).")
        for path, node in walk("", self.state_tree):
            if node.data is not None:
                command = {"type": "write", "path": path, "data": node.data}
                websocket.write_message(umsgpack.packb(command), binary=True)

    def run(self):
        """starts and runs the websocket bridge"""
        self.ioloop.start()


def start_server_as_subprocess(zmq_url=None):
    """Starts the ZMQWebSocketBridge server as a subprocess."""
    raise NotImplementedError()
