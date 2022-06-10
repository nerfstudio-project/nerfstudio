from __future__ import absolute_import, division, print_function

import atexit
import base64
import os
import re
import sys
import subprocess
import multiprocessing

if sys.version_info >= (3, 0):
    ADDRESS_IN_USE_ERROR = OSError
else:
    import socket

    ADDRESS_IN_USE_ERROR = socket.error

import tornado.web
import tornado.ioloop
import tornado.websocket
import tornado.gen

import zmq
import zmq.eventloop.ioloop
from zmq.eventloop.zmqstream import ZMQStream

from .tree import SceneTree, walk, find_node


def capture(pattern, s):
    match = re.match(pattern, s)
    if not match:
        raise ValueError("Could not match {:s} with pattern {:s}".format(s, pattern))
    else:
        return match.groups()[0]


def match_zmq_url(line):
    return capture(r"^zmq_url=(.*)$", line)


def match_web_url(line):
    return capture(r"^web_url=(.*)$", line)


def start_zmq_server_as_subprocess(zmq_url=None, server_args=[]):
    """
    Starts the ZMQ server as a subprocess, passing *args through popen.
    Optional Keyword Arguments:
        zmq_url  
    """
    # Need -u for unbuffered output: https://stackoverflow.com/a/25572491
    args = [sys.executable, "-u", "-m", "meshcat.servers.zmqserver"]
    if zmq_url is not None:
        args.append("--zmq-url")
        args.append(zmq_url)
    if server_args:
        args.append(*server_args)
    # Note: Pass PYTHONPATH to be robust to workflows like Google Colab,
    # where meshcat might have been added directly via sys.path.append.
    env = {'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}
    kwargs = {
        'stdout': subprocess.PIPE,
        'env': env
    }
    # Use start_new_session if it's available. Without it, in jupyter the server
    # goes down when we cancel execution of any cell in the notebook.
    if sys.version_info.major >= 3:
        kwargs['start_new_session'] = True
    server_proc = subprocess.Popen(args, **kwargs)

    line = ""
    while "zmq_url" not in line:
        line = server_proc.stdout.readline().strip().decode("utf-8")
    zmq_url = match_zmq_url(line)
    web_url = match_web_url(server_proc.stdout.readline().strip().decode("utf-8"))

    print(zmq_url)
    print(web_url)

    def cleanup(server_proc):
        server_proc.kill()
        server_proc.wait()

    atexit.register(cleanup, server_proc)
    return server_proc, zmq_url, web_url


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

VIEWER_ROOT = os.path.join(os.path.dirname(__file__), "..", "viewer", "dist")
VIEWER_HTML = "index.html"

DEFAULT_FILESERVER_PORT = 7000
MAX_ATTEMPTS = 1000
DEFAULT_ZMQ_METHOD = "tcp"
DEFAULT_ZMQ_PORT = 6000

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
        raise (Exception("Could not find an available port in the range: [{:d}, {:d})".format(default_port,
                                                                                              max_attempts + default_port)))


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        self.bridge = kwargs.pop("bridge")
        super(WebSocketHandler, self).__init__(*args, **kwargs)

    def open(self):
        self.bridge.websocket_pool.add(self)
        print("opened:", self, file=sys.stderr)
        self.bridge.send_scene(self)

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
    """.format(base64.b64encode(data).decode("utf-8"))


class StaticFileHandlerNoCache(tornado.web.StaticFileHandler):
    """Ensures static files do not get cached.

    Taken from: https://stackoverflow.com/a/18879658/7829525
    """

    def set_extra_headers(self, path):
        # Disable cache
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')


class ZMQWebSocketBridge(object):
    context = zmq.Context()

    def __init__(self,
                 zmq_url=None,
                 host="127.0.0.1",
                 port=None,
                 certfile=None,
                 keyfile=None,
                 ngrok_http_tunnel=False,
                 zmq_port=None):
        self.host = host
        self.websocket_pool = set()
        self.app = self.make_app()
        self.ioloop = tornado.ioloop.IOLoop.current()

        if zmq_url is None:
            def f(port):
                return self.setup_zmq("{:s}://{:s}:{:d}".format(DEFAULT_ZMQ_METHOD, self.host, port))

            # TODO(ethan): handle the port setting better
            (self.zmq_socket, self.zmq_stream, self.zmq_url), _ = find_available_port(f,
                                                                                      zmq_port if zmq_port is not None else DEFAULT_ZMQ_PORT)
        else:
            self.zmq_socket, self.zmq_stream, self.zmq_url = self.setup_zmq(zmq_url)

        protocol = "http:"
        listen_kwargs = {}
        if certfile is not None or keyfile is not None:
            if certfile is None:
                raise (Exception("You must supply a certfile if you supply a keyfile"))
            if keyfile is None:
                raise (Exception("You must supply a keyfile if you supply a certfile"))

            listen_kwargs["ssl_options"] = {"certfile": certfile,
                                            "keyfile": keyfile}
            protocol = "https:"

        if port is None:
            _, self.fileserver_port = find_available_port(self.app.listen, DEFAULT_FILESERVER_PORT, **listen_kwargs)
        else:
            self.app.listen(port, **listen_kwargs)
            self.fileserver_port = port
        self.web_url = "{protocol}//{host}:{port}/static/".format(
            protocol=protocol, host=self.host, port=self.fileserver_port)

        # Note: The (significant) advantage of putting this in here is not only
        # so that the workflow is convenient, but also so that the server
        # administers the public web_url when clients ask for it.
        if ngrok_http_tunnel:
            if protocol == "https:":
                # TODO(russt): Consider plumbing ngrok auth through here for
                # someone who has paid for ngrok and wants to use https.
                raise (Exception('The free version of ngrok does not support https'))

            # Conditionally import pyngrok
            try:
                import pyngrok.conf
                import pyngrok.ngrok

                kwargs = {}
                # Use start_new_session if it's available. Without it, in
                # jupyter the server goes down when we cancel execution of any
                # cell in the notebook.
                if sys.version_info.major >= 3:
                    kwargs['start_new_session'] = True
                config = pyngrok.conf.PyngrokConfig(**kwargs)
                self.web_url = pyngrok.ngrok.connect(self.fileserver_port, "http", pyngrok_config=config) + "/static/"
                print("\n")  # ensure any pyngrok output is properly terminated.

                def cleanup():
                    pyngrok.ngrok.kill()

                atexit.register(cleanup)

            except ImportError as e:
                if "pyngrok" in e.__class__.__name__:
                    raise (Exception("You must install pyngrok (e.g. via `pip install pyngrok`)."))

        self.tree = SceneTree()

    def make_app(self):
        return tornado.web.Application([
            (r"/static/(.*)", StaticFileHandlerNoCache, {"path": VIEWER_ROOT, "default_filename": VIEWER_HTML}),
            (r"/", WebSocketHandler, {"bridge": self})
        ])

    def wait_for_websockets(self):
        if len(self.websocket_pool) > 0:
            self.zmq_socket.send(b"ok")
        else:
            self.ioloop.call_later(0.1, self.wait_for_websockets)

    def handle_zmq(self, frames):
        print(frames)
        cmd = frames[0].decode("utf-8")
        if cmd == "url":
            self.zmq_socket.send(self.web_url.encode("utf-8"))
        elif cmd == "wait":
            self.ioloop.add_callback(self.wait_for_websockets)
        elif cmd in MESHCAT_COMMANDS:
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
        elif cmd == "get_scene":
            # when the server gets this command, return the tree
            # as a series of msgpack-backed binary blobs
            drawing_commands = ""
            for node in walk(self.tree):
                if node.object is not None:
                    drawing_commands += create_command(node.object)
                for p in node.properties:
                    drawing_commands += create_command(p)
                if node.transform is not None:
                    drawing_commands += create_command(node.transform)
                if node.animation is not None:
                    drawing_commands += create_command(node.animation)

            # now that we have the drawing commands, generate the full
            # HTML that we want to generate, including the javascript assets
            mainminjs_path = os.path.join(VIEWER_ROOT, "main.min.js")
            mainminjs_src = ""
            with open(mainminjs_path, "r") as f:
                mainminjs_src = f.readlines()
            mainminjs_src = "".join(mainminjs_src)

            html = """
                <!DOCTYPE html>
                <html>
                    <head> <meta charset=utf-8> <title>MeshCat</title> </head>
                    <body>
                        <div id="meshcat-pane">
                        </div>
                        <script>
                            {mainminjs}
                        </script>
                        <script>
                            var viewer = new MeshCat.Viewer(document.getElementById("meshcat-pane"));
                            {commands}
                        </script>
                         <style>
                            body {{margin: 0; }}
                            #meshcat-pane {{
                                width: 100vw;
                                height: 100vh;
                                overflow: hidden;
                            }}
                        </style>
                        <script id="embedded-json"></script>
                    </body>
                </html>
            """.format(mainminjs=mainminjs_src, commands=drawing_commands)
            self.zmq_socket.send(html.encode('utf-8'))
        else:
            self.zmq_socket.send(b"error: unrecognized comand")

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
    parser.add_argument('--zmq-url', '-z', type=str, nargs="?", default=None)
    parser.add_argument('--open', '-o', action="store_true")
    parser.add_argument('--certfile', type=str, default=None)
    parser.add_argument('--keyfile', type=str, default=None)
    parser.add_argument('--ngrok_http_tunnel', action="store_true", help="""    
ngrok is a service for creating a public URL from your local machine, which 
is very useful if you would like to make your meshcat server public.""")
    results = parser.parse_args()
    bridge = ZMQWebSocketBridge(zmq_url=results.zmq_url,
                                certfile=results.certfile,
                                keyfile=results.keyfile,
                                ngrok_http_tunnel=results.ngrok_http_tunnel)
    print("zmq_url={:s}".format(bridge.zmq_url))
    print("web_url={:s}".format(bridge.web_url))
    if results.open:
        webbrowser.open(bridge.web_url, new=2)

    try:
        bridge.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
