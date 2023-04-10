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

"""
Code to use the viewer as a subprocess.
"""

import socket

from rich.console import Console

CONSOLE = Console()


def is_port_open(port: int):
    """Returns True if the port is open.

    Args:
        port: Port to check.

    Returns:
        True if the port is open, False otherwise.
    """
    try:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _ = sock.bind(("", port))
        sock.close()
        return True
    except OSError:
        return False


def get_free_port(default_port: int = None):
    """Returns a free port on the local machine. Try to use default_port is possible.

    Args:
        default_port: Port to try to use.

    Returns:
        A free port on the local machine.
    """
    if default_port:
        if is_port_open(default_port):
            return default_port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    return port
