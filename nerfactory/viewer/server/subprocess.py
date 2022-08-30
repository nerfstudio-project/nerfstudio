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

"""
Code to use the viewer as a subprocess.
"""

import atexit
import subprocess
import sys

from nerfactory.viewer.server import server


def run_viewer_bridge_server_as_subprocess(
    zmq_port: int, websocket_port: int, log_filename="viewer_bridge_server_log.txt"
):
    """Runs the viewer bridge server as a subprocess.

    Args:
        zmq_port: Port to use for the ZMQ server.
        websocket_port: Port to use for the websocket server.

    Returns:
        None
    """
    args = [sys.executable, "-u", "-m", server.__name__]
    args.append("--zmq-port")
    args.append(str(zmq_port))
    args.append("--websocket-port")
    args.append(str(websocket_port))
    with open(log_filename, "w", encoding="utf8") as logfile:
        with subprocess.Popen(args, stdout=logfile, stderr=logfile, start_new_session=True) as process:

            def cleanup(process):
                process.kill()
                process.wait()

            atexit.register(cleanup, process)
