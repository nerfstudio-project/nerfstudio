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
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Union

from rich.console import Console

from nerfstudio.viewer.server import server

CONSOLE = Console()


def run_viewer_bridge_server_as_subprocess(zmq_port: int, websocket_port: int, log_filename: Union[str, None]):
    """Runs the viewer bridge server as a subprocess.

    Args:
        zmq_port: Port to use for the ZMQ server.
        websocket_port: Port to use for the websocket server.
        log_filename: Filename to use for the log file. If None, no log file is created.

    Returns:
        None
    """
    args = [sys.executable, "-u", "-m", server.__name__]
    args.append("--zmq-port")
    args.append(str(zmq_port))
    args.append("--websocket-port")
    args.append(str(websocket_port))
    # supress output if no log filename is specified
    logfile = open(  # pylint: disable=consider-using-with
        log_filename if log_filename else os.devnull, "w", encoding="utf8"
    )
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        args, stdout=logfile, stderr=logfile, start_new_session=True
    )

    def cleanup(process):
        process.kill()
        process.wait()

    def poll_process():
        """
        Continually check to see if the viewer bridge server process is still running and has not failed.
        If it fails, alert the user and exit the entire program.
        """
        while process.poll() is None:
            time.sleep(0.5)
        string = f"\nThe viewer bridge server subprocess failed. Please check the log file {log_filename}.\n"
        string += (
            "You likely have to modify --viewer.zmq-port and/or --viewer.websocket-port in the "
            "config to avoid conflicting ports.\n"
        )
        string += "Try modifying --viewer.zmq-port 6000 --viewer.websocket-port 7007\n"
        CONSOLE.print(f"[bold red]{string}")
        cleanup(process)
        # This exists the entire program. sys.exit() will only kill the thread that this runs in.
        os.kill(os.getpid(), signal.SIGKILL)

    # continually check to see if the process stopped
    t1 = threading.Thread(target=poll_process)
    t1.daemon = True
    t1.start()
    atexit.register(cleanup, process)
