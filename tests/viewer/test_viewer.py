"""
Test communications with the viewer.
"""

import time

from nerfactory.viewer.server.subprocess import run_viewer_bridge_server_as_subprocess


def test_run_subprocess():
    """Test running the viewer bridge server as a subprocess."""
    zmq_port = 6000 + 10  # don't use the default ports to avoid conflicting with current training jobs
    websocket_port = 7007 + 10
    print("Starting the viewer bridge server as a subprocess.")
    run_viewer_bridge_server_as_subprocess(zmq_port, websocket_port)
    print("Started the viewer bridge server as a subprocess.")
    # do something will the viewer bridge server runs
    time.sleep(1)


if __name__ == "__main__":
    test_run_subprocess()
    print("Tests passed.")
